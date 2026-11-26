import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch as th

EPYMARL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(EPYMARL_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from envs.maca_wrapper import MaCAWrapper
from llm_marl.llm_client import CardLLMError
from llm_marl.llm_config import load_llm_config
from llm_marl.opponent_features import OpponentFeatureExtractor
from llm_marl.opponent_profile import (
    OpponentProfileGenerator,
    OpponentProfileStore,
    normalize_opponent_style,
)
from llm_marl.runtime import CardRuntime
from modules.critics.opponent_aux import OpponentAuxHead, masked_aux_loss
from modules.mixers.oa_qmix import OAQMixer


class OAQMixIntegrationTest(unittest.TestCase):
    def test_oa_qmixer_forward_backward_2layer(self):
        args = SimpleNamespace(
            n_agents=3,
            state_shape=61,
            opp_profile_dim=8,
            mixing_embed_dim=32,
            hypernet_layers=2,
            hypernet_embed=64,
        )
        mixer = OAQMixer(args)
        self.assertEqual(mixer.critic_input_dim, 69)
        agent_qs = th.randn(4, 10, 3, requires_grad=True)
        states = th.randn(4, 10, 61)
        opp_profile = th.rand(4, 10, 8)
        q_tot = mixer(agent_qs, states, opp_profile)
        self.assertEqual(tuple(q_tot.shape), (4, 10, 1))
        q_tot.mean().backward()
        self.assertIsNotNone(agent_qs.grad)
        self.assertGreater(agent_qs.grad.abs().sum().item(), 0.0)

    def test_aux_head_masked_loss(self):
        head = OpponentAuxHead(input_dim=72, hidden_dim=32, style_classes=4, n_agents=5)
        agent_repr = th.randn(6, 64, requires_grad=True)
        opp_profile = th.rand(6, 8)
        outputs = head(agent_repr, opp_profile)
        labels = {
            "style": th.tensor([1, 2, 3, 0, 1, 2], dtype=th.long),
            "focus": th.zeros(6, dtype=th.long),
            "danger": th.zeros(6),
        }
        masks = {
            "style": th.ones(6, dtype=th.bool),
            "focus": th.zeros(6, dtype=th.bool),
            "danger": th.zeros(6, dtype=th.bool),
        }
        loss, metrics = masked_aux_loss(outputs, labels, masks)
        loss.backward()
        self.assertGreater(loss.item(), 0.0)
        self.assertGreater(metrics["opponent_aux_loss"], 0.0)
        self.assertIsNotNone(agent_repr.grad)

    def test_maca_card_context_and_feature_extraction_3v3_and_5v5(self):
        extractor = OpponentFeatureExtractor()
        for map_name, n_agents in (("3v3", 3), ("5v5", 5)):
            env = MaCAWrapper(map_name=map_name, seed=42)
            try:
                env.reset()
                desc = env.get_card_env_descriptor()
                self.assertEqual(desc["env_name"], "maca")
                self.assertEqual(desc["n_agents"], n_agents)

                prev_ctx = env.get_card_context()
                steps = []
                for step_i in range(60):
                    avail = env.get_avail_actions()
                    actions = [
                        next(i for i, flag in enumerate(av) if flag == 1)
                        for av in avail
                    ]
                    _, rew, term, trunc, info = env.step(actions)
                    curr_ctx = env.get_card_context()
                    steps.append(
                        {
                            "step_index": step_i,
                            "actions": actions,
                            "env_reward": rew,
                            "prev_context": prev_ctx,
                            "curr_context": curr_ctx,
                            "env_info": info,
                        }
                    )
                    prev_ctx = curr_ctx
                    if term or trunc:
                        break

                feats = extractor.extract_episode({"steps": steps})
                self.assertEqual(feats["diagnostics"], [])
                self.assertTrue(0.0 <= feats["engage_distance_proxy"] <= 1.0)
                self.assertTrue(0.0 <= feats["formation_spread_proxy"] <= 1.0)
                self.assertTrue(0.0 <= feats["focus_fire_rate"] <= 1.0)
            finally:
                env.close()

    def test_heuristic_profile_8d_vector(self):
        summary = {
            "episodes": 8,
            "means": {
                "focus_fire_rate": 0.72,
                "low_hp_chase_ratio": 0.35,
                "target_switch_rate": 0.25,
                "engage_distance_proxy": 0.41,
                "formation_spread_proxy": 0.18,
                "damage_burst_window_proxy": 1.0,
            },
            "diagnostics": [],
        }
        profile = OpponentProfileGenerator(vector_dim=8, mode="heuristic").build(
            summary, ["ep1"]
        )
        self.assertEqual(profile.opponent_style, "aggressive_focus_fire")
        self.assertEqual(len(profile.behavior_vector), 8)
        self.assertEqual(profile.behavior_vector[:4], [1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(profile.behavior_vector[4], 0.72)
        self.assertAlmostEqual(profile.behavior_vector[7], 0.41)
        self.assertEqual(normalize_opponent_style("deathball", [0.4] * 8), "balanced")

    def test_llm_oa_qmix_startup_validation_and_mocked_api_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_env = os.path.join(tmpdir, ".env.llm.bad")
            with open(bad_env, "w", encoding="utf-8") as f:
                f.write(
                    "LLM_PROVIDER=openai_compatible\n"
                    "LLM_BASE_URL=https://api.example.com/v1\n"
                    "LLM_API_KEY=YOUR_API_KEY_HERE\n"
                    "LLM_MODEL=YOUR_MODEL_NAME_HERE\n"
                )

            args_bad = SimpleNamespace(
                learner="q_learner",
                mixer="oa_qmix",
                runner="episode",
                common_reward=True,
                local_results_path=tmpdir,
                unique_token="bad_llm_test",
                card={
                    "llm": {"env_file": bad_env, "model": ""},
                    "opponent": {
                        "enabled": True,
                        "profile": {"enabled": True, "mode": "llm", "vector_dim": 8},
                        "critic": {"enabled": True},
                    },
                },
            )
            logger = SimpleNamespace(
                log_stat=lambda *a, **k: None,
                console_logger=SimpleNamespace(warning=lambda *a, **k: None),
            )
            with self.assertRaises(CardLLMError):
                CardRuntime(
                    args_bad,
                    logger,
                    {"n_agents": 3, "n_actions": 15, "episode_limit": 200},
                )

            good_env = os.path.join(tmpdir, ".env.llm.good")
            with open(good_env, "w", encoding="utf-8") as f:
                f.write(
                    "LLM_PROVIDER=openai_compatible\n"
                    "LLM_BASE_URL=https://api.deepseek.com/v1\n"
                    "LLM_API_KEY=sk-test-valid-key\n"
                    "LLM_MODEL=deepseek-chat\n"
                    "LLM_MAX_RETRIES=0\n"
                )
            cfg = load_llm_config(good_env)
            self.assertEqual(cfg.model, "deepseek-chat")

            args_good = SimpleNamespace(
                learner="q_learner",
                mixer="oa_qmix",
                runner="episode",
                common_reward=True,
                local_results_path=tmpdir,
                unique_token="good_llm_test",
                card={
                    "save_episodes": False,
                    "llm": {"env_file": good_env, "model": ""},
                    "opponent": {
                        "enabled": True,
                        "feature_extraction": {"enabled": True},
                        "profile": {
                            "enabled": True,
                            "mode": "llm",
                            "vector_dim": 8,
                            "min_episodes": 1,
                            "refresh_episodes": 1,
                            "record_calls": True,
                        },
                        "critic": {"enabled": True},
                    },
                },
            )

            class FakeResponse:
                status_code = 200

                def raise_for_status(self):
                    return None

                def json(self):
                    content = json.dumps(
                        {
                            "opponent_style": "aggressive_focus_fire",
                            "behavior_vector": [
                                0.78,
                                0.62,
                                0.25,
                                0.35,
                                0.60,
                                0.55,
                                0.48,
                                0.15,
                            ],
                            "counter_strategy": "maintain mutual radar support and focus nearest attacker",
                            "confidence": 0.85,
                            "profile_version": "magpie-profile-v1",
                        }
                    )
                    return {
                        "choices": [{"message": {"content": content}}],
                        "usage": {
                            "prompt_tokens": 220,
                            "completion_tokens": 65,
                            "total_tokens": 285,
                        },
                    }

            with patch("llm_marl.llm_client.requests.post", return_value=FakeResponse()) as mock_post:
                rt = CardRuntime(
                    args_good,
                    logger,
                    {"n_agents": 3, "n_actions": 15, "episode_limit": 200},
                )
                # Before min_episodes=1, no LLM call yet
                self.assertEqual(mock_post.call_count, 0)
                self.assertEqual(rt.current_opponent_profile_vector(), [0.0] * 8)

                init_ctx = {
                    "ally_alive": 3.0,
                    "enemy_alive": 3.0,
                    "ally_health_total": 3.0,
                    "enemy_health_total": 3.0,
                    "ally_health_fraction": 1.0,
                    "enemy_health_fraction": 1.0,
                    "engage_distance": 0.4,
                    "formation_spread": 0.2,
                    "opponent_targets": [0, 0, 1],
                }
                rt.start_episode(0, {"env_name": "maca", "map_name": "3v3"}, init_ctx, False)
                rt.compute_reward(
                    env_id=0,
                    step_index=0,
                    actions=[1, 2, 9],
                    env_reward=0.5,
                    pre_context=init_ctx,
                    post_context=init_ctx,
                    terminated=True,
                    truncated=False,
                    env_info={"battle_won": 1},
                )
                rt.finish_episode(0, {"battle_won": 1})

                self.assertEqual(mock_post.call_count, 1)
                self.assertEqual(rt.opponent_profile.generation_mode, "llm")
                self.assertEqual(rt.opponent_profile.opponent_style, "aggressive_focus_fire")
                self.assertAlmostEqual(rt.current_opponent_profile_vector()[0], 0.78)

                loaded_prof = OpponentProfileStore.load(
                    os.path.join(rt.output_dir, "opponent_profile.json")
                )
                self.assertEqual(loaded_prof["generation_mode"], "llm")
                self.assertEqual(rt.llm_usage_totals["total_tokens"], 285)

            # Also verify graceful fallback to heuristic on transient API error
            with patch(
                "llm_marl.llm_client.requests.post",
                side_effect=RuntimeError("network unreachable"),
            ):
                rt.start_episode(0, {"env_name": "maca", "map_name": "3v3"}, init_ctx, False)
                rt.compute_reward(
                    env_id=0,
                    step_index=0,
                    actions=[1, 2, 9],
                    env_reward=-0.2,
                    pre_context=init_ctx,
                    post_context=init_ctx,
                    terminated=True,
                    truncated=False,
                    env_info={"battle_won": 0},
                )
                rt.finish_episode(0, {"battle_won": 0})
                self.assertEqual(
                    rt.opponent_profile.generation_mode, "llm_fallback_heuristic"
                )

    def test_oa_qmix_cli_smoke_3v3_and_5v5(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for map_name in ("3v3", "5v5"):
                cmd = [
                    sys.executable,
                    os.path.join(SRC_ROOT, "main.py"),
                    "--unobserved",
                    "--config=oa_qmix",
                    "--env-config=maca",
                    "with",
                    f"env_args.map_name={map_name}",
                    "env_args.time_limit=30",
                    "runner=parallel",
                    "batch_size_run=2",
                    "batch_size=2",
                    "buffer_size=8",
                    "use_rnn=True",
                    "t_max=120",
                    "test_interval=60",
                    "test_nepisode=2",
                    "log_interval=60",
                    "runner_log_interval=60",
                    "learner_log_interval=60",
                    f"local_results_path={tmpdir}",
                    "seed=7",
                ]
                env = dict(os.environ)
                env["MAGPIE_SACRED_SAVE_GIT_INFO"] = "0"
                proc = subprocess.run(
                    cmd,
                    cwd=EPYMARL_ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                combined = proc.stdout + "\n" + proc.stderr
                self.assertIn("Finished Training", combined)
                self.assertIn("oa_qmix_enabled", combined)

    def test_finetuned_oa_qmix_nq_learner_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                os.path.join(SRC_ROOT, "main.py"),
                "--unobserved",
                "--config=Finetuned-LLM-OA-QMIX",
                "--env-config=maca",
                "with",
                "env_args.map_name=3v3",
                "env_args.time_limit=30",
                "card.opponent.profile.mode=heuristic",
                "card.opponent.profile.min_episodes=0",
                "card.opponent.profile.refresh_episodes=2",
                "batch_size_run=2",
                "batch_size=2",
                "buffer_size=8",
                "t_max=120",
                "test_interval=60",
                "test_nepisode=2",
                "log_interval=60",
                "runner_log_interval=60",
                "learner_log_interval=60",
                f"local_results_path={tmpdir}",
                "seed=11",
            ]
            env = dict(os.environ)
            env["MAGPIE_SACRED_SAVE_GIT_INFO"] = "0"
            proc = subprocess.run(
                cmd,
                cwd=EPYMARL_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            combined = proc.stdout + "\n" + proc.stderr
            self.assertIn("Finished Training", combined)
            self.assertIn("Finetuned Mixer (OAQMixer)", combined)
            self.assertIn("oa_qmix_enabled", combined)


if __name__ == "__main__":
    unittest.main()
