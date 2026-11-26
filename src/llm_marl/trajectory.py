import json
import os

from .opponent_features import FEATURE_KEYS, OpponentFeatureExtractor


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


class _FeatureAccumulator:
    def __init__(self):
        self.count = 0
        self.sums = {key: 0.0 for key in FEATURE_KEYS}
        self.diagnostics = set()

    def add(self, features):
        if not features:
            return
        self.count += 1
        for key in FEATURE_KEYS:
            try:
                self.sums[key] += float(features.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
        for item in features.get("diagnostics", []) or []:
            self.diagnostics.add(item)

    def to_summary(self):
        if self.count <= 0:
            return {"episodes": 0, "means": {}, "diagnostics": ["no_opponent_features"]}
        return {
            "episodes": self.count,
            "means": {key: self.sums[key] / self.count for key in FEATURE_KEYS},
            "diagnostics": sorted(self.diagnostics),
        }


class TrajectoryStore:
    def __init__(self, output_dir, save_episodes=True, summary_write_interval=20):
        self.output_dir = output_dir
        self.save_episodes = bool(save_episodes)
        self.summary_write_interval = max(1, int(summary_write_interval or 1))
        self.episodes_dir = os.path.join(output_dir, "episodes")
        os.makedirs(output_dir, exist_ok=True)
        if self.save_episodes:
            os.makedirs(self.episodes_dir, exist_ok=True)
        self._active = {}
        self._completed = []
        self._train_summaries = []
        self._test_summaries = []
        self._accumulators = {
            "train": _FeatureAccumulator(),
            "test": _FeatureAccumulator(),
            "all": _FeatureAccumulator(),
        }
        self._counter = self._initial_counter() if self.save_episodes else 0
        self._log_buffer = {"train": [], "test": []}
        self._feature_extractor = OpponentFeatureExtractor()

    def _initial_counter(self):
        highest = -1
        if not os.path.isdir(self.episodes_dir):
            return 0
        for _, _, files in os.walk(self.episodes_dir):
            for filename in files:
                if not filename.startswith("episode_") or not filename.endswith(".json"):
                    continue
                stem = filename[len("episode_") : -len(".json")]
                try:
                    highest = max(highest, int(stem))
                except ValueError:
                    continue
        return highest + 1

    def start_episode(self, env_id, metadata, initial_context, test_mode):
        self._active[env_id] = {
            "env_id": env_id,
            "metadata": _to_jsonable(metadata),
            "initial_context": _to_jsonable(initial_context),
            "steps": [],
            "test_mode": bool(test_mode),
        }

    def record_step(
        self,
        env_id,
        step_index,
        actions,
        env_reward,
        shaped_reward,
        subrewards,
        aux_stats,
        prev_context,
        curr_context,
        terminated,
        truncated,
        env_info,
    ):
        episode = self._active[env_id]
        episode["steps"].append(
            {
                "step_index": int(step_index),
                "actions": _to_jsonable(actions),
                "env_reward": _to_jsonable(env_reward),
                "shaped_reward": _to_jsonable(shaped_reward),
                "subrewards": _to_jsonable(subrewards),
                "aux_stats": _to_jsonable(aux_stats),
                "prev_context": _to_jsonable(prev_context),
                "curr_context": _to_jsonable(curr_context),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "env_info": _to_jsonable(env_info),
            }
        )

    def finish_episode(self, env_id, outcome, env_info):
        episode = self._active.pop(env_id)
        steps = episode["steps"]

        team_return = 0.0
        subreward_totals = {}
        final_context = {}
        for step in steps:
            shaped = step["shaped_reward"]
            if isinstance(shaped, list):
                team_return += float(sum(shaped))
            else:
                team_return += float(shaped)
            for key, value in step["subrewards"].items():
                subreward_totals[key] = subreward_totals.get(key, 0.0) + float(value)
            final_context = step["curr_context"]

        episode_length = len(steps)
        opponent_features = self._feature_extractor.extract_episode(episode)
        summary = {
            "episode_id": self._counter,
            "env_id": env_id,
            "test_mode": episode["test_mode"],
            "metadata": episode["metadata"],
            "outcome": outcome,
            "episode_length": episode_length,
            "team_return": team_return,
            "normalized_team_return": team_return / max(1, episode_length),
            "subreward_totals": subreward_totals,
            "opponent_features": opponent_features,
            "combat_summary": {
                "ally_alive": final_context.get("ally_alive"),
                "enemy_alive": final_context.get("enemy_alive"),
                "ally_health_total": final_context.get("ally_health_total"),
                "enemy_health_total": final_context.get("enemy_health_total"),
                "battle_won": bool(env_info.get("battle_won", False)),
            },
            "env_info": _to_jsonable(env_info),
        }
        episode["summary"] = summary
        episode["opponent_features"] = opponent_features

        bucket = "test" if episode["test_mode"] else "train"
        self._log_buffer[bucket].append(summary)
        if episode["test_mode"]:
            self._test_summaries.append(summary)
        else:
            self._train_summaries.append(summary)
        self._accumulators[bucket].add(opponent_features)
        self._accumulators["all"].add(opponent_features)

        episode_path = ""
        if self.save_episodes:
            bucket_dir = os.path.join(self.episodes_dir, bucket)
            os.makedirs(bucket_dir, exist_ok=True)
            episode_path = os.path.join(bucket_dir, f"episode_{self._counter:06d}.json")
            with open(episode_path, "w", encoding="utf-8") as handle:
                json.dump(_to_jsonable(episode), handle, indent=2, sort_keys=True)
                handle.write("\n")

        summary["episode_path"] = episode_path
        # Retain lightweight summary record only (avoid keeping millions of step dicts in RAM)
        self._completed.append({"summary": summary})
        self._counter += 1
        if self._counter == 1 or self._counter % self.summary_write_interval == 0:
            self._write_opponent_feature_summary()
        return summary

    def train_episode_count(self):
        return len(self._train_summaries)

    def recent_summaries(self, test_mode=False, limit=10):
        source = self._test_summaries if test_mode else self._train_summaries
        if limit is None or limit <= 0:
            return list(source)
        return list(source[-int(limit) :])

    def completed_summaries(self, test_mode=None):
        if test_mode is None:
            return [episode["summary"] for episode in self._completed]
        if test_mode:
            return list(self._test_summaries)
        return list(self._train_summaries)

    def consume_log_summaries(self, test_mode):
        bucket = "test" if test_mode else "train"
        payload = list(self._log_buffer[bucket])
        self._log_buffer[bucket].clear()
        return payload

    def aggregate_opponent_features(self, test_mode=None, window=0):
        window = int(window or 0)
        if window > 0:
            if test_mode is None:
                recent = self._completed[-window:]
                summaries = [item["summary"] for item in recent]
            else:
                source = self._test_summaries if test_mode else self._train_summaries
                summaries = source[-window:]
            return self._feature_extractor.aggregate(summaries)
        if test_mode is None:
            return self._accumulators["all"].to_summary()
        bucket = "test" if test_mode else "train"
        return self._accumulators[bucket].to_summary()

    def flush_summary(self):
        self._write_opponent_feature_summary()

    def _write_opponent_feature_summary(self):
        path = os.path.join(self.output_dir, "opponent_feature_summary.json")
        payload = {
            "train": self.aggregate_opponent_features(test_mode=False),
            "test": self.aggregate_opponent_features(test_mode=True),
            "all": self.aggregate_opponent_features(test_mode=None),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
