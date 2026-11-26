import copy
import torch as th
from torch.optim import Adam, RMSprop

from components.episode_buffer import EpisodeBatch
from modules.critics.opponent_aux import OpponentAuxHead, masked_aux_loss
from modules.mixers.nmix import Mixer
from modules.mixers.oa_qmix import OAQMixer
from modules.mixers.vdn import VDNMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num, orthogonal_init_


class NQLearner:
    """
    PyMARL2 NQLearner supporting both Finetuned-QMIX (Mixer) and
    Opponent-Aware Finetuned-QMIX / Finetuned-LLM-OA-QMIX (OAQMixer):
    - Multi-step TD(lambda) / Q(lambda) target estimation
    - Adam optimizer (lr=0.001)
    - Episode-based target network hard updates
    - 0.5 * TD-error^2 loss with no_grad target rollouts
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.device = th.device("cuda" if args.use_cuda else "cpu")
        self.params = list(mac.parameters())

        if args.mixer == "vdn":
            assert args.common_reward, "VDN only supports common reward setting"
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            assert args.common_reward, "Finetuned-QMIX only supports common reward setting"
            if getattr(args, "oa_qmix_enabled", False):
                self.mixer = OAQMixer(args)
                if getattr(args, "use_orthogonal", False):
                    for m in self.mixer.modules():
                        orthogonal_init_(m)
            else:
                self.mixer = Mixer(args)
        elif args.mixer == "oa_qmix":
            assert args.common_reward, "Finetuned-OA-QMIX only supports common reward setting"
            self.mixer = OAQMixer(args)
            if getattr(args, "use_orthogonal", False):
                for m in self.mixer.modules():
                    orthogonal_init_(m)
        else:
            raise ValueError(f"Mixer {args.mixer} not recognised in NQLearner.")

        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        self.aux_head = None
        if getattr(args, "opponent_aux_enabled", False):
            hidden_dim = int(getattr(args, "rnn_hidden_dim", getattr(args, "hidden_dim", 64)))
            input_dim = hidden_dim + int(getattr(args, "opp_profile_dim", 0) or 0)
            self.aux_head = OpponentAuxHead(
                input_dim=input_dim,
                hidden_dim=args.mixing_embed_dim,
                style_classes=4,
                n_agents=args.n_agents,
            )
            self.params += list(self.aux_head.parameters())

        self.logger.console_logger.info(
            f"Finetuned Mixer ({self.mixer.__class__.__name__}) Size: {get_parameters_num(self.mixer.parameters())}"
        )

        if getattr(self.args, "optimizer", "adam") == "adam":
            self.optimiser = Adam(
                params=self.params,
                lr=args.lr,
                weight_decay=getattr(args, "weight_decay", 0),
            )
        else:
            self.optimiser = RMSprop(
                params=self.params,
                lr=args.lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps,
            )

        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        self.target_update_interval = getattr(
            self.args,
            "target_update_interval",
            getattr(self.args, "target_update_interval_or_tau", 200),
        )

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        has_opp_profile = "opp_profile" in batch.scheme

        # Online Q-values
        self.mac.agent.train()
        mac_out = []
        mac_hidden = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            if self.aux_head is not None:
                mac_hidden.append(
                    self.mac.hidden_states.reshape(batch.batch_size, self.n_agents, -1)
                )
        mac_out = th.stack(mac_out, dim=1)
        mac_hidden = th.stack(mac_hidden, dim=1) if mac_hidden else None

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Target Q-values and TD(lambda) return targets
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            target_mac_out = th.stack(target_mac_out, dim=1)

            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            if isinstance(self.target_mixer, OAQMixer):
                target_max_qvals = self.target_mixer(
                    target_max_qvals,
                    batch["state"],
                    batch["opp_profile"] if has_opp_profile else None,
                )
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            td_lambda = getattr(self.args, "td_lambda", 0.6)
            if getattr(self.args, "q_lambda", False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                if isinstance(self.target_mixer, OAQMixer):
                    qvals = self.target_mixer(
                        qvals,
                        batch["state"],
                        batch["opp_profile"] if has_opp_profile else None,
                    )
                else:
                    qvals = self.target_mixer(qvals, batch["state"])
                targets = build_q_lambda_targets(
                    rewards,
                    terminated,
                    mask,
                    target_max_qvals,
                    qvals,
                    self.args.gamma,
                    td_lambda,
                )
            else:
                targets = build_td_lambda_targets(
                    rewards,
                    terminated,
                    mask,
                    target_max_qvals,
                    self.args.n_agents,
                    self.args.gamma,
                    td_lambda,
                )

        if isinstance(self.mixer, OAQMixer):
            chosen_action_qvals = self.mixer(
                chosen_action_qvals,
                batch["state"][:, :-1],
                batch["opp_profile"][:, :-1] if has_opp_profile else None,
            )
        else:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = chosen_action_qvals - targets.detach()
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask
        loss = masked_td_error.sum() / mask.sum()

        aux_metrics = None
        if self.aux_head is not None and "opp_label_mask" in batch.scheme:
            hidden_dim = int(
                getattr(self.args, "rnn_hidden_dim", getattr(self.args, "hidden_dim", 64))
            )
            agent_repr = mac_hidden[:, :-1].mean(dim=2).reshape(-1, hidden_dim)
            opp_prof = batch["opp_profile"][:, :-1].reshape(
                -1, int(getattr(self.args, "opp_profile_dim", 0) or 0)
            )
            label_mask = batch["opp_label_mask"][:, :-1].reshape(-1, 3) > 0
            labels = {
                "style": batch["opp_style_label"][:, :-1].reshape(-1),
                "focus": batch["enemy_focus_target"][:, :-1].reshape(-1),
                "danger": batch["danger_window"][:, :-1].reshape(-1),
            }
            valid = mask.reshape(-1, mask.size(-1)).any(dim=-1)
            masks = {
                "style": label_mask[:, 0] & valid,
                "focus": label_mask[:, 1] & valid,
                "danger": label_mask[:, 2] & valid,
            }
            aux_loss, aux_metrics = masked_aux_loss(
                self.aux_head(agent_repr, opp_prof), labels, masks
            )
            loss = loss + self.args.opponent_aux_loss_weight * aux_loss

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        self.train_t += 1

        if (episode_num - self.last_target_update_episode) / self.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss.item(), t_env)
            if isinstance(self.mixer, OAQMixer):
                self.logger.log_stat("oa_qmix_enabled", 1.0, t_env)
                self.logger.log_stat("critic_input_dim", self.mixer.critic_input_dim, t_env)
                if has_opp_profile:
                    self.logger.log_stat(
                        "opp_profile_norm",
                        batch["opp_profile"][:, :-1].norm(dim=-1).mean().item(),
                        t_env,
                    )
            if aux_metrics is not None:
                for key, value in aux_metrics.items():
                    self.logger.log_stat("aux/" + key, value, t_env)
            self.logger.log_stat(
                "grad_norm",
                grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
                t_env,
            )
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs",
                (td_error.abs() * mask).sum().item() / mask_elems,
                t_env,
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.aux_head is not None:
            self.aux_head.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        if self.aux_head is not None:
            th.save(self.aux_head.state_dict(), "{}/opponent_aux.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage)
            )
        if self.aux_head is not None:
            self.aux_head.load_state_dict(
                th.load(
                    "{}/opponent_aux.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
