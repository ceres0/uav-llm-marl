import math
import os
import sys
import numpy as np

from .multiagentenv import MultiAgentEnv


MAP_ALIASES = {
    "3v3": "600_600_fighter3v3",
    "fighter3v3": "600_600_fighter3v3",
    "3v3_large": "1000_1000_fighter3v3",
    "5v5": "800_800_fighter5v5",
    "fighter5v5": "800_800_fighter5v5",
    "5v5_large": "1000_1000_fighter5v5",
    "10v10": "1000_1000_fighter10v10",
    "fighter10v10": "1000_1000_fighter10v10",
    "2_10_vs_2_10": "1000_1000_2_10_vs_2_10",
}


def _find_maca_root(maca_path=None):
    candidates = []
    if maca_path:
        candidates.append(maca_path)
    if os.environ.get("MACA_PATH"):
        candidates.append(os.environ["MACA_PATH"])

    # Relative to epymarl/src/envs/maca_wrapper.py -> workspace/MaCA
    here = os.path.dirname(os.path.abspath(__file__))
    epymarl_root = os.path.dirname(os.path.dirname(here))
    workspace_root = os.path.dirname(epymarl_root)
    candidates.extend(
        [
            os.path.join(workspace_root, "MaCA"),
            os.path.join(epymarl_root, "MaCA"),
            os.path.join(os.getcwd(), "MaCA"),
        ]
    )

    for path in candidates:
        abs_p = os.path.abspath(path)
        if os.path.isdir(os.path.join(abs_p, "environment")) and os.path.isdir(
            os.path.join(abs_p, "maps")
        ):
            return abs_p

    raise FileNotFoundError(
        "Could not locate MaCA directory. Set `env_args.maca_path` or `$MACA_PATH`."
    )


class MaCAWrapper(MultiAgentEnv):
    """
    EPyMARL MultiAgentEnv wrapper for the MaCA UAV combat arena.

    Controls Side 1 (Red) with MARL agents against Side 2 (Blue) controlled by
    MaCA's rule-based opponent (`airc_homo_rule`). Supports both cooperative
    common-reward algorithms (QMIX, VDN, COMA, QTRAN) and individual-reward
    algorithms (MAPPO, IPPO, MAA2C, IA2C, IQL).
    """

    OWN_FEAT_DIM = 14
    ALLY_FEAT_DIM = 10
    ENEMY_FEAT_DIM = 13
    STATE_UNIT_DIM = 10

    def __init__(
        self,
        map_name="3v3",
        seed=None,
        time_limit=None,
        maca_path=None,
        opponent_policy="attack",
        n_move_dirs=8,
        random_pos=False,
        mask_invalid_attacks=True,
        obs_share_radar=True,
        reward_scale=True,
        reward_scale_rate=20.0,
        reward_negative_scale=0.5,
        reward_only_positive=False,
        continuing_episode=False,
        common_reward=True,
        reward_scalarisation="sum",
        render=False,
        **kwargs,
    ):
        self.maca_root = _find_maca_root(maca_path)
        env_dir = os.path.join(self.maca_root, "environment")
        for p in (self.maca_root, env_dir):
            if p not in sys.path:
                sys.path.insert(0, p)

        from interface import Environment as MaCAEnvironment
        import world.config as maca_config
        import world.position_calc as position_calc
        from agent.airc_homo_rule.agent import Agent as RuleOpponent
        from obs_construct.airc_homo_rule.construct import ObsConstruct as RuleObsConstruct

        self._maca_config = maca_config
        self._position_calc = position_calc

        resolved_name = MAP_ALIASES.get(str(map_name), str(map_name))
        if resolved_name.endswith(".map") and os.path.exists(resolved_name):
            self.map_path = os.path.abspath(resolved_name)
        else:
            fname = resolved_name if resolved_name.endswith(".map") else f"{resolved_name}.map"
            self.map_path = os.path.join(self.maca_root, "maps", fname)
        if not os.path.exists(self.map_path):
            raise FileNotFoundError(f"MaCA map file not found: {self.map_path}")

        self.map_name = map_name
        self._seed = int(seed) if seed is not None else 0
        self.random_pos = bool(random_pos)
        self.render_flag = bool(render)
        self.n_move_dirs = int(n_move_dirs)
        self.move_courses = [
            int(round(360.0 * i / self.n_move_dirs)) % 360 for i in range(self.n_move_dirs)
        ]
        self.mask_invalid_attacks = bool(mask_invalid_attacks)
        self.obs_share_radar = bool(obs_share_radar)

        self.reward_scale = bool(reward_scale)
        self.reward_scale_rate = float(reward_scale_rate)
        self.reward_negative_scale = float(reward_negative_scale)
        self.reward_only_positive = bool(reward_only_positive)
        self.continuing_episode = bool(continuing_episode)
        self.common_reward = bool(common_reward)
        self.reward_scalarisation = reward_scalarisation

        if self.reward_scalarisation not in ("sum", "mean"):
            raise ValueError(
                f"Invalid reward_scalarisation: {self.reward_scalarisation} (expected 'sum' or 'mean')"
            )

        # Probe map size to choose default episode_limit when time_limit is None
        from world.load_map import Map as MaCAMap

        map_meta = MaCAMap(self.map_path)
        size_x, size_y = map_meta.get_map_size()
        if time_limit is None or int(time_limit) <= 0:
            if max(size_x, size_y) <= 600:
                self.episode_limit = 200
            elif max(size_x, size_y) <= 800:
                self.episode_limit = 250
            else:
                self.episode_limit = 400
        else:
            self.episode_limit = int(time_limit)

        self._env = MaCAEnvironment(
            self.map_path,
            "raw",
            "raw",
            max_step=self.episode_limit,
            render=self.render_flag,
            random_pos=self.random_pos,
            log=False,
            random_seed=self._seed,
        )

        self.size_x, self.size_y = self._env.get_map_size()
        self.max_dist = math.hypot(self.size_x, self.size_y)
        (
            self.n_ally_detectors,
            self.n_ally_fighters,
            self.n_enemy_detectors,
            self.n_enemy_fighters,
        ) = self._env.get_unit_num()
        (
            self.ally_det_props,
            self.ally_fgt_props,
            self.enemy_det_props,
            self.enemy_fgt_props,
        ) = self._env.get_unit_property_list()

        self.n_agents = self.n_ally_detectors + self.n_ally_fighters
        self.n_enemies = self.n_enemy_detectors + self.n_enemy_fighters

        # Action space: [no-op] + [move_0 .. move_{M-1}] + [long_missile_0 .. N_e-1] + [short_missile_0 .. N_e-1]
        self.n_actions = 1 + self.n_move_dirs + 2 * self.n_enemies

        self.l_missile_range = float(self._maca_config.get_l_missile_dis())
        self.s_missile_range = float(self._maca_config.get_s_missile_dis())
        self.max_l_missile = float(max(1, self._maca_config.get_l_missile_num()))
        self.max_s_missile = float(max(1, self._maca_config.get_s_missile_num()))
        self.max_missiles_total = self.max_l_missile + self.max_s_missile

        # Opponent rule agent for Side 2 (Blue)
        self.opponent_policy = opponent_policy
        self._opponent_obs_builder = RuleObsConstruct(
            self.size_x, self.size_y, self.n_enemy_detectors, self.n_enemy_fighters
        )
        self._opponent_agent = RuleOpponent(
            rule_set=self.opponent_policy, seed=self._seed + 10007
        )
        self._opponent_agent.set_map_info(
            self.size_x,
            self.size_y,
            self.n_enemy_detectors,
            self.n_enemy_fighters,
            enemy_total_num=self.n_agents,
        )

        self._step_cnt = 0
        self._side1_raw = None
        self._side2_raw = None
        self._obs = None
        self._last_opp_targets = []

        # Cumulative battle statistics
        self.battles_won = 0
        self.battles_game = 0
        self.battles_draw = 0
        self.timeouts = 0

        self._refresh_raw_obs()

    def _refresh_raw_obs(self):
        self._side1_raw, self._side2_raw = self._env.get_obs_raw()
        self._obs = self._build_all_obs()

    def _get_ally_units(self):
        det_list = self._side1_raw["detector_obs_list"]
        fgt_list = self._side1_raw["fighter_obs_list"]
        units = []
        for idx, d in enumerate(det_list):
            units.append(("detector", idx, d, self.ally_det_props[idx]))
        for idx, f in enumerate(fgt_list):
            units.append(("fighter", idx, f, self.ally_fgt_props[idx]))
        return units

    def _collect_detected_enemies(self, ally_units):
        """
        Collect enemy detections from friendly sensors (`side1_obs_raw` only).
        Returns:
          team_detected: dict[enemy_id -> {'pos_x', 'pos_y', 'type'}]
          detector_seen_ids: set of enemy_ids seen by alive friendly detectors
        """
        team_detected = {}
        detector_seen_ids = set()

        for u_type, _, u_obs, _ in ally_units:
            if not u_obs["alive"]:
                continue
            for item in u_obs.get("r_visible_list", []):
                eid = item["id"]
                team_detected[eid] = item
                if u_type == "detector" and u_obs.get("r_iswork", False):
                    detector_seen_ids.add(eid)

        for item in self._side1_raw["joint_obs_dict"].get("passive_detection_enemy_list", []):
            eid = item["id"]
            if eid not in team_detected:
                team_detected[eid] = item

        return team_detected, detector_seen_ids

    def _build_all_obs(self):
        ally_units = self._get_ally_units()
        team_detected, _ = self._collect_detected_enemies(ally_units)
        strike_targets = {
            st["target_id"] for st in self._side1_raw["joint_obs_dict"].get("strike_list", [])
        }

        obs_size = self.get_obs_size()
        all_obs = []

        for agent_id, (u_type, _, u_obs, u_prop) in enumerate(ally_units):
            if not u_obs["alive"]:
                all_obs.append(np.zeros(obs_size, dtype=np.float32))
                continue

            feats = []
            own_x = float(u_obs["pos_x"])
            own_y = float(u_obs["pos_y"])
            own_course_rad = math.radians(float(u_obs["course"]))
            is_fighter = 1.0 if u_type == "fighter" else 0.0
            speed_norm = float(u_prop.get("speed", 3.0)) / 5.0
            r_work = 1.0 if u_obs.get("r_iswork", False) else 0.0
            r_fp_norm = float(u_obs.get("r_fre_point", 0)) / 20.0

            if u_type == "fighter":
                j_work = 1.0 if u_obs.get("j_iswork", False) else 0.0
                j_fp_norm = float(u_obs.get("j_fre_point", 0)) / 21.0
                l_left = float(u_obs.get("l_missile_left", 0)) / self.max_l_missile
                s_left = float(u_obs.get("s_missile_left", 0)) / self.max_s_missile
                striking_cnt = float(len(u_obs.get("striking_list", []))) / self.max_missiles_total
            else:
                j_work = 0.0
                j_fp_norm = 0.0
                l_left = 0.0
                s_left = 0.0
                striking_cnt = 0.0

            # 1. Own features (14 dims)
            feats.extend(
                [
                    1.0,
                    own_x / self.size_x,
                    own_y / self.size_y,
                    math.cos(own_course_rad),
                    math.sin(own_course_rad),
                    is_fighter,
                    speed_norm,
                    r_work,
                    r_fp_norm,
                    j_work,
                    j_fp_norm,
                    l_left,
                    s_left,
                    striking_cnt,
                ]
            )

            # 2. Allied features ((n_agents - 1) * 10 dims)
            for other_id, (o_type, _, o_obs, _) in enumerate(ally_units):
                if other_id == agent_id:
                    continue
                if not o_obs["alive"]:
                    feats.extend([0.0] * self.ALLY_FEAT_DIM)
                    continue
                dx = float(o_obs["pos_x"]) - own_x
                dy = float(o_obs["pos_y"]) - own_y
                dist = math.hypot(dx, dy)
                bearing = math.atan2(dy, dx)
                o_course_rad = math.radians(float(o_obs["course"]))
                o_l = (
                    float(o_obs.get("l_missile_left", 0)) / self.max_l_missile
                    if o_type == "fighter"
                    else 0.0
                )
                o_s = (
                    float(o_obs.get("s_missile_left", 0)) / self.max_s_missile
                    if o_type == "fighter"
                    else 0.0
                )
                feats.extend(
                    [
                        1.0,
                        dx / self.size_x,
                        dy / self.size_y,
                        dist / self.max_dist,
                        math.cos(bearing),
                        math.sin(bearing),
                        math.cos(o_course_rad),
                        math.sin(o_course_rad),
                        o_l,
                        o_s,
                    ]
                )

            # 3. Enemy features (n_enemies * 13 dims)
            self_visible_map = {item["id"]: item for item in u_obs.get("r_visible_list", [])}
            self_jam_ids = {item["id"] for item in u_obs.get("j_recv_list", [])}

            for k in range(self.n_enemies):
                eid = k + 1
                vis_self = 1.0 if (eid in self_visible_map or eid in self_jam_ids) else 0.0
                enemy_info = (
                    team_detected.get(eid)
                    if self.obs_share_radar
                    else self_visible_map.get(eid)
                )
                if enemy_info is None:
                    feats.extend([0.0] * self.ENEMY_FEAT_DIM)
                    continue

                ex = float(enemy_info["pos_x"])
                ey = float(enemy_info["pos_y"])
                dx = ex - own_x
                dy = ey - own_y
                dist = math.hypot(dx, dy)
                world_ang = math.atan2(dy, dx)
                rel_ang = world_ang - own_course_rad
                in_l_range = 1.0 if dist <= self.l_missile_range else 0.0
                in_s_range = 1.0 if dist <= self.s_missile_range else 0.0
                e_is_fighter = 1.0 if enemy_info.get("type", 1) == 1 else 0.0
                under_strike = 1.0 if eid in strike_targets else 0.0

                feats.extend(
                    [
                        vis_self,
                        1.0,
                        dx / self.size_x,
                        dy / self.size_y,
                        dist / self.max_dist,
                        math.cos(world_ang),
                        math.sin(world_ang),
                        math.cos(rel_ang),
                        math.sin(rel_ang),
                        in_l_range,
                        in_s_range,
                        e_is_fighter,
                        under_strike,
                    ]
                )

            all_obs.append(np.asarray(feats, dtype=np.float32))

        return all_obs

    def get_obs(self):
        return self._obs

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return (
            self.OWN_FEAT_DIM
            + max(0, self.n_agents - 1) * self.ALLY_FEAT_DIM
            + self.n_enemies * self.ENEMY_FEAT_DIM
        )

    def get_state(self):
        """Global state vector for centralized training (CTDE)."""
        state_feats = []

        o_strikes_by_attacker = {}
        o_strikes_by_target = {}
        for st in self._side1_raw["joint_obs_dict"].get("strike_list", []):
            o_strikes_by_attacker[st["attacker_id"]] = (
                o_strikes_by_attacker.get(st["attacker_id"], 0) + 1
            )
            o_strikes_by_target[st["target_id"]] = (
                o_strikes_by_target.get(st["target_id"], 0) + 1
            )

        e_strikes_by_target = {}
        for st in self._side2_raw["joint_obs_dict"].get("strike_list", []):
            e_strikes_by_target[st["target_id"]] = (
                e_strikes_by_target.get(st["target_id"], 0) + 1
            )

        def _append_side_units(det_list, fgt_list, strike_count_map):
            for u in det_list:
                if not u["alive"]:
                    state_feats.extend([0.0] * self.STATE_UNIT_DIM)
                else:
                    c_rad = math.radians(float(u["course"]))
                    state_feats.extend(
                        [
                            1.0,
                            float(u["pos_x"]) / self.size_x,
                            float(u["pos_y"]) / self.size_y,
                            math.cos(c_rad),
                            math.sin(c_rad),
                            1.0 if u.get("r_iswork", False) else 0.0,
                            0.0,
                            0.0,
                            0.0,
                            float(strike_count_map.get(u["id"], 0)) / self.max_missiles_total,
                        ]
                    )
            for u in fgt_list:
                if not u["alive"]:
                    state_feats.extend([0.0] * self.STATE_UNIT_DIM)
                else:
                    c_rad = math.radians(float(u["course"]))
                    state_feats.extend(
                        [
                            1.0,
                            float(u["pos_x"]) / self.size_x,
                            float(u["pos_y"]) / self.size_y,
                            math.cos(c_rad),
                            math.sin(c_rad),
                            1.0 if u.get("r_iswork", False) else 0.0,
                            1.0 if u.get("j_iswork", False) else 0.0,
                            float(u.get("l_missile_left", 0)) / self.max_l_missile,
                            float(u.get("s_missile_left", 0)) / self.max_s_missile,
                            float(strike_count_map.get(u["id"], 0)) / self.max_missiles_total,
                        ]
                    )

        _append_side_units(
            self._side1_raw["detector_obs_list"],
            self._side1_raw["fighter_obs_list"],
            o_strikes_by_attacker,
        )
        _append_side_units(
            self._side2_raw["detector_obs_list"],
            self._side2_raw["fighter_obs_list"],
            o_strikes_by_target,
        )
        state_feats.append(float(self._step_cnt) / float(max(1, self.episode_limit)))
        return np.asarray(state_feats, dtype=np.float32)

    def get_state_size(self):
        return (self.n_agents + self.n_enemies) * self.STATE_UNIT_DIM + 1

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        ally_units = self._get_ally_units()
        u_type, _, u_obs, u_prop = ally_units[agent_id]

        if not u_obs["alive"]:
            return [1] + [0] * (self.n_actions - 1)

        avail = [0] * self.n_actions
        own_x = float(u_obs["pos_x"])
        own_y = float(u_obs["pos_y"])
        speed = float(u_prop.get("speed", 3.0))

        # Check valid movement headings (mask moves that push out of battlefield bounds)
        move_valid_any = False
        for m_idx, course in enumerate(self.move_courses):
            nx = own_x + round(math.cos(math.radians(course)) * speed)
            ny = own_y + round(math.sin(math.radians(course)) * speed)
            if 0.0 <= nx <= float(self.size_x) and 0.0 <= ny <= float(self.size_y):
                avail[1 + m_idx] = 1
                move_valid_any = True
        if not move_valid_any:
            for m_idx in range(self.n_move_dirs):
                avail[1 + m_idx] = 1

        if u_type != "fighter":
            return avail

        l_left = int(u_obs.get("l_missile_left", 0))
        s_left = int(u_obs.get("s_missile_left", 0))
        if l_left <= 0 and s_left <= 0:
            return avail

        if not self.mask_invalid_attacks:
            for k in range(self.n_enemies):
                if l_left > 0:
                    avail[1 + self.n_move_dirs + k] = 1
                if s_left > 0:
                    avail[1 + self.n_move_dirs + self.n_enemies + k] = 1
            return avail

        # Targetable enemies: detected by own radar/jammer OR alive friendly detector radar
        team_detected, detector_seen_ids = self._collect_detected_enemies(ally_units)
        own_radar_map = {item["id"]: item for item in u_obs.get("r_visible_list", [])}
        own_jam_ids = (
            {item["id"] for item in u_obs.get("j_recv_list", [])}
            if u_obs.get("j_iswork", False)
            else set()
        )

        for k in range(self.n_enemies):
            eid = k + 1
            can_lock = (
                (u_obs.get("r_iswork", False) and eid in own_radar_map)
                or (eid in own_jam_ids and eid in team_detected)
                or (eid in detector_seen_ids)
            )
            if not can_lock:
                continue
            target_info = own_radar_map.get(eid) or team_detected.get(eid)
            if target_info is None:
                continue
            dist = self._position_calc.get_distance(
                own_x, own_y, target_info["pos_x"], target_info["pos_y"]
            )
            if l_left > 0 and dist <= self.l_missile_range:
                avail[1 + self.n_move_dirs + k] = 1
            if s_left > 0 and dist <= self.s_missile_range:
                avail[1 + self.n_move_dirs + self.n_enemies + k] = 1

        return avail

    def get_total_actions(self):
        return self.n_actions

    def _decode_side1_actions(self, actions):
        ally_units = self._get_ally_units()
        team_detected, _ = self._collect_detected_enemies(ally_units)

        det_actions = []
        fgt_actions = []

        for agent_id, act_raw in enumerate(actions):
            act = int(act_raw)
            u_type, _, u_obs, _ = ally_units[agent_id]
            own_x = u_obs["pos_x"]
            own_y = u_obs["pos_y"]
            cur_course = int(u_obs["course"]) % 360

            if not u_obs["alive"] or act == 0:
                if u_type == "detector":
                    det_actions.append(
                        {"course": cur_course, "r_iswork": False, "r_fre_point": 0}
                    )
                else:
                    fgt_actions.append(
                        {
                            "course": cur_course,
                            "r_iswork": False,
                            "r_fre_point": 0,
                            "j_iswork": False,
                            "j_fre_point": 0,
                            "hit_target": 0,
                            "missile_type": 0,
                        }
                    )
                continue

            if 1 <= act <= self.n_move_dirs:
                chosen_course = self.move_courses[act - 1]
                if u_type == "detector":
                    det_actions.append(
                        {"course": chosen_course, "r_iswork": True, "r_fre_point": 1}
                    )
                else:
                    fgt_actions.append(
                        {
                            "course": chosen_course,
                            "r_iswork": True,
                            "r_fre_point": 2,
                            "j_iswork": True,
                            "j_fre_point": 2,
                            "hit_target": 0,
                            "missile_type": 0,
                        }
                    )
            else:
                att_offset = act - (1 + self.n_move_dirs)
                if att_offset < self.n_enemies:
                    target_id = att_offset + 1
                    missile_type = 1
                else:
                    target_id = (att_offset - self.n_enemies) + 1
                    missile_type = 2

                # Steer toward target to keep radar illumination on target
                own_radar_map = {
                    item["id"]: item for item in u_obs.get("r_visible_list", [])
                }
                t_info = own_radar_map.get(target_id) or team_detected.get(target_id)
                if t_info is not None:
                    chosen_course = self._position_calc.angle_cal(
                        own_x, own_y, t_info["pos_x"], t_info["pos_y"]
                    )
                else:
                    chosen_course = cur_course

                if u_type == "detector":
                    det_actions.append(
                        {"course": chosen_course, "r_iswork": True, "r_fre_point": 1}
                    )
                else:
                    fgt_actions.append(
                        {
                            "course": chosen_course,
                            "r_iswork": True,
                            "r_fre_point": 2,
                            "j_iswork": True,
                            "j_fre_point": 2,
                            "hit_target": target_id,
                            "missile_type": missile_type,
                        }
                    )

        return det_actions, fgt_actions

    def _shape_value(self, val):
        v = float(val)
        if v < 0.0:
            if self.reward_only_positive:
                v = 0.0
            else:
                v *= self.reward_negative_scale
        if self.reward_scale and self.reward_scale_rate > 0.0:
            v /= self.reward_scale_rate
        return v

    def step(self, actions):
        self._step_cnt += 1

        side1_det_act, side1_fgt_act = self._decode_side1_actions(actions)

        opp_obs = self._opponent_obs_builder.obs_construct(self._side2_raw)
        side2_det_act, side2_fgt_act = self._opponent_agent.get_action(
            opp_obs, self._step_cnt
        )

        opp_targets = []
        for f_idx, f_act in enumerate(side2_fgt_act):
            u_info = opp_obs.get(f_idx)
            if not u_info or not u_info.get("alive", False):
                continue
            hit_t = int(f_act.get("hit_target", 0) or 0)
            if 1 <= hit_t <= self.n_agents:
                opp_targets.append(hit_t - 1)
            else:
                t_list = u_info.get("target_list") or []
                if t_list:
                    tid = int(t_list[0].get("id", 0) or 0)
                    if 1 <= tid <= self.n_agents:
                        opp_targets.append(tid - 1)

        self._env.step(side1_det_act, side1_fgt_act, side2_det_act, side2_fgt_act)
        self._refresh_raw_obs()

        if not opp_targets:
            for st in self._side2_raw["joint_obs_dict"].get("strike_list", []):
                tid = int(st.get("target_id", 0) or 0)
                if 1 <= tid <= self.n_agents:
                    opp_targets.append(tid - 1)
        self._last_opp_targets = opp_targets

        (
            o_det_rew,
            o_fgt_rew,
            o_game_rew,
            _,
            _,
            e_game_rew,
        ) = self._env.get_reward()

        raw_agent_rews = np.concatenate([o_det_rew, o_fgt_rew]).astype(np.float64)
        shaped_agent_rews = [self._shape_value(r) for r in raw_agent_rews]

        terminated = bool(self._env.get_done())
        truncated = False
        info = {}

        ally_alive = sum(
            1
            for u in self._side1_raw["detector_obs_list"]
            + self._side1_raw["fighter_obs_list"]
            if u["alive"]
        )
        enemy_alive = sum(
            1
            for u in self._side2_raw["detector_obs_list"]
            + self._side2_raw["fighter_obs_list"]
            if u["alive"]
        )

        game_bonus = 0.0
        if terminated:
            self.battles_game += 1
            hit_time_limit = (
                self._step_cnt >= self.episode_limit
                and ally_alive > 0
                and enemy_alive > 0
            )
            if hit_time_limit:
                self.timeouts += 1

            info["episode_limit"] = bool(hit_time_limit and self.continuing_episode)
            info["dead_allies"] = int(self.n_agents - ally_alive)
            info["dead_enemies"] = int(self.n_enemies - enemy_alive)

            if o_game_rew > e_game_rew:
                self.battles_won += 1
                info["battle_won"] = 1
                info["battle_draw"] = 0
                info["battle_lost"] = 0
            elif o_game_rew < e_game_rew:
                info["battle_won"] = 0
                info["battle_draw"] = 0
                info["battle_lost"] = 1
            else:
                self.battles_draw += 1
                info["battle_won"] = 0
                info["battle_draw"] = 1
                info["battle_lost"] = 0

            game_bonus = self._shape_value(o_game_rew)

        if self.common_reward:
            total_r = float(sum(shaped_agent_rews) + game_bonus)
            if self.reward_scalarisation == "mean":
                total_r /= float(self.n_agents)
            reward_out = float(total_r)
        else:
            per_agent_bonus = game_bonus / float(self.n_agents)
            reward_out = [float(r + per_agent_bonus) for r in shaped_agent_rews]

        return self._obs, reward_out, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._step_cnt = 0
        self._last_opp_targets = []
        self._env.reset()
        self._opponent_agent.reset()
        self._refresh_raw_obs()
        return self._obs, {}

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        if seed is not None:
            self._seed = int(seed)
            self._env.env.seed(self._seed)

    def save_replay(self):
        pass

    def get_stats(self):
        return {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.battles_draw,
            "win_rate": (
                float(self.battles_won) / float(self.battles_game)
                if self.battles_game > 0
                else 0.0
            ),
            "timeouts": self.timeouts,
        }

    def get_card_context(self):
        """
        Centralized training-time combat context for OA-QMIX / LLM-OA-QMIX
        opponent profiling (never passed to decentralized actors).
        """
        ally_units = (
            self._side1_raw["detector_obs_list"] + self._side1_raw["fighter_obs_list"]
        )
        enemy_units = (
            self._side2_raw["detector_obs_list"] + self._side2_raw["fighter_obs_list"]
        )
        ally_pos = [
            (float(u["pos_x"]), float(u["pos_y"])) for u in ally_units if u["alive"]
        ]
        enemy_pos = [
            (float(u["pos_x"]), float(u["pos_y"])) for u in enemy_units if u["alive"]
        ]
        ally_alive = len(ally_pos)
        enemy_alive = len(enemy_pos)

        inv_max_dist = 1.0 / max(1e-6, self.max_dist)
        if ally_pos and enemy_pos:
            pair_dists = [
                math.hypot(ax - ex, ay - ey) * inv_max_dist
                for ax, ay in ally_pos
                for ex, ey in enemy_pos
            ]
            engage_distance = sum(pair_dists) / len(pair_dists)
        else:
            engage_distance = 0.0

        spread_source = enemy_pos if len(enemy_pos) >= 2 else ally_pos
        if len(spread_source) >= 2:
            spread_dists = []
            for i in range(len(spread_source)):
                x1, y1 = spread_source[i]
                for j in range(i + 1, len(spread_source)):
                    x2, y2 = spread_source[j]
                    spread_dists.append(math.hypot(x1 - x2, y1 - y2) * inv_max_dist)
            formation_spread = sum(spread_dists) / len(spread_dists)
        else:
            formation_spread = 0.0

        return {
            "ally_alive": float(ally_alive),
            "enemy_alive": float(enemy_alive),
            "ally_health_total": float(ally_alive),
            "enemy_health_total": float(enemy_alive),
            "ally_health_fraction": float(ally_alive) / float(max(1, self.n_agents)),
            "enemy_health_fraction": float(enemy_alive) / float(max(1, self.n_enemies)),
            "engage_distance": float(min(1.0, max(0.0, engage_distance))),
            "formation_spread": float(min(1.0, max(0.0, formation_spread))),
            "opponent_targets": list(self._last_opp_targets),
        }

    def get_card_env_descriptor(self):
        return {
            "env_name": "maca",
            "map_name": str(self.map_name),
            "opponent_policy": str(self.opponent_policy),
            "episode_limit": int(self.episode_limit),
            "n_agents": int(self.n_agents),
            "n_enemies": int(self.n_enemies),
            "n_actions": int(self.n_actions),
        }
