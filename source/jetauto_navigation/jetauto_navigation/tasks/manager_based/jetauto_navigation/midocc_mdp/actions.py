from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class PlanarPoseStepAction(ActionTerm):
    """Mid-occ route: bounded quantized planar pose steps."""

    cfg: "PlanarPoseStepActionCfg"

    def __init__(self, cfg: "PlanarPoseStepActionCfg", env) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._step_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._substep_scale = 1.0 / float(env.cfg.decimation)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._step_cmd

    def process_actions(self, actions: torch.Tensor):
        actions = actions.clamp(-1.0, 1.0)
        self._raw_actions[:] = actions

        bins = torch.zeros_like(actions)
        bins[actions > self.cfg.threshold] = 1.0
        bins[actions < -self.cfg.threshold] = -1.0

        self._step_cmd[:, 0] = bins[:, 0] * self.cfg.lin_xy_step[0]
        self._step_cmd[:, 1] = bins[:, 1] * self.cfg.lin_xy_step[1]
        self._step_cmd[:, 2] = bins[:, 2] * self.cfg.yaw_step

    def apply_actions(self):
        quat_w = self.robot.data.root_quat_w
        pos = self.robot.data.root_pos_w.clone()
        env_origins = self._env.scene.env_origins

        delta_body = torch.stack(
            (
                self._step_cmd[:, 0] * self._substep_scale,
                self._step_cmd[:, 1] * self._substep_scale,
                torch.zeros_like(self._step_cmd[:, 0]),
            ),
            dim=-1,
        )
        pos += math_utils.quat_apply(quat_w, delta_body)

        if self.cfg.z_lock is not None:
            pos[:, 2] = self.cfg.z_lock
        if self.cfg.x_limits is not None:
            pos_local_x = (pos[:, 0] - env_origins[:, 0]).clamp(self.cfg.x_limits[0], self.cfg.x_limits[1])
            pos[:, 0] = env_origins[:, 0] + pos_local_x
        if self.cfg.y_limits is not None:
            pos_local_y = (pos[:, 1] - env_origins[:, 1]).clamp(self.cfg.y_limits[0], self.cfg.y_limits[1])
            pos[:, 1] = env_origins[:, 1] + pos_local_y

        rot_m = math_utils.matrix_from_quat(quat_w)
        yaw = torch.atan2(rot_m[:, 1, 0], rot_m[:, 0, 0]) + self._step_cmd[:, 2] * self._substep_scale
        if self.cfg.yaw_limits is not None:
            yaw = yaw.clamp(self.cfg.yaw_limits[0], self.cfg.yaw_limits[1])
        zeros = torch.zeros_like(yaw)
        quat_yaw = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

        self.robot.write_root_pose_to_sim(torch.cat([pos, quat_yaw], dim=-1))
        self.robot.write_root_velocity_to_sim(torch.zeros(self.num_envs, 6, device=self.device))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._step_cmd.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._step_cmd[env_ids] = 0.0


@configclass
class PlanarPoseStepActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PlanarPoseStepAction

    asset_name: str = MISSING
    lin_xy_step: tuple[float, float] = (0.04, 0.04)
    yaw_step: float = math.radians(15.0)
    threshold: float = 0.33
    z_lock: float | None = 0.22
    x_limits: tuple[float, float] | None = None
    y_limits: tuple[float, float] | None = None
    yaw_limits: tuple[float, float] | None = None


class PlanarPoseContinuousAction(ActionTerm):
    """Mid-occ route: bounded continuous planar pose deltas.

    Unlike :class:`PlanarPoseStepAction`, this action keeps the policy-to-motion
    map smooth: each action dimension in ``[-1, 1]`` is scaled directly to a
    planar translation / yaw delta for one environment step.
    """

    cfg: "PlanarPoseContinuousActionCfg"

    def __init__(self, cfg: "PlanarPoseContinuousActionCfg", env) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._delta_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._substep_scale = 1.0 / float(env.cfg.decimation)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._delta_cmd

    def process_actions(self, actions: torch.Tensor):
        actions = actions.clamp(-1.0, 1.0)
        self._raw_actions[:] = actions

        self._delta_cmd[:, 0] = actions[:, 0] * self.cfg.lin_xy_step[0]
        self._delta_cmd[:, 1] = actions[:, 1] * self.cfg.lin_xy_step[1]
        self._delta_cmd[:, 2] = actions[:, 2] * self.cfg.yaw_step

    def apply_actions(self):
        quat_w = self.robot.data.root_quat_w
        pos = self.robot.data.root_pos_w.clone()
        env_origins = self._env.scene.env_origins

        delta_body = torch.stack(
            (
                self._delta_cmd[:, 0] * self._substep_scale,
                self._delta_cmd[:, 1] * self._substep_scale,
                torch.zeros_like(self._delta_cmd[:, 0]),
            ),
            dim=-1,
        )
        pos += math_utils.quat_apply(quat_w, delta_body)

        if self.cfg.z_lock is not None:
            pos[:, 2] = self.cfg.z_lock
        if self.cfg.x_limits is not None:
            pos_local_x = (pos[:, 0] - env_origins[:, 0]).clamp(self.cfg.x_limits[0], self.cfg.x_limits[1])
            pos[:, 0] = env_origins[:, 0] + pos_local_x
        if self.cfg.y_limits is not None:
            pos_local_y = (pos[:, 1] - env_origins[:, 1]).clamp(self.cfg.y_limits[0], self.cfg.y_limits[1])
            pos[:, 1] = env_origins[:, 1] + pos_local_y

        rot_m = math_utils.matrix_from_quat(quat_w)
        yaw = torch.atan2(rot_m[:, 1, 0], rot_m[:, 0, 0]) + self._delta_cmd[:, 2] * self._substep_scale
        if self.cfg.yaw_limits is not None:
            yaw = yaw.clamp(self.cfg.yaw_limits[0], self.cfg.yaw_limits[1])
        zeros = torch.zeros_like(yaw)
        quat_yaw = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

        self.robot.write_root_pose_to_sim(torch.cat([pos, quat_yaw], dim=-1))
        self.robot.write_root_velocity_to_sim(torch.zeros(self.num_envs, 6, device=self.device))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._delta_cmd.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._delta_cmd[env_ids] = 0.0


@configclass
class PlanarPoseContinuousActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PlanarPoseContinuousAction

    asset_name: str = MISSING
    lin_xy_step: tuple[float, float] = (0.04, 0.04)
    yaw_step: float = math.radians(15.0)
    z_lock: float | None = 0.22
    x_limits: tuple[float, float] | None = None
    y_limits: tuple[float, float] | None = None
    yaw_limits: tuple[float, float] | None = None
