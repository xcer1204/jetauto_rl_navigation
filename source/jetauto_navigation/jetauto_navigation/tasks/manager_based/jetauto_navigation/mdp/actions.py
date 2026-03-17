from __future__ import annotations

from dataclasses import MISSING
from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class PlanarVelocityAction(ActionTerm):
    """Maps policy actions to planar base velocity commands for Jetauto."""

    cfg: "PlanarVelocityActionCfg"

    def __init__(self, cfg: "PlanarVelocityActionCfg", env) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._cmd_body = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._cmd_body

    def process_actions(self, actions: torch.Tensor):
        actions = actions.clamp(-1.0, 1.0)
        self._raw_actions[:] = actions

        target_cmd = torch.zeros_like(self._cmd_body)
        target_cmd[:, 0] = actions[:, 0] * self.cfg.lin_xy_scale[0]
        target_cmd[:, 1] = actions[:, 1] * self.cfg.lin_xy_scale[1]
        target_cmd[:, 2] = actions[:, 2] * self.cfg.yaw_scale

        alpha = self.cfg.smoothing
        self._cmd_body[:] = (1.0 - alpha) * self._cmd_body + alpha * target_cmd

    def apply_actions(self):
        quat_w = self.robot.data.root_quat_w
        lin_vel_body = torch.stack((self._cmd_body[:, 0], self._cmd_body[:, 1], torch.zeros_like(self._cmd_body[:, 0])), dim=-1)
        lin_vel_world = math_utils.quat_apply(quat_w, lin_vel_body)

        root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        root_vel[:, 0] = lin_vel_world[:, 0]
        root_vel[:, 1] = lin_vel_world[:, 1]
        root_vel[:, 5] = self._cmd_body[:, 2]
        self.robot.write_root_velocity_to_sim(root_vel)

        if self.cfg.z_lock is not None:
            self._lock_to_plane(self.cfg.z_lock)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._cmd_body.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._cmd_body[env_ids] = 0.0

    def _lock_to_plane(self, z_lock: float) -> None:
        pos = self.robot.data.root_pos_w.clone()
        pos[:, 2] = z_lock

        rot_m = math_utils.matrix_from_quat(self.robot.data.root_quat_w)
        yaw = torch.atan2(rot_m[:, 1, 0], rot_m[:, 0, 0])
        zeros = torch.zeros_like(yaw)
        quat_yaw = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

        self.robot.write_root_pose_to_sim(torch.cat([pos, quat_yaw], dim=-1))


@configclass
class PlanarVelocityActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PlanarVelocityAction

    asset_name: str = MISSING
    lin_xy_scale: tuple[float, float] = (0.6, 0.6)
    yaw_scale: float = 1.5
    smoothing: float = 0.2
    z_lock: float | None = 0.01



class PlanarPoseStepAction(ActionTerm):
    """Maps policy actions to quantized planar pose steps for Jetauto.

    This keeps the policy interface at 3 continuous values, but each dimension is
    discretized internally into {-step, 0, +step}. The robot is then moved by a
    fixed body-frame translation and a fixed yaw increment each control step.
    """

    cfg: "PlanarPoseStepActionCfg"

    def __init__(self, cfg: "PlanarPoseStepActionCfg", env) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._step_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        # Action terms are applied every physics sub-step. Scale the commanded
        # pose delta so the total change over one environment step stays
        # constant when decimation changes.
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

        delta_body = torch.stack(
            (
                self._step_cmd[:, 0] * self._substep_scale,
                self._step_cmd[:, 1] * self._substep_scale,
                torch.zeros_like(self._step_cmd[:, 0]),
            ),
            # (self._step_cmd[:, 0], self._step_cmd[:, 1], torch.zeros_like(self._step_cmd[:, 0])),
            dim=-1,
        )
        delta_world = math_utils.quat_apply(quat_w, delta_body)
        pos += delta_world

        if self.cfg.z_lock is not None:
            pos[:, 2] = self.cfg.z_lock

        rot_m = math_utils.matrix_from_quat(quat_w)
        yaw = torch.atan2(rot_m[:, 1, 0], rot_m[:, 0, 0]) + self._step_cmd[:, 2] * self._substep_scale
        # yaw = torch.atan2(rot_m[:, 1, 0], rot_m[:, 0, 0]) + self._step_cmd[:, 2]
        zeros = torch.zeros_like(yaw)
        quat_yaw = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

        self.robot.write_root_pose_to_sim(torch.cat([pos, quat_yaw], dim=-1))

        root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self.robot.write_root_velocity_to_sim(root_vel)

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
    lin_xy_step: tuple[float, float] = (0.08, 0.08)
    yaw_step: float = 0.2617993877991494
    threshold: float = 0.33
    z_lock: float | None = 0.01