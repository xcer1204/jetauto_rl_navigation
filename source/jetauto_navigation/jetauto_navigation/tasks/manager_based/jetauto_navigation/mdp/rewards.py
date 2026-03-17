# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from .multitask_inference import DEFAULT_OCCLUSION_CLASS_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def goal_distance_tanh(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w - env.scene.env_origins

    cmd = env.command_manager.get_command(command_name)
    red = env.scene["cone_red"].data.object_pos_w[:, 0, :] - env.scene.env_origins
    green = env.scene["cone_green"].data.object_pos_w[:, 0, :] - env.scene.env_origins
    blue = env.scene["cone_blue"].data.object_pos_w[:, 0, :] - env.scene.env_origins
    goals = torch.stack([red, green, blue], dim=1)
    goal_pos = (goals * cmd.unsqueeze(-1)).sum(dim=1)

    dist = torch.norm(goal_pos[:, :2] - robot_pos[:, :2], dim=1)
    return 1.0 - torch.tanh(dist / std)


def goal_heading_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w - env.scene.env_origins
    robot_quat = robot.data.root_quat_w

    cmd = env.command_manager.get_command(command_name)
    red = env.scene["cone_red"].data.object_pos_w[:, 0, :] - env.scene.env_origins
    green = env.scene["cone_green"].data.object_pos_w[:, 0, :] - env.scene.env_origins
    blue = env.scene["cone_blue"].data.object_pos_w[:, 0, :] - env.scene.env_origins
    goals = torch.stack([red, green, blue], dim=1)
    goal_pos = (goals * cmd.unsqueeze(-1)).sum(dim=1)

    delta = goal_pos[:, :2] - robot_pos[:, :2]
    desired_yaw = torch.atan2(delta[:, 1], delta[:, 0])

    forward_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    forward = math_utils.quat_apply(robot_quat, forward_axis)
    current_yaw = torch.atan2(forward[:, 1], forward[:, 0])
    yaw_error = wrap_to_pi(desired_yaw - current_yaw)
    return 1.0 - torch.abs(yaw_error) / torch.pi


def goal_reach_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w - env.scene.env_origins

    cmd = env.command_manager.get_command(command_name)
    red = env.scene["cone_red"].data.object_pos_w[:, 0, :] - env.scene.env_origins
    green = env.scene["cone_green"].data.object_pos_w[:, 0, :] - env.scene.env_origins
    blue = env.scene["cone_blue"].data.object_pos_w[:, 0, :] - env.scene.env_origins
    goals = torch.stack([red, green, blue], dim=1)
    goal_pos = (goals * cmd.unsqueeze(-1)).sum(dim=1)

    dist_xy = torch.norm(goal_pos[:, :2] - robot_pos[:, :2], dim=1)
    return (dist_xy <= threshold).float()

def target_visibility_reward(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward proportional to target visibility ratio in [0,1]."""
    vis = env.extras.get("vis_ratio", None)
    if vis is None:
        return torch.zeros(env.num_envs, device=env.device)
    return scale * vis


def _get_predicted_occlusion_classes(env: ManagerBasedRLEnv) -> torch.Tensor:
    pred = env.extras.get("pred_occ_class", None)
    if pred is None:
        return torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    pred_t = torch.as_tensor(pred, device=env.device, dtype=torch.long).reshape(-1)
    if pred_t.numel() == 1:
        pred_t = pred_t.repeat(env.num_envs)
    elif pred_t.numel() != env.num_envs:
        pred_t = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    return pred_t


def _resolve_success_class_index(env: ManagerBasedRLEnv, success_class_name: str) -> int:
    class_names = env.extras.get("pred_occ_class_names", DEFAULT_OCCLUSION_CLASS_NAMES)
    class_names = tuple(str(name) for name in class_names)
    try:
        return class_names.index(str(success_class_name))
    except ValueError:
        return 0

def visibility_progress_reward(
    env: ManagerBasedRLEnv,
    success_threshold: float = 0.7,
    success_bonus: float = 5.0,
    success_class_name: str = "0-20%",
    idle_penalty: float = -0.01,
    collision_penalty: float = -5.0,
    x_limits: tuple[float, float] = (-0.1, 3.1),
    y_limits: tuple[float, float] = (-3.4, 1.8),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward success when the multitask model predicts the best occlusion bucket."""
    del success_threshold

    pred_occ = _get_predicted_occlusion_classes(env)
    success_class_index = _resolve_success_class_index(env, success_class_name)
    success_mask = pred_occ == success_class_index
    visibility_term = torch.where(
        success_mask,
        torch.full((env.num_envs,), success_bonus, device=env.device, dtype=torch.float32),
        torch.full((env.num_envs,), idle_penalty, device=env.device, dtype=torch.float32),
    )

    robot: Articulation = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w - env.scene.env_origins
    in_bounds_x = (robot_pos[:, 0] >= x_limits[0]) & (robot_pos[:, 0] <= x_limits[1])
    in_bounds_y = (robot_pos[:, 1] >= y_limits[0]) & (robot_pos[:, 1] <= y_limits[1])
    collision_mask = ~(in_bounds_x & in_bounds_y)
    collision_term = torch.where(
        collision_mask,
        torch.full_like(visibility_term, collision_penalty),
        torch.zeros_like(visibility_term),
    )

    reward = visibility_term + collision_term

    env.extras["occ_ratio_mean"] = 0.0
    env.extras["delta_occ_mean"] = 0.0
    env.extras["visibility_term_mean"] = float(visibility_term.mean().item())
    env.extras["collision_penalty_mean"] = float(collision_term.mean().item())
    env.extras["visibility_success"] = bool(success_mask.any().item())
    env.extras["pred_occ_success"] = bool(success_mask.any().item())

    return reward
