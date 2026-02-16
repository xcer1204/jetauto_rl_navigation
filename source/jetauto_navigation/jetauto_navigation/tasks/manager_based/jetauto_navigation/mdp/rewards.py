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
