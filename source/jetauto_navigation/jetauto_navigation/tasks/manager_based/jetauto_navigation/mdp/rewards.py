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

def visibility_progress_reward(
    env: ManagerBasedRLEnv,
    success_threshold: float = 0.7,
    success_bonus: float = 5.0,
    idle_penalty: float = -0.01,
    collision_penalty: float = -5.0,
    x_limits: tuple[float, float] = (-0.1, 3.1),
    y_limits: tuple[float, float] = (-3.4, 1.8),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward visibility improvement, successful visibility, and boundary collisions.

    This mirrors the previous direct-environment logic:
    - Use the change in occlusion ratio as a dense term.
    - Give a success bonus when visibility exceeds the threshold.
    - Penalize leaving the valid area (used here as the closest proxy for wall collision).
    """
    vis = env.extras.get("vis_ratio", None)
    if vis is None:
        vis_t = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    else:
        vis_t = torch.as_tensor(vis, device=env.device, dtype=torch.float32).reshape(-1)
        if vis_t.numel() == 1:
            vis_t = vis_t.repeat(env.num_envs)
        elif vis_t.numel() != env.num_envs:
            vis_t = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    vis_t = vis_t.clamp(0.0, 1.0)

    prev_vis = getattr(env, "_prev_vis_ratio", None)
    if (
        prev_vis is None
        or not isinstance(prev_vis, torch.Tensor)
        or prev_vis.shape != vis_t.shape
        or prev_vis.device != vis_t.device
    ):
        prev_vis = torch.zeros_like(vis_t)
    else:
        prev_vis = prev_vis.clone()

    # Freshly reset environments should start from the same baseline as the old direct env.
    reset_mask = env.episode_length_buf == 0
    prev_vis[reset_mask] = 0.0

    occ_t = (1.0 - vis_t).clamp(0.0, 1.0)
    occ_prev = (1.0 - prev_vis).clamp(0.0, 1.0)
    delta_occ = occ_prev - occ_t

    visibility_term = torch.where(
        vis_t > success_threshold,
        torch.full_like(vis_t, success_bonus),
        torch.full_like(vis_t, idle_penalty),
    )

    robot: Articulation = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w - env.scene.env_origins
    in_bounds_x = (robot_pos[:, 0] >= x_limits[0]) & (robot_pos[:, 0] <= x_limits[1])
    in_bounds_y = (robot_pos[:, 1] >= y_limits[0]) & (robot_pos[:, 1] <= y_limits[1])
    collision_mask = ~(in_bounds_x & in_bounds_y)
    collision_term = torch.where(
        collision_mask,
        torch.full_like(vis_t, collision_penalty),
        torch.zeros_like(vis_t),
    )


    # reward = delta_occ + visibility_term + collision_term

    reward = visibility_term + collision_term

    env._prev_vis_ratio = vis_t.detach().clone()
    env.extras["occ_ratio_mean"] = float(occ_t.mean().item())
    env.extras["delta_occ_mean"] = float(delta_occ.mean().item())
    env.extras["visibility_term_mean"] = float(visibility_term.mean().item())
    env.extras["collision_penalty_mean"] = float(collision_term.mean().item())
    env.extras["visibility_success"] = bool((vis_t > success_threshold).any().item())

    # print(f"[DEBUG] delta_occ: {delta_occ.mean().item():.4f}, visibility_term: {visibility_term.mean().item():.4f}, collision_term: {collision_term.mean().item():.4f}"
    # )
    return reward
