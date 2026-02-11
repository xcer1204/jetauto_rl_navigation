from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def goal_reached(
    env,
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
    return dist_xy <= threshold


def robot_out_of_bounds(
    env,
    x_limits: tuple[float, float] = (-1.2, 3.4),
    y_limits: tuple[float, float] = (-2.8, 2.2),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w - env.scene.env_origins
    x_ok = (robot_pos[:, 0] >= x_limits[0]) & (robot_pos[:, 0] <= x_limits[1])
    y_ok = (robot_pos[:, 1] >= y_limits[0]) & (robot_pos[:, 1] <= y_limits[1])
    return ~(x_ok & y_ok)
