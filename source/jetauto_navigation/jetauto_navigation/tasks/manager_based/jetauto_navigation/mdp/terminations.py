from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from .multitask_inference import DEFAULT_OCCLUSION_CLASS_NAMES


def _get_predicted_occlusion_classes(env) -> torch.Tensor:
    pred = env.extras.get("pred_occ_class", None)
    if pred is None:
        return torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    pred_t = torch.as_tensor(pred, device=env.device, dtype=torch.long).reshape(-1)
    if pred_t.numel() == 1:
        pred_t = pred_t.repeat(env.num_envs)
    elif pred_t.numel() != env.num_envs:
        pred_t = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    return pred_t


def _resolve_success_class_index(env, success_class_name: str) -> int:
    class_names = env.extras.get("pred_occ_class_names", DEFAULT_OCCLUSION_CLASS_NAMES)
    class_names = tuple(str(name) for name in class_names)
    try:
        return class_names.index(str(success_class_name))
    except ValueError:
        return 0


# def goal_reached(
#     env,
#     command_name: str,
#     threshold: float = 0.35,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     robot: Articulation = env.scene[asset_cfg.name]
#     robot_pos = robot.data.root_pos_w - env.scene.env_origins

#     cmd = env.command_manager.get_command(command_name)
#     red = env.scene["cone_red"].data.object_pos_w[:, 0, :] - env.scene.env_origins
#     green = env.scene["cone_green"].data.object_pos_w[:, 0, :] - env.scene.env_origins
#     blue = env.scene["cone_blue"].data.object_pos_w[:, 0, :] - env.scene.env_origins
#     goals = torch.stack([red, green, blue], dim=1)
#     goal_pos = (goals * cmd.unsqueeze(-1)).sum(dim=1)

#     dist_xy = torch.norm(goal_pos[:, :2] - robot_pos[:, :2], dim=1)
#     return dist_xy <= threshold


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


def visibility_success(
    env,
    threshold: float = 0.9,
    success_class_name: str = "0-20%",
    x_limits: tuple[float, float] = (-0.1, 3.1),
    y_limits: tuple[float, float] = (-3.4, 1.8),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the multitask model predicts the best occlusion bucket in bounds."""
    del threshold

    pred_occ = _get_predicted_occlusion_classes(env)
    success_class_index = _resolve_success_class_index(env, success_class_name)
    failed = robot_out_of_bounds(env, x_limits=x_limits, y_limits=y_limits, asset_cfg=asset_cfg)
    return (pred_occ == success_class_index) & (~failed)
