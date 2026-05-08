from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg


def _sample_from_intervals(
    intervals: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    count: int,
    device,
) -> torch.Tensor:
    if not intervals:
        raise ValueError("Sampling interval list cannot be empty.")

    out = torch.empty(count, device=device)
    interval_ids = torch.randint(0, len(intervals), (count,), device=device)
    for idx, (low, high) in enumerate(intervals):
        mask = interval_ids == idx
        if bool(mask.any()):
            out[mask] = torch.empty(int(mask.sum().item()), device=device).uniform_(float(low), float(high))
    return out


def randomize_robot_and_cones(
    env,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    red_cfg: SceneEntityCfg = SceneEntityCfg("cone_red"),
    green_cfg: SceneEntityCfg = SceneEntityCfg("cone_green"),
    blue_cfg: SceneEntityCfg = SceneEntityCfg("cone_blue"),
    robot_x_range: tuple[float, float] = (0.4, 0.6),
    robot_y_range: tuple[float, float] = (-1.5, 0.5),
    robot_x_ranges: list[tuple[float, float]] | tuple[tuple[float, float], ...] | None = None,
    robot_y_ranges: list[tuple[float, float]] | tuple[tuple[float, float], ...] | None = None,
    robot_yaw_range: tuple[float, float] = (-math.pi, math.pi),
    red_x_range: tuple[float, float] = (1.8, 2.8),
    red_y_range: tuple[float, float] = (1.0, 1.6),
    green_x_range: tuple[float, float] = (1.8, 2.8),
    green_y_range: tuple[float, float] = (-0.2, 0.3),
    blue_x_range: tuple[float, float] = (1.8, 2.8),
    blue_y_range: tuple[float, float] = (-1.8, -1.2),
    cone_z: float = 0.03,
    cone_pose_ranges: dict[str, list[tuple[float, float]]] | None = None,
    z_lock: float = 0.01,
):
    if env_ids is None:
        return

    robot: Articulation = env.scene[robot_cfg.name]
    env_origins = env.scene.env_origins[env_ids]

    root_state = robot.data.default_root_state[env_ids].clone()
    root_x = (
        torch.empty(len(env_ids), device=env.device).uniform_(*robot_x_range)
        if robot_x_ranges is None
        else _sample_from_intervals(robot_x_ranges, len(env_ids), env.device)
    )
    root_y = (
        torch.empty(len(env_ids), device=env.device).uniform_(*robot_y_range)
        if robot_y_ranges is None
        else _sample_from_intervals(robot_y_ranges, len(env_ids), env.device)
    )
    root_state[:, 0] = env_origins[:, 0] + root_x
    root_state[:, 1] = env_origins[:, 1] + root_y
    root_state[:, 2] = env_origins[:, 2] + z_lock

    yaw = torch.empty(len(env_ids), device=env.device).uniform_(*robot_yaw_range)
    zeros = torch.zeros_like(yaw)
    root_state[:, 3:7] = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    root_state[:, 7:] = 0.0

    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)

    if cone_pose_ranges is not None:
        _randomize_cones_with_region_permutation(env, env_ids, red_cfg, green_cfg, blue_cfg, cone_pose_ranges)
    else:
        _randomize_single_cone(env, env_ids, red_cfg, red_x_range, red_y_range, cone_z)
        _randomize_single_cone(env, env_ids, green_cfg, green_x_range, green_y_range, cone_z)
        _randomize_single_cone(env, env_ids, blue_cfg, blue_x_range, blue_y_range, cone_z)


def _randomize_single_cone(
    env,
    env_ids: torch.Tensor,
    cone_cfg: SceneEntityCfg,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z: float,
):
    cone: RigidObjectCollection = env.scene[cone_cfg.name]
    state = cone.data.default_object_state[env_ids].clone()
    env_origins = env.scene.env_origins[env_ids]
    state[:, 0, 0] = env_origins[:, 0] + torch.empty(len(env_ids), device=env.device).uniform_(*x_range)
    state[:, 0, 1] = env_origins[:, 1] + torch.empty(len(env_ids), device=env.device).uniform_(*y_range)
    state[:, 0, 2] = env_origins[:, 2] + z
    state[:, 0, 3] = 1.0
    state[:, 0, 4:7] = 0.0
    state[:, 0, 7:] = 0.0
    cone.write_object_state_to_sim(state, env_ids=env_ids)


def _randomize_cones_with_region_permutation(
    env,
    env_ids: torch.Tensor,
    red_cfg: SceneEntityCfg,
    green_cfg: SceneEntityCfg,
    blue_cfg: SceneEntityCfg,
    pose_ranges: dict[str, list[tuple[float, float]]],
):
    x_ranges = pose_ranges.get("x", [])
    y_ranges = pose_ranges.get("y", [])
    z_ranges = pose_ranges.get("z", [])
    if len(x_ranges) != 3 or len(y_ranges) != 3 or len(z_ranges) != 3:
        raise ValueError("cone_pose_ranges must provide exactly 3 ranges for x/y/z.")

    device = env.device
    num_envs = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]
    slot_pos = torch.zeros(num_envs, 3, 3, device=device)
    for slot in range(3):
        slot_pos[:, slot, 0] = torch.empty(num_envs, device=device).uniform_(*x_ranges[slot])
        slot_pos[:, slot, 1] = torch.empty(num_envs, device=device).uniform_(*y_ranges[slot])
        slot_pos[:, slot, 2] = torch.empty(num_envs, device=device).uniform_(*z_ranges[slot])

    rgb_pos = torch.zeros(num_envs, 3, 3, device=device)
    for i in range(num_envs):
        rgb_pos[i, 0] = slot_pos[i, 0]
        rgb_pos[i, 1] = slot_pos[i, 1]
        rgb_pos[i, 2] = slot_pos[i, 2]

    _write_cone_states(env, env_ids, red_cfg, env_origins, rgb_pos[:, 0, :])
    _write_cone_states(env, env_ids, green_cfg, env_origins, rgb_pos[:, 1, :])
    _write_cone_states(env, env_ids, blue_cfg, env_origins, rgb_pos[:, 2, :])


def _write_cone_states(
    env,
    env_ids: torch.Tensor,
    cone_cfg: SceneEntityCfg,
    env_origins: torch.Tensor,
    positions_local: torch.Tensor,
):
    cone: RigidObjectCollection = env.scene[cone_cfg.name]
    state = cone.data.default_object_state[env_ids].clone()
    state[:, 0, 0:3] = env_origins + positions_local
    state[:, 0, 3] = 1.0
    state[:, 0, 4:7] = 0.0
    state[:, 0, 7:] = 0.0
    cone.write_object_state_to_sim(state, env_ids=env_ids)
