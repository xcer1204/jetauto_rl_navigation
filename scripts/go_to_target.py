# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to drive Jetauto to a target position in the empty single-room env."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import os

from isaaclab.app import AppLauncher

DEFAULT_VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_a.mp4")

# add argparse arguments
parser = argparse.ArgumentParser(description="Drive Jetauto to a target position without training.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Jetauto-Navigation-Direct-SingleRoom-Empty-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--target",
    type=float,
    nargs=3,
    default=(0.8, 0.0, 0.0),
    help="Target position (x y z) in env frame unless --target_frame=world.",
)
parser.add_argument(
    "--target_frame",
    choices=("env", "world"),
    default="env",
    help="Interpret target coordinates in the env frame or world frame.",
)
parser.add_argument(
    "--target_quat",
    type=float,
    nargs=4,
    default=None,
    help="Target orientation quaternion (w x y z) in world frame. Defaults to yaw facing origin.",
)
parser.add_argument(
    "--target_quat_order",
    choices=("wxyz", "xyzw"),
    default="wxyz",
    help="Order of --target_quat components.",
)
parser.add_argument("--reach_threshold", type=float, default=0.1, help="Stop when distance < threshold (m).")
parser.add_argument("--kv", type=float, default=1.2, help="Linear velocity gain.")
parser.add_argument("--kw", type=float, default=2.0, help="Yaw rate gain.")
parser.add_argument("--speed_scale", type=float, default=0.5, help="Scale max linear/yaw speed (1.0 = full).")
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=30.0,
    help="Episode length in seconds (increase to avoid early resets).",
)
parser.add_argument("--stop_on_reach", action="store_true", help="Exit when all envs reach the target.")
parser.add_argument("--video", action="store_true", default=False, help="Record camera_a RGB video.")
parser.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH, help="Output video path.")
parser.add_argument("--video_fps", type=int, default=0, help="Video FPS (0 = auto from env step).")
parser.add_argument("--video_env_id", type=int, default=0, help="Env index to record from.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import jetauto_navigation.tasks  # noqa: F401
try:
    import imageio.v2 as imageio
except ImportError:
    import imageio


def _target_in_world(env, target_xyz, target_frame):
    device = env.unwrapped.device
    target = torch.tensor(target_xyz, device=device, dtype=torch.float32).view(1, 3)
    target = target.repeat(env.unwrapped.num_envs, 1)
    if target_frame == "env":
        target = target + env.unwrapped.scene.env_origins
    return target


def _sync_logical_objects(env, target_world):
    if hasattr(env.unwrapped, "_target_pos_current"):
        env.unwrapped._target_pos_current.copy_(target_world)
    if hasattr(env.unwrapped, "_obstacle_pos_current"):
        offset = torch.tensor([100.0, 100.0, 0.0], device=target_world.device)
        env.unwrapped._obstacle_pos_current.copy_(target_world + offset)


def _target_quat_in_world(env, quat, order):
    if quat is None:
        return None
    device = env.unwrapped.device
    quat_tensor = torch.tensor(quat, device=device, dtype=torch.float32).view(1, 4)
    if order == "xyzw":
        quat_tensor = quat_tensor[:, [3, 0, 1, 2]]
    quat_tensor = quat_tensor.repeat(env.unwrapped.num_envs, 1)
    quat_tensor = quat_tensor / quat_tensor.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return quat_tensor


def _default_target_quat(env, target_xyz, target_frame, target_world):
    device = env.unwrapped.device
    if target_frame == "env":
        target_xy = torch.tensor(target_xyz[:2], device=device, dtype=torch.float32).view(1, 2)
        target_xy = target_xy.repeat(env.unwrapped.num_envs, 1)
    else:
        target_xy = target_world[:, :2]
    yaw = torch.atan2(-target_xy[:, 1], -target_xy[:, 0])
    half = 0.5 * yaw
    quat = torch.stack(
        [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1
    )
    return quat


def _yaw_from_quat(quat_wxyz):
    rot = math_utils.matrix_from_quat(quat_wxyz)
    return torch.atan2(rot[:, 1, 0], rot[:, 0, 0])


def _resolve_step_dt(env):
    if hasattr(env, "step_dt"):
        return env.step_dt
    if hasattr(env.unwrapped, "step_dt"):
        return env.unwrapped.step_dt
    return None


def _get_camera_rgb(env, env_id):
    if not hasattr(env.unwrapped, "_camera_a"):
        return None
    rgb = env.unwrapped._camera_a.data.output["rgb"][env_id]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        if np.nanmax(rgb) <= 1.5:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb


def main():
    """Drive to a fixed target using a simple proportional controller."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    writer = None
    video_env_id = None
    if args_cli.video:
        video_env_id = max(0, min(args_cli.video_env_id, env.unwrapped.num_envs - 1))
        step_dt = _resolve_step_dt(env)
        fps = args_cli.video_fps
        if fps <= 0:
            fps = int(round(1.0 / step_dt)) if step_dt else 30
        writer = imageio.get_writer(args_cli.video_path, fps=fps)
        first_frame = _get_camera_rgb(env, video_env_id)
        if first_frame is not None:
            writer.append_data(first_frame)

    target_world = _target_in_world(env, args_cli.target, args_cli.target_frame)
    target_quat_world = _target_quat_in_world(env, args_cli.target_quat, args_cli.target_quat_order)
    if target_quat_world is None:
        target_quat_world = _default_target_quat(env, args_cli.target, args_cli.target_frame, target_world)
    _sync_logical_objects(env, target_world)

    action_shape = env.action_space.shape
    if len(action_shape) == 1:
        action_shape = (env.unwrapped.num_envs, action_shape[0])
    action = torch.zeros(action_shape, device=env.unwrapped.device)

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                _sync_logical_objects(env, target_world)

                robot_pos = env.unwrapped.robot_a.data.root_pos_w
                robot_quat = env.unwrapped.robot_a.data.root_quat_w

                delta_xy = target_world[:, :2] - robot_pos[:, :2]
                dist = torch.norm(delta_xy, dim=-1)

                max_v = env.unwrapped.v_xy_max * max(args_cli.speed_scale, 0.0)
                v_mag = torch.clamp(args_cli.kv * dist, max=max_v)
                v_dir = delta_xy / dist.unsqueeze(-1).clamp(min=1e-6)
                v_world_xy = v_dir * v_mag.unsqueeze(-1)
                v_world = torch.cat([v_world_xy, torch.zeros_like(v_mag).unsqueeze(-1)], dim=-1)

                v_body = math_utils.quat_apply(math_utils.quat_inv(robot_quat), v_world)

                action.zero_()
                action[:, 0] = (v_body[:, 0] / env.unwrapped.v_xy_max).clamp(-1.0, 1.0)
                action[:, 1] = (v_body[:, 1] / env.unwrapped.v_xy_max).clamp(-1.0, 1.0)

                rot = math_utils.matrix_from_quat(robot_quat)
                yaw = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
                desired_yaw = _yaw_from_quat(target_quat_world)
                yaw_err = desired_yaw - yaw
                yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
                max_w = env.unwrapped.w_z_max * max(args_cli.speed_scale, 0.0)
                w_des = torch.clamp(args_cli.kw * yaw_err, min=-max_w, max=max_w)
                action[:, 2] = (w_des / env.unwrapped.w_z_max).clamp(-1.0, 1.0)

                reached = dist < args_cli.reach_threshold
                if reached.any():
                    action[reached] = 0.0

                env.step(action)

                if writer is not None:
                    frame = _get_camera_rgb(env, video_env_id)
                    if frame is not None:
                        writer.append_data(frame)

                if args_cli.stop_on_reach and bool(reached.all().item()):
                    break
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
