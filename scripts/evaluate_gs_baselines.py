from __future__ import annotations

import argparse
import json
import os
import random
import time
import traceback
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate simple action baselines on a GS-based Jetauto task.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Override the number of environments.")
parser.add_argument("--task", type=str, default="Jetauto-VRRobo-Manager-MidOcc-v0", help="Gym task name.")
parser.add_argument(
    "--baseline",
    type=str,
    default="static",
    choices=["static", "random", "sweep"],
    help="Baseline policy to evaluate.",
)
parser.add_argument("--seed", type=int, default=None, help="Evaluation seed. Use -1 for a random seed.")
parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate.")
parser.add_argument("--max_steps", type=int, default=None, help="Optional max env-steps per episode.")
parser.add_argument(
    "--heartbeat_interval_s",
    type=float,
    default=60.0,
    help="Print a progress heartbeat while long-running episodes are still active.",
)
parser.add_argument(
    "--success_class_name",
    type=str,
    default=None,
    help="Predicted occlusion class counted as success. If omitted, read it from the environment config.",
)
parser.add_argument(
    "--x_limits",
    type=float,
    nargs=2,
    default=None,
    metavar=("X_MIN", "X_MAX"),
    help="Valid x-range used for out-of-bounds accounting. If omitted, read it from the environment config.",
)
parser.add_argument(
    "--y_limits",
    type=float,
    nargs=2,
    default=None,
    metavar=("Y_MIN", "Y_MAX"),
    help="Valid y-range used for out-of-bounds accounting. If omitted, read it from the environment config.",
)
parser.add_argument(
    "--policy_term",
    type=str,
    default="gs_image",
    help="Policy observation term whose GS network endpoint parameters should be overridden.",
)
parser.add_argument(
    "--output_json",
    type=str,
    default=None,
    help="Optional path to save evaluation metrics as JSON.",
)
parser.add_argument("--render_server_host", type=str, default=None, help="Override the GS render server host.")
parser.add_argument("--render_server_port", type=int, default=None, help="Override the GS render server RPC port.")
parser.add_argument("--rgb_socket_host", type=str, default=None, help="Override the GS RGB receiver host.")
parser.add_argument("--rgb_socket_port", type=int, default=None, help="Override the GS RGB receiver TCP port.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _needs_local_cameras(task_name: str, policy_term: str) -> bool:
    if "IsaacRGB" in task_name:
        return True
    if policy_term == "rgb_feature":
        return True
    return False


args_cli.enable_cameras = _needs_local_cameras(args_cli.task, args_cli.policy_term)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import jetauto_navigation  # noqa: F401


def _override_policy_term_network_endpoints(
    env_cfg,
    policy_term_name: str,
    render_server_host: str | None,
    render_server_port: int | None,
    rgb_socket_host: str | None,
    rgb_socket_port: int | None,
) -> bool:
    overrides = {}
    if render_server_host is not None:
        overrides["render_server_host"] = str(render_server_host)
    if render_server_port is not None:
        overrides["render_server_port"] = int(render_server_port)
    if rgb_socket_host is not None:
        overrides["rgb_socket_host"] = str(rgb_socket_host)
    if rgb_socket_port is not None:
        overrides["rgb_socket_port"] = int(rgb_socket_port)
    if not overrides:
        return False

    observations_cfg = getattr(env_cfg, "observations", None)
    policy_cfg = getattr(observations_cfg, "policy", None) if observations_cfg is not None else None
    if policy_cfg is None:
        return False

    term_cfg = None
    selected_name = None
    for candidate in dict.fromkeys([policy_term_name, "gs_image"]):
        if not candidate:
            continue
        term_cfg = getattr(policy_cfg, candidate, None)
        if term_cfg is not None:
            selected_name = candidate
            break
    if term_cfg is None:
        return False

    params = getattr(term_cfg, "params", None)
    if params is None:
        params = {}
        setattr(term_cfg, "params", params)
    elif not isinstance(params, dict):
        params = dict(params)
        setattr(term_cfg, "params", params)

    previous = {key: params.get(key) for key in overrides}
    params.update(overrides)
    print(
        "[evaluate_gs_baselines] Overriding GS network endpoints "
        f"term={selected_name} previous={previous} current={overrides}",
        flush=True,
    )
    return True


def _get_term_params(cfg_section, term_name: str) -> dict:
    if cfg_section is None:
        return {}
    term_cfg = getattr(cfg_section, term_name, None)
    params = getattr(term_cfg, "params", None)
    if params is None:
        return {}
    return dict(params)


def _resolve_eval_settings(env_cfg) -> tuple[str, tuple[float, float], tuple[float, float]]:
    termination_cfg = getattr(env_cfg, "terminations", None)
    rewards_cfg = getattr(env_cfg, "rewards", None)

    visibility_success_params = _get_term_params(termination_cfg, "visibility_success")
    out_of_bounds_params = _get_term_params(termination_cfg, "out_of_bounds")
    reward_params = _get_term_params(rewards_cfg, "visibility_progress")

    success_class_name = args_cli.success_class_name
    if success_class_name is None:
        success_class_name = visibility_success_params.get(
            "success_class_name",
            reward_params.get("success_class_name", "0-20%"),
        )

    x_limits = args_cli.x_limits
    if x_limits is None:
        x_limits = out_of_bounds_params.get("x_limits", visibility_success_params.get("x_limits"))
    if x_limits is None:
        x_limits = reward_params.get("x_limits", (-0.1, 3.1))

    y_limits = args_cli.y_limits
    if y_limits is None:
        y_limits = out_of_bounds_params.get("y_limits", visibility_success_params.get("y_limits"))
    if y_limits is None:
        y_limits = reward_params.get("y_limits", (-3.4, 1.8))

    return str(success_class_name), tuple(float(v) for v in x_limits), tuple(float(v) for v in y_limits)


def _to_1d_tensor(value, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    return tensor.reshape(-1)


def _get_class_names(base_env) -> tuple[str, ...]:
    class_names = base_env.extras.get("pred_occ_class_names", ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%"))
    return tuple(str(name) for name in class_names)


def _get_predicted_occlusion_classes(base_env, num_envs: int, device: torch.device | str) -> torch.Tensor:
    pred = base_env.extras.get("pred_occ_class", None)
    if pred is None:
        return torch.full((num_envs,), -1, device=device, dtype=torch.long)

    pred_t = _to_1d_tensor(pred, device=device, dtype=torch.long)
    if pred_t.numel() == 1:
        pred_t = pred_t.repeat(num_envs)
    elif pred_t.numel() != num_envs:
        pred_t = torch.full((num_envs,), -1, device=device, dtype=torch.long)
    return pred_t


def _resolve_success_class_index(base_env, success_class_name: str) -> int:
    try:
        return _get_class_names(base_env).index(str(success_class_name))
    except ValueError:
        return 0


def _compute_out_of_bounds_mask(base_env, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> torch.Tensor:
    robot = base_env.scene["robot"]
    robot_pos = robot.data.root_pos_w - base_env.scene.env_origins
    in_bounds_x = (robot_pos[:, 0] >= x_limits[0]) & (robot_pos[:, 0] <= x_limits[1])
    in_bounds_y = (robot_pos[:, 1] >= y_limits[0]) & (robot_pos[:, 1] <= y_limits[1])
    return ~(in_bounds_x & in_bounds_y)


def _get_termination_term_mask(base_env, term_name: str, num_envs: int, device: torch.device | str) -> torch.Tensor | None:
    termination_manager = getattr(base_env, "termination_manager", None)
    if termination_manager is None:
        return None

    try:
        term_value = termination_manager.get_term(term_name)
    except Exception:
        return None

    return _to_1d_tensor(term_value, device=device, dtype=torch.bool)


def _baseline_actions(
    baseline: str,
    episode_steps_since_reset: torch.Tensor,
    action_shape: tuple[int, ...],
    device: torch.device | str,
    generator: torch.Generator,
) -> torch.Tensor:
    if baseline == "static":
        return torch.zeros(action_shape, device=device, dtype=torch.float32)

    if baseline == "random":
        choices = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=torch.float32)
        indices = torch.randint(0, 3, action_shape, generator=generator, device="cpu").to(device=device)
        return choices[indices]

    if baseline == "sweep":
        pattern = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ],
            device=device,
            dtype=torch.float32,
        )
        phase = episode_steps_since_reset.to(device=device, dtype=torch.long) % pattern.shape[0]
        actions = pattern[phase]
        if tuple(actions.shape) != tuple(action_shape):
            actions = actions.reshape(action_shape)
        return actions

    raise ValueError(f"Unsupported baseline: {baseline}")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=False if args_cli.disable_fabric else None,
    )
    if not _override_policy_term_network_endpoints(
        env_cfg,
        args_cli.policy_term,
        args_cli.render_server_host,
        args_cli.render_server_port,
        args_cli.rgb_socket_host,
        args_cli.rgb_socket_port,
    ) and any(
        value is not None
        for value in (
            args_cli.render_server_host,
            args_cli.render_server_port,
            args_cli.rgb_socket_host,
            args_cli.rgb_socket_port,
        )
    ):
        print(
            "[evaluate_gs_baselines] GS network endpoint override skipped "
            f"because neither '{args_cli.policy_term}' nor 'gs_image' was found in env_cfg.observations.policy.",
            flush=True,
        )

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    if args_cli.seed is not None:
        random.seed(args_cli.seed)
        np.random.seed(args_cli.seed)
        torch.manual_seed(args_cli.seed)
        env_cfg.seed = args_cli.seed

    log_root_path = os.path.abspath(os.path.join("logs", "skrl", "jetauto_vrrobo_manager", "baselines"))
    env_cfg.log_dir = log_root_path

    print("[evaluate_gs_baselines] Creating gym environment...", flush=True)
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    except BaseException as exc:
        print(
            f"[evaluate_gs_baselines] gym.make raised {type(exc).__name__}: {exc!r}",
            flush=True,
        )
        print(traceback.format_exc(), flush=True)
        raise
    print("[evaluate_gs_baselines] gym environment created.", flush=True)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        print("[evaluate_gs_baselines] Converted DirectMARLEnv to single-agent wrapper.", flush=True)

    num_envs = int(getattr(env.unwrapped, "num_envs", getattr(env, "num_envs", 1)))
    device = getattr(env.unwrapped, "device", "cpu")
    success_class_name, x_limits, y_limits = _resolve_eval_settings(env_cfg)
    action_shape = tuple(int(v) for v in env.action_space.shape)
    if not action_shape:
        action_shape = (num_envs, 1)
    if action_shape[0] != num_envs:
        action_shape = (num_envs, *action_shape)

    print(f"[INFO] Baseline: {args_cli.baseline}")
    print(f"[INFO] Evaluation success class: {success_class_name}")
    print(f"[INFO] Evaluation bounds: x={x_limits}, y={y_limits}")
    print(f"[INFO] Action shape: {action_shape}")

    episode_returns = torch.zeros(num_envs, device=device, dtype=torch.float32)
    episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
    episode_steps_since_reset = torch.zeros(num_envs, device=device, dtype=torch.long)

    finished_returns: list[float] = []
    finished_lengths: list[int] = []
    finished_success: list[int] = []
    finished_occ_classes: list[int] = []
    reason_counter = {"success": 0, "out_of_bounds": 0, "time_out": 0, "other_terminated": 0}

    generator = torch.Generator(device="cpu")
    if args_cli.seed is not None:
        generator.manual_seed(args_cli.seed)
    else:
        generator.seed()

    print("[evaluate_gs_baselines] Resetting environment...", flush=True)
    env.reset()
    print("[evaluate_gs_baselines] Environment reset complete.", flush=True)

    eval_started_at = time.time()
    total_env_steps = 0
    next_progress = max(1, args_cli.episodes // 10)
    progress_interval = max(1, args_cli.episodes // 10)
    last_heartbeat_at = eval_started_at
    while simulation_app.is_running() and len(finished_returns) < args_cli.episodes:
        with torch.inference_mode():
            actions = _baseline_actions(
                args_cli.baseline,
                episode_steps_since_reset,
                action_shape,
                device,
                generator,
            )
            _, reward, terminated, truncated, _ = env.step(actions)

        total_env_steps += 1
        reward_t = _to_1d_tensor(reward, device=device, dtype=torch.float32)
        term_t = _to_1d_tensor(terminated, device=device, dtype=torch.bool)
        trunc_t = _to_1d_tensor(truncated, device=device, dtype=torch.bool)

        episode_returns += reward_t
        episode_lengths += 1
        episode_steps_since_reset += 1

        done_t = term_t | trunc_t
        if args_cli.max_steps is not None and args_cli.max_steps > 0:
            max_step_done = episode_steps_since_reset >= int(args_cli.max_steps)
            trunc_t = trunc_t | max_step_done
            done_t = term_t | trunc_t

        if not done_t.any():
            now = time.time()
            if args_cli.heartbeat_interval_s > 0 and now - last_heartbeat_at >= args_cli.heartbeat_interval_s:
                active_lengths = episode_lengths[episode_lengths > 0]
                active_avg = float(active_lengths.float().mean().item()) if active_lengths.numel() else 0.0
                print(
                    f"[EVAL] heartbeat env_steps={total_env_steps} "
                    f"finished={len(finished_returns)}/{args_cli.episodes} "
                    f"active_avg_len={active_avg:.1f} wall_time_s={now - eval_started_at:.1f}",
                    flush=True,
                )
                last_heartbeat_at = now
            continue

        base_env = env.unwrapped
        success_mask = _get_termination_term_mask(base_env, "visibility_success", num_envs=num_envs, device=device)
        out_of_bounds = _get_termination_term_mask(base_env, "out_of_bounds", num_envs=num_envs, device=device)
        pred_occ = _get_predicted_occlusion_classes(base_env, num_envs=num_envs, device=device)

        if success_mask is None or out_of_bounds is None:
            success_class_index = _resolve_success_class_index(base_env, success_class_name)
            out_of_bounds = _compute_out_of_bounds_mask(base_env, x_limits, y_limits)
            success_mask = (pred_occ == success_class_index) & (~out_of_bounds)

        done_ids = torch.nonzero(done_t, as_tuple=False).reshape(-1).tolist()
        for idx in done_ids:
            if len(finished_returns) >= args_cli.episodes:
                break

            finished_returns.append(float(episode_returns[idx].item()))
            finished_lengths.append(int(episode_lengths[idx].item()))

            success = bool(success_mask[idx].item())
            finished_success.append(1 if success else 0)
            finished_occ_classes.append(int(pred_occ[idx].item()))

            if success:
                reason_counter["success"] += 1
            elif bool(out_of_bounds[idx].item()):
                reason_counter["out_of_bounds"] += 1
            elif bool(trunc_t[idx].item()):
                reason_counter["time_out"] += 1
            else:
                reason_counter["other_terminated"] += 1

            episode_returns[idx] = 0.0
            episode_lengths[idx] = 0
            episode_steps_since_reset[idx] = 0

        if len(finished_returns) >= next_progress or len(finished_returns) >= args_cli.episodes:
            print(
                f"[EVAL] progress={len(finished_returns)}/{args_cli.episodes} "
                f"success_rate={100.0 * np.mean(finished_success):.1f}% "
                f"avg_return={np.mean(finished_returns):.3f} "
                f"env_steps={total_env_steps} wall_time_s={time.time() - eval_started_at:.1f}",
                flush=True,
            )
            while next_progress <= len(finished_returns):
                next_progress += progress_interval
            last_heartbeat_at = time.time()

    wall_time_s = time.time() - eval_started_at

    if not finished_returns:
        raise RuntimeError("No episode finished during evaluation. Increase runtime or reduce --episodes.")

    class_names = _get_class_names(env.unwrapped)
    occ_counts = {name: 0 for name in class_names}
    occ_counts["unknown"] = 0
    for class_index in finished_occ_classes:
        if 0 <= class_index < len(class_names):
            occ_counts[class_names[class_index]] += 1
        else:
            occ_counts["unknown"] += 1

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": args_cli.task,
        "baseline": args_cli.baseline,
        "seed": args_cli.seed,
        "episodes": len(finished_returns),
        "success_class_name": success_class_name,
        "x_limits": list(x_limits),
        "y_limits": list(y_limits),
        "success_rate": float(np.mean(finished_success)),
        "avg_return": float(np.mean(finished_returns)),
        "std_return": float(np.std(finished_returns)),
        "avg_length": float(np.mean(finished_lengths)),
        "std_length": float(np.std(finished_lengths)),
        "avg_final_occ_class_index": float(np.mean(finished_occ_classes)),
        "final_pred_occ_class_counts": occ_counts,
        "reasons": reason_counter,
        "wall_time_s": float(wall_time_s),
    }

    print("\n=== Baseline Evaluation Summary ===")
    print(f"Baseline       : {metrics['baseline']}")
    print(f"Episodes       : {metrics['episodes']}")
    print(f"Success rate   : {metrics['success_rate'] * 100.0:.2f}%")
    print(f"Avg return     : {metrics['avg_return']:.4f} +/- {metrics['std_return']:.4f}")
    print(f"Avg length     : {metrics['avg_length']:.2f} +/- {metrics['std_length']:.2f}")
    print(f"Final occ      : {metrics['final_pred_occ_class_counts']}")
    print(
        "Reasons        : "
        f"success={reason_counter['success']}, "
        f"out_of_bounds={reason_counter['out_of_bounds']}, "
        f"time_out={reason_counter['time_out']}, "
        f"other_terminated={reason_counter['other_terminated']}"
    )
    print(f"Wall time (s)  : {metrics['wall_time_s']:.2f}")

    output_json = args_cli.output_json
    if output_json is None:
        output_json = os.path.join(
            log_root_path,
            f"evaluate_{args_cli.baseline}_{datetime.now():%Y-%m-%d_%H-%M-%S}.json",
        )
    output_json = os.path.abspath(output_json)
    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Metrics saved to: {output_json}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
