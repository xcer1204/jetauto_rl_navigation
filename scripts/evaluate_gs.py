from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from datetime import datetime
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a GS-based Jetauto checkpoint with skrl.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Override the number of environments.")
parser.add_argument("--task", type=str, default="Jetauto-VRRobo-Manager-Play-v0", help="Gym task name.")
parser.add_argument(
    "--agent",
    type=str,
    default="skrl_cfg_entry_point",
    help="Gym registry key used to load the skrl agent config.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. If omitted, use the latest run.")
parser.add_argument("--seed", type=int, default=None, help="Evaluation seed. Use -1 for a random seed.")
parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate.")
parser.add_argument("--max_steps", type=int, default=None, help="Optional max env-steps per episode.")
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
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="ML backend used by skrl.",
)
parser.add_argument(
    "--policy_term",
    type=str,
    default="gs_image",
    help="Policy observation term to prepend to critic observations.",
)
parser.add_argument(
    "--full_policy_fallback",
    action="store_true",
    default=False,
    help="Use the full policy vector if the requested policy term is not found.",
)
parser.add_argument(
    "--output_json",
    type=str,
    default=None,
    help="Optional path to save evaluation metrics as JSON.",
)
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
import skrl
import torch
from packaging import version

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import jetauto_navigation  # noqa: F401
# from jetauto_navigation.gs_env_wrapper import GSEnvWrapper
from jetauto_navigation.tasks.manager_based.jetauto_navigation.gs_env_wrapper import GSEnvWrapper

SKRL_MIN_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_MIN_VERSION):
    raise RuntimeError(
        f"Unsupported skrl version: {skrl.__version__}. Install skrl>={SKRL_MIN_VERSION} before running evaluate_gs.py."
    )

if args_cli.ml_framework.startswith("torch"):
    from skrl.agents.torch.ppo import PPO_RNN
    from skrl.memories.torch import RandomMemory
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveLR
    from skrl.utils.runner.torch import Runner

    from jetauto_navigation.tasks.manager_based.jetauto_navigation.agents.skrl_lstm_models import (
        LSTMDeterministicValue,
        LSTMGaussianPolicy,
    )
else:
    from skrl.utils.runner.jax import Runner


def _resolve_algorithm_name(agent_cfg_key: str) -> str:
    if agent_cfg_key == "skrl_cfg_entry_point":
        return "ppo"
    if agent_cfg_key.endswith("_cfg_entry_point"):
        agent_cfg_key = agent_cfg_key.removesuffix("_cfg_entry_point")
    if agent_cfg_key.startswith("skrl_"):
        agent_cfg_key = agent_cfg_key.removeprefix("skrl_")
    return agent_cfg_key.lower()


def _resolve_checkpoint(log_root_path: str, algorithm: str) -> str:
    if args_cli.checkpoint:
        return os.path.abspath(args_cli.checkpoint)
    return get_checkpoint_path(
        log_root_path,
        run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
        other_dirs=["checkpoints"],
    )


def _checkpoint_looks_recurrent(checkpoint_path: str) -> bool:
    normalized = checkpoint_path.replace("\\", "/").lower()
    return "_lstm_" in normalized or "_rnn_" in normalized


def _reward_shaper_function(scale: float):
    def _reward_shaper(rewards, *_, **__):
        return rewards * scale

    return _reward_shaper


def _resolve_torch_component(name: Any) -> Any:
    if not isinstance(name, str):
        return name
    mapping = {
        "KLAdaptiveLR": KLAdaptiveLR,
        "RunningStandardScaler": RunningStandardScaler,
    }
    return mapping.get(name, name)


def _extract_layers(model_cfg: dict[str, Any], key: str, default: list[int] | None = None) -> list[int]:
    if default is None:
        default = []
    layers = model_cfg.get(key)
    if isinstance(layers, (list, tuple)):
        return [int(v) for v in layers]
    network_cfg = model_cfg.get("network")
    if isinstance(network_cfg, list) and network_cfg and isinstance(network_cfg[0], dict):
        net_layers = network_cfg[0].get("layers")
        if isinstance(net_layers, (list, tuple)):
            return [int(v) for v in net_layers]
    return [int(v) for v in default]


def _build_ppo_rnn_models(env, models_cfg: dict[str, Any]) -> dict[str, Any]:
    policy_cfg = copy.deepcopy(models_cfg.get("policy", {}))
    value_cfg = copy.deepcopy(models_cfg.get("value", {}))

    policy_class = str(policy_cfg.pop("class", "LSTMGaussianPolicy"))
    value_class = str(value_cfg.pop("class", "LSTMDeterministicValue"))
    if policy_class != "LSTMGaussianPolicy":
        raise ValueError(f"Unsupported LSTM policy class: {policy_class}")
    if value_class != "LSTMDeterministicValue":
        raise ValueError(f"Unsupported LSTM value class: {value_class}")

    policy = LSTMGaussianPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        num_envs=env.num_envs,
        clip_actions=bool(policy_cfg.get("clip_actions", False)),
        clip_log_std=bool(policy_cfg.get("clip_log_std", True)),
        min_log_std=float(policy_cfg.get("min_log_std", -20.0)),
        max_log_std=float(policy_cfg.get("max_log_std", 2.0)),
        initial_log_std=float(policy_cfg.get("initial_log_std", 0.0)),
        encoder_layers=_extract_layers(policy_cfg, "encoder_layers"),
        rnn_hidden_size=int(policy_cfg.get("rnn_hidden_size", 256)),
        rnn_num_layers=int(policy_cfg.get("rnn_num_layers", 1)),
        sequence_length=int(policy_cfg.get("sequence_length", 1)),
        head_layers=_extract_layers(policy_cfg, "head_layers", default=[256, 128]),
        activation=str(policy_cfg.get("activation", policy_cfg.get("activations", "elu"))),
    )
    value = LSTMDeterministicValue(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        num_envs=env.num_envs,
        clip_actions=bool(value_cfg.get("clip_actions", False)),
        encoder_layers=_extract_layers(value_cfg, "encoder_layers"),
        rnn_hidden_size=int(value_cfg.get("rnn_hidden_size", 256)),
        rnn_num_layers=int(value_cfg.get("rnn_num_layers", 1)),
        sequence_length=int(value_cfg.get("sequence_length", 1)),
        head_layers=_extract_layers(value_cfg, "head_layers", default=[256, 128]),
        activation=str(value_cfg.get("activation", value_cfg.get("activations", "elu"))),
    )
    return {"policy": policy, "value": value}


def _build_ppo_rnn_agent_cfg(raw_cfg: dict[str, Any], observation_space, device) -> dict[str, Any]:
    cfg = copy.deepcopy(raw_cfg)
    cfg.pop("class", None)

    cfg["learning_rate_scheduler"] = _resolve_torch_component(cfg.get("learning_rate_scheduler"))
    cfg["state_preprocessor"] = _resolve_torch_component(cfg.get("state_preprocessor"))
    cfg["value_preprocessor"] = _resolve_torch_component(cfg.get("value_preprocessor"))

    for key in ("learning_rate_scheduler_kwargs", "state_preprocessor_kwargs", "value_preprocessor_kwargs"):
        cfg[key] = {} if cfg.get(key) is None else dict(cfg[key])

    if "rewards_shaper_scale" in cfg:
        cfg["rewards_shaper"] = _reward_shaper_function(float(cfg["rewards_shaper_scale"]))

    if cfg.get("state_preprocessor") is not None:
        cfg["state_preprocessor_kwargs"].update({"size": observation_space, "device": device})
    if cfg.get("value_preprocessor") is not None:
        cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})

    return cfg


def _uses_recurrent_policy(agent_cfg_key: str, agent_cfg: dict[str, Any]) -> bool:
    agent_class = str(agent_cfg.get("agent", {}).get("class", "")).lower()
    if agent_class == "ppo_rnn":
        return True
    lowered = str(agent_cfg_key).lower()
    return "lstm" in lowered or "rnn" in lowered


def _override_policy_term_history_len(env_cfg, policy_term_name: str, history_len: int) -> bool:
    observations_cfg = getattr(env_cfg, "observations", None)
    policy_cfg = getattr(observations_cfg, "policy", None) if observations_cfg is not None else None
    term_cfg = getattr(policy_cfg, policy_term_name, None) if policy_cfg is not None else None
    if term_cfg is None:
        return False

    params = getattr(term_cfg, "params", None)
    if params is None:
        params = {}
        setattr(term_cfg, "params", params)
    elif not isinstance(params, dict):
        params = dict(params)
        setattr(term_cfg, "params", params)

    previous = params.get("history_len", 4)
    params["history_len"] = int(max(1, history_len))
    print(
        "[evaluate_gs] Overriding policy term history "
        f"term={policy_term_name} previous={previous} current={params['history_len']}",
        flush=True,
    )
    return True


def _get_policy_terms(base_env) -> list[str]:
    obs_manager = getattr(base_env, "observation_manager", None)
    if obs_manager is None:
        return []
    active_terms = getattr(obs_manager, "active_terms", {})
    return [str(term) for term in active_terms.get("policy", [])]


def _resolve_policy_term(base_env, requested_term: str, fallback_to_full_policy: bool) -> str:
    policy_terms = _get_policy_terms(base_env)
    if not policy_terms:
        return requested_term

    if requested_term in policy_terms:
        return requested_term

    if requested_term == "gs_image" and len(policy_terms) == 1:
        resolved_term = policy_terms[0]
        print(
            f"[INFO] Policy term '{requested_term}' is not defined by the environment. "
            f"Using the only available policy term: '{resolved_term}'."
        )
        return resolved_term

    if fallback_to_full_policy:
        print(
            f"[INFO] Policy term '{requested_term}' is not defined by the environment. "
            "Falling back to the full policy observation vector."
        )
        return requested_term

    available_terms = ", ".join(policy_terms)
    raise ValueError(
        f"Policy term '{requested_term}' is not defined by the environment. "
        f"Available policy terms: {available_terms}"
    )


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
    class_names = base_env.extras.get("pred_occ_class_names", ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%"))
    class_names = tuple(str(name) for name in class_names)
    try:
        return class_names.index(str(success_class_name))
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


def _build_eval_agent(env, agent_cfg: dict[str, Any]):
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    trainer_cfg = copy.deepcopy(agent_cfg.get("trainer", {}))
    trainer_cfg.pop("class", None)

    agent_class = str(agent_cfg.get("agent", {}).get("class", "")).lower()
    if agent_class != "ppo_rnn":
        runner = Runner(env, agent_cfg)
        return runner.agent

    if not args_cli.ml_framework.startswith("torch"):
        raise RuntimeError("PPO_RNN evaluation in evaluate_gs.py currently supports only the torch backend.")

    models = _build_ppo_rnn_models(env, agent_cfg.get("models", {}))
    raw_agent_cfg = agent_cfg.get("agent", {})
    rollouts = int(raw_agent_cfg.get("rollouts", 16))

    memory_cfg = copy.deepcopy(agent_cfg.get("memory", {}))
    memory_class = str(memory_cfg.pop("class", "RandomMemory")).lower()
    if memory_class != "randommemory":
        raise ValueError(f"Unsupported memory class for PPO_RNN: {memory_class}")
    memory_size = int(memory_cfg.pop("memory_size", -1))
    if memory_size < 0:
        memory_size = rollouts
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device, **memory_cfg)

    ppo_rnn_cfg = _build_ppo_rnn_agent_cfg(raw_agent_cfg, env.observation_space, env.device)
    agent = PPO_RNN(
        models=models,
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        cfg=ppo_rnn_cfg,
    )
    agent.init(trainer_cfg=trainer_cfg)
    return agent


def _advance_eval_rnn_state(agent, terminated, truncated) -> None:
    if not getattr(agent, "_rnn", False):
        return

    final_states = getattr(agent, "_rnn_final_states", None)
    initial_states = getattr(agent, "_rnn_initial_states", None)
    if not isinstance(final_states, dict) or not isinstance(initial_states, dict):
        return

    done = torch.as_tensor(terminated) | torch.as_tensor(truncated)
    done = done.reshape(-1).to(device=agent.device, dtype=torch.bool)

    policy_states = [state.clone() for state in final_states.get("policy", [])]
    value_states = [state.clone() for state in final_states.get("value", [])]

    finished = done.nonzero(as_tuple=False).reshape(-1)
    if finished.numel():
        for state in policy_states:
            state[:, finished] = 0
        if final_states.get("value") is final_states.get("policy"):
            value_states = policy_states
        else:
            for state in value_states:
                state[:, finished] = 0

    initial_states["policy"] = policy_states
    initial_states["value"] = value_states


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=False if args_cli.disable_fabric else None,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    if not isinstance(agent_cfg, dict):
        raise TypeError(f"Expected a dict agent config from '{args_cli.agent}', but received: {type(agent_cfg)}")

    if _uses_recurrent_policy(args_cli.agent, agent_cfg):
        if not _override_policy_term_history_len(env_cfg, args_cli.policy_term, history_len=1):
            print(
                "[evaluate_gs] Recurrent-policy history override skipped "
                f"because term '{args_cli.policy_term}' was not found in env_cfg.observations.policy.",
                flush=True,
            )

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed")
    agent_cfg["seed"] = seed
    env_cfg.seed = seed

    algorithm = _resolve_algorithm_name(args_cli.agent)
    experiment_cfg = agent_cfg.setdefault("agent", {}).setdefault("experiment", {})
    experiment_dir = experiment_cfg.get("directory", "jetauto_vrrobo_manager")
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", experiment_dir))
    checkpoint_path = _resolve_checkpoint(log_root_path, algorithm)
    log_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    if _checkpoint_looks_recurrent(checkpoint_path) and not _uses_recurrent_policy(args_cli.agent, agent_cfg):
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' appears to come from an LSTM/RNN run, "
            f"but --agent={args_cli.agent!r} resolves to a non-recurrent config. "
            "Please rerun with --agent skrl_lstm_cfg_entry_point."
        )
    env_cfg.log_dir = log_dir

    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    policy_term_name = _resolve_policy_term(env.unwrapped, args_cli.policy_term, args_cli.full_policy_fallback)
    success_class_name, x_limits, y_limits = _resolve_eval_settings(env_cfg)
    print(f"[INFO] Evaluation success class: {success_class_name}")
    print(f"[INFO] Evaluation bounds: x={x_limits}, y={y_limits}")

    env = GSEnvWrapper(
        env,
        policy_term_name=policy_term_name,
        fallback_to_full_policy=args_cli.full_policy_fallback,
    )
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    agent = _build_eval_agent(env, agent_cfg)
    agent.load(checkpoint_path)
    agent.set_running_mode("eval")

    num_envs = int(getattr(env.unwrapped, "num_envs", getattr(env, "num_envs", 1)))
    device = getattr(env.unwrapped, "device", "cpu")

    episode_returns = torch.zeros(num_envs, device=device, dtype=torch.float32)
    episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
    episode_steps_since_reset = torch.zeros(num_envs, device=device, dtype=torch.long)

    finished_returns: list[float] = []
    finished_lengths: list[int] = []
    finished_success: list[int] = []
    reason_counter = {"success": 0, "out_of_bounds": 0, "time_out": 0, "other_terminated": 0}

    obs, _ = env.reset()
    eval_started_at = time.time()

    while simulation_app.is_running() and len(finished_returns) < args_cli.episodes:
        with torch.inference_mode():
            outputs = agent.act(obs, timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {agent: outputs[-1][agent].get("mean_actions", outputs[0][agent]) for agent in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, reward, terminated, truncated, _ = env.step(actions)
            _advance_eval_rnn_state(agent, terminated, truncated)

        reward_t = _to_1d_tensor(reward, device=device, dtype=torch.float32)
        term_t = _to_1d_tensor(terminated, device=device, dtype=torch.bool)
        trunc_t = _to_1d_tensor(truncated, device=device, dtype=torch.bool)

        episode_returns += reward_t
        episode_lengths += 1
        episode_steps_since_reset += 1

        done_t = term_t | trunc_t

        # Optional evaluation-only cap per episode.
        if args_cli.max_steps is not None and args_cli.max_steps > 0:
            max_step_done = episode_steps_since_reset >= int(args_cli.max_steps)
            trunc_t = trunc_t | max_step_done
            done_t = term_t | trunc_t

        if not done_t.any():
            continue

        base_env = env.unwrapped
        success_mask = _get_termination_term_mask(base_env, "visibility_success", num_envs=num_envs, device=device)
        out_of_bounds = _get_termination_term_mask(base_env, "out_of_bounds", num_envs=num_envs, device=device)

        if success_mask is None or out_of_bounds is None:
            pred_occ = _get_predicted_occlusion_classes(base_env, num_envs=num_envs, device=device)
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

        if len(finished_returns) and len(finished_returns) % max(1, args_cli.episodes // 10) == 0:
            print(
                f"[EVAL] progress={len(finished_returns)}/{args_cli.episodes} "
                f"success_rate={100.0 * np.mean(finished_success):.1f}% "
                f"avg_return={np.mean(finished_returns):.3f}"
            )

    wall_time_s = time.time() - eval_started_at

    if not finished_returns:
        raise RuntimeError("No episode finished during evaluation. Increase runtime or reduce --episodes.")

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": args_cli.task,
        "checkpoint": checkpoint_path,
        "episodes": len(finished_returns),
        "success_class_name": success_class_name,
        "x_limits": list(x_limits),
        "y_limits": list(y_limits),
        "success_rate": float(np.mean(finished_success)),
        "avg_return": float(np.mean(finished_returns)),
        "std_return": float(np.std(finished_returns)),
        "avg_length": float(np.mean(finished_lengths)),
        "std_length": float(np.std(finished_lengths)),
        "reasons": reason_counter,
        "wall_time_s": float(wall_time_s),
    }

    print("\n=== Evaluation Summary ===")
    print(f"Episodes       : {metrics['episodes']}")
    print(f"Success rate   : {metrics['success_rate'] * 100.0:.2f}%")
    print(f"Avg return     : {metrics['avg_return']:.4f} +/- {metrics['std_return']:.4f}")
    print(f"Avg length     : {metrics['avg_length']:.2f} +/- {metrics['std_length']:.2f}")
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
        output_json = os.path.join(log_dir, "eval", f"evaluate_{datetime.now():%Y-%m-%d_%H-%M-%S}.json")
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
