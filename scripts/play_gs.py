from __future__ import annotations

import argparse
import copy
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher
import numpy as np
from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


parser = argparse.ArgumentParser(description="Play a GS-based Jetauto checkpoint with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record a rollout video.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in steps.")
parser.add_argument(
    "--save_paired_views",
    action="store_true",
    default=False,
    help="Save top-down renders and robot-view RGB images with the same step index.",
)
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
parser.add_argument("--render_server_host", type=str, default=None, help="Override the GS render server host.")
parser.add_argument("--render_server_port", type=int, default=None, help="Override the GS render server RPC port.")
parser.add_argument("--rgb_socket_host", type=str, default=None, help="Override the GS RGB receiver host.")
parser.add_argument("--rgb_socket_port", type=int, default=None, help="Override the GS RGB receiver TCP port.")
parser.add_argument("--real_time", action="store_true", default=False, help="Throttle playback to real-time.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# if args_cli.video:
#     args_cli.enable_cameras = True

def _needs_local_cameras(task_name: str, policy_term: str, record_video: bool) -> bool:
    if record_video:
        return True
    if "IsaacRGB" in task_name:
        return True
    if policy_term == "rgb_feature":
        return True
    return False


args_cli.enable_cameras = _needs_local_cameras(args_cli.task, args_cli.policy_term, args_cli.video)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import skrl
import torch
from packaging import version

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import jetauto_navigation  # noqa: F401
from jetauto_navigation.tasks.manager_based.jetauto_navigation.gs_env_wrapper import GSEnvWrapper


SKRL_MIN_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_MIN_VERSION):
    raise RuntimeError(
        f"Unsupported skrl version: {skrl.__version__}. Install skrl>={SKRL_MIN_VERSION} before running play_gs.py."
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
        "[play_gs] Overriding policy term history "
        f"term={policy_term_name} previous={previous} current={params['history_len']}",
        flush=True,
    )
    return True


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
        "[play_gs] Overriding GS network endpoints "
        f"term={selected_name} previous={previous} current={overrides}",
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
        raise RuntimeError("PPO_RNN playback in play_gs.py currently supports only the torch backend.")

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


def _create_progress_bar(total: int | None, desc: str):
    if tqdm is None:
        print("[WARN] tqdm is not installed; progress bar is disabled.", flush=True)
        return None
    return tqdm(total=total, desc=desc, unit="step", dynamic_ncols=True)


def _make_playback_run_name(checkpoint_path: str) -> str:
    checkpoint_stem = Path(checkpoint_path).stem.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{checkpoint_stem}"


def _configure_debug_image_save_dir(env_cfg, policy_term_name: str, playback_run_name: str) -> str | None:
    observations_cfg = getattr(env_cfg, "observations", None)
    policy_cfg = getattr(observations_cfg, "policy", None) if observations_cfg is not None else None
    term_cfg = getattr(policy_cfg, policy_term_name, None) if policy_cfg is not None else None
    if term_cfg is None:
        return None

    params = getattr(term_cfg, "params", None)
    if params is None or not isinstance(params, dict):
        return None
    if not bool(params.get("save_debug_images", False)):
        return None

    base_dir = Path(str(params.get("save_dir", "logs/gs_render_debug_play")))
    save_dir = base_dir / playback_run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    params["save_dir"] = str(save_dir)
    return str(save_dir)


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


def _should_probe_render_frame(step: int) -> bool:
    if step <= 3:
        return True
    if step <= 20:
        return step % 5 == 0
    return step % 50 == 0


def _probe_render_frame(base_env, step: str | int, state: dict[str, Any]) -> None:
    try:
        frame = base_env.render()
    except Exception as exc:  # pragma: no cover - best effort runtime diagnostics
        if not state.get("warned_exception"):
            print(f"[WARN] Render probe failed at step={step}: {exc}", flush=True)
            state["warned_exception"] = True
        return

    if frame is None:
        if not state.get("warned_none"):
            print(
                "[WARN] env.render() returned None. Offscreen rendering may not be producing frames on this server.",
                flush=True,
            )
            state["warned_none"] = True
        return

    if not isinstance(frame, np.ndarray):
        if not state.get("warned_type"):
            print(
                f"[WARN] env.render() returned an unexpected type: {type(frame)}",
                flush=True,
            )
            state["warned_type"] = True
        return

    if frame.size == 0:
        if not state.get("warned_empty"):
            print(
                "[WARN] env.render() returned an empty frame. This usually means offscreen rendering is not working.",
                flush=True,
            )
            state["warned_empty"] = True
        return

    frame_min = int(frame.min())
    frame_max = int(frame.max())
    frame_mean = float(frame.mean())

    if not state.get("printed_first_frame"):
        print(
            "[DEBUG] First render frame "
            f"step={step} shape={tuple(frame.shape)} dtype={frame.dtype} "
            f"min={frame_min} max={frame_max} mean={frame_mean:.2f}",
            flush=True,
        )
        state["printed_first_frame"] = True

    if frame_max == 0:
        state["consecutive_black"] = int(state.get("consecutive_black", 0)) + 1
        if state["consecutive_black"] >= 3 and not state.get("warned_black"):
            print(
                "[WARN] Observed multiple consecutive black render frames. "
                "This usually means the server is not producing valid offscreen RGB frames.",
                flush=True,
            )
            state["warned_black"] = True
        return

    if state.get("consecutive_black", 0) > 0 and not state.get("reported_recovery"):
        print("[INFO] Render probe recovered from black frames and is now returning non-zero pixels.", flush=True)
        state["reported_recovery"] = True
    state["consecutive_black"] = 0


def _resolve_observation_term_instance(base_env, group_name: str, term_name: str):
    obs_manager = getattr(base_env, "observation_manager", None)
    if obs_manager is None:
        return None
    group_names = getattr(obs_manager, "_group_obs_term_names", {})
    group_cfgs = getattr(obs_manager, "_group_obs_term_cfgs", {})
    if group_name not in group_names or group_name not in group_cfgs:
        return None
    for name, term_cfg in zip(group_names[group_name], group_cfgs[group_name]):
        if name == term_name:
            return getattr(term_cfg, "func", None)
    return None


def _configure_paired_view_dirs(log_dir: str, playback_run_name: str) -> dict[str, Path]:
    base_dir = Path(log_dir) / "paired_views" / playback_run_name
    topdown_dir = base_dir / "topdown"
    egoview_dir = base_dir / "egoview"
    topdown_dir.mkdir(parents=True, exist_ok=True)
    egoview_dir.mkdir(parents=True, exist_ok=True)
    return {"base": base_dir, "topdown": topdown_dir, "egoview": egoview_dir}


def _save_paired_views(
    *,
    base_env,
    obs_term_instance,
    paired_dirs: dict[str, Path],
    step_index: int,
    env_index: int = 0,
) -> bool:
    frame = base_env.render()
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return False
    images_np = getattr(obs_term_instance, "latest_images_np", None)
    if not isinstance(images_np, np.ndarray) or images_np.ndim != 2 or images_np.shape[0] == 0:
        return False
    env_index = min(max(int(env_index), 0), images_np.shape[0] - 1)
    robot_img = images_np[env_index].reshape(3, 180, 320).transpose(1, 2, 0)
    topdown_path = paired_dirs["topdown"] / f"step{step_index:06d}.png"
    egoview_path = paired_dirs["egoview"] / f"step{step_index:06d}.png"
    Image.fromarray(frame).save(topdown_path)
    Image.fromarray(robot_img).save(egoview_path)
    return True


def main():
    env = None
    progress_bar = None
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
                "[play_gs] Recurrent-policy history override skipped "
                f"because term '{args_cli.policy_term}' was not found in env_cfg.observations.policy.",
                flush=True,
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
            "[play_gs] GS network endpoint override skipped "
            f"because neither '{args_cli.policy_term}' nor 'gs_image' was found in env_cfg.observations.policy.",
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
    playback_run_name = _make_playback_run_name(checkpoint_path)
    if _checkpoint_looks_recurrent(checkpoint_path) and not _uses_recurrent_policy(args_cli.agent, agent_cfg):
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' appears to come from an LSTM/RNN run, "
            f"but --agent={args_cli.agent!r} resolves to a non-recurrent config. "
            "Please rerun with --agent skrl_lstm_cfg_entry_point."
        )
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")

    env_cfg.log_dir = log_dir
    debug_image_dir = _configure_debug_image_save_dir(env_cfg, args_cli.policy_term, playback_run_name)
    if debug_image_dir is not None:
        print(f"[INFO] Playback GS images will be saved under: {debug_image_dir}")
    paired_view_dirs = _configure_paired_view_dirs(log_dir, playback_run_name) if args_cli.save_paired_views else None
    if paired_view_dirs is not None:
        print(f"[INFO] Paired top-down/ego-view frames will be saved under: {paired_view_dirs['base']}")

    try:
        render_mode = "rgb_array" if (args_cli.video or args_cli.save_paired_views) else None
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        base_render_env = env.unwrapped
        policy_term_name = _resolve_policy_term(base_render_env, args_cli.policy_term, args_cli.full_policy_fallback)
        obs_term_instance = (
            _resolve_observation_term_instance(base_render_env, "policy", policy_term_name) if args_cli.save_paired_views else None
        )
        if args_cli.save_paired_views and obs_term_instance is None:
            raise RuntimeError(
                f"Unable to resolve observation term instance for group='policy', term='{policy_term_name}'."
            )

        try:
            step_dt = env.step_dt
        except AttributeError:
            step_dt = env.unwrapped.step_dt

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play", playback_run_name),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording video during playback.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = GSEnvWrapper(
            env,
            policy_term_name=policy_term_name,
            fallback_to_full_policy=args_cli.full_policy_fallback,
        )
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        agent = _build_eval_agent(env, agent_cfg)
        agent.load(checkpoint_path)
        agent.set_running_mode("eval")

        progress_total = args_cli.video_length if (args_cli.video or args_cli.save_paired_views) else None
        progress_desc = "Recording" if args_cli.video else ("Saving Pairs" if args_cli.save_paired_views else "Playback")
        progress_bar = _create_progress_bar(progress_total, progress_desc)

        obs, _ = env.reset()
        render_probe_state: dict[str, Any] = {}
        if args_cli.video:
            _probe_render_frame(base_render_env, "reset", render_probe_state)
        if paired_view_dirs is not None:
            saved_reset = _save_paired_views(
                base_env=base_render_env,
                obs_term_instance=obs_term_instance,
                paired_dirs=paired_view_dirs,
                step_index=0,
            )
            if saved_reset:
                print("[INFO] Saved paired views for reset step=0.", flush=True)
            if not saved_reset:
                print("[WARN] Failed to save paired views for reset step.", flush=True)

        rollout_start_time = time.time()
        timestep = 0
        max_steps = args_cli.video_length if (args_cli.video or args_cli.save_paired_views) else None
        while True:
            if max_steps is not None and timestep >= max_steps:
                break
            if max_steps is None and not simulation_app.is_running():
                break
            start_time = time.time()
            with torch.inference_mode():
                outputs = agent.act(obs, timestep=0, timesteps=0)
                if hasattr(env, "possible_agents"):
                    actions = {
                        agent: outputs[-1][agent].get("mean_actions", outputs[0][agent]) for agent in env.possible_agents
                    }
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                obs, _, terminated, truncated, _ = env.step(actions)
                _advance_eval_rnn_state(agent, terminated, truncated)

            timestep += 1
            if paired_view_dirs is not None:
                saved_pair = _save_paired_views(
                    base_env=base_render_env,
                    obs_term_instance=obs_term_instance,
                    paired_dirs=paired_view_dirs,
                    step_index=timestep,
                )
                if saved_pair and timestep <= 3:
                    print(f"[INFO] Saved paired views for step={timestep}.", flush=True)
                if not saved_pair and timestep <= 3:
                    print(f"[WARN] Failed to save paired views at step={timestep}.", flush=True)

            if args_cli.video and _should_probe_render_frame(timestep):
                _probe_render_frame(base_render_env, timestep, render_probe_state)

            if progress_bar is not None:
                progress_bar.update(1)
                if timestep == 1 or timestep % 10 == 0:
                    elapsed = max(time.time() - rollout_start_time, 1e-6)
                    progress_bar.set_postfix_str(f"{timestep / elapsed:.1f} step/s")

            sleep_time = step_dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
