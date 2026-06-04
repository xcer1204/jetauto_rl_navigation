from __future__ import annotations

import argparse
import copy
import os
import random
import time
from datetime import datetime
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train the GS-based Jetauto task with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Step interval between video recordings.")
parser.add_argument("--num_envs", type=int, default=None, help="Override the number of environments.")
parser.add_argument("--task", type=str, default="Jetauto-VRRobo-Manager-v0", help="Gym task name.")
parser.add_argument(
    "--agent",
    type=str,
    default="skrl_cfg_entry_point",
    help="Gym registry key used to load the skrl agent config.",
)
parser.add_argument("--seed", type=int, default=None, help="Training seed. Use -1 for a random seed.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path used to resume training.")
parser.add_argument(
    "--max_iterations",
    type=int,
    default=None,
    help="Override training length. For rollout-based agents this is iterations; otherwise it is timesteps.",
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
parser.add_argument("--render_server_host", type=str, default=None, help="Override the GS render server host.")
parser.add_argument("--render_server_port", type=int, default=None, help="Override the GS render server RPC port.")
parser.add_argument("--rgb_socket_host", type=str, default=None, help="Override the GS RGB receiver host.")
parser.add_argument("--rgb_socket_port", type=int, default=None, help="Override the GS RGB receiver TCP port.")
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
from packaging import version

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

import jetauto_navigation  # noqa: F401
# from jetauto_navigation.gs_env_wrapper import GSEnvWrapper
from jetauto_navigation.tasks.manager_based.jetauto_navigation.gs_env_wrapper import GSEnvWrapper



SKRL_MIN_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_MIN_VERSION):
    raise RuntimeError(
        f"Unsupported skrl version: {skrl.__version__}. Install skrl>={SKRL_MIN_VERSION} before running train_gs.py."
    )

if args_cli.ml_framework.startswith("torch"):
    import torch

    from skrl.agents.torch.ppo import PPO_RNN
    from skrl.memories.torch import RandomMemory
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveLR
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils.runner.torch import Runner

    from jetauto_navigation.tasks.manager_based.jetauto_navigation.agents.skrl_lstm_models import (
        LSTMDeterministicValue,
        LSTMGaussianPolicy,
    )
else:
    from skrl.utils.runner.jax import Runner


def _cuda_memory_summary(device: Any) -> str:
    if not args_cli.ml_framework.startswith("torch"):
        return "cuda summary unavailable (non-torch backend)"
    if not torch.cuda.is_available():
        return "cuda unavailable"
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return f"device={torch_device} non_cuda"
    device_index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    except TypeError:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    allocated_bytes = torch.cuda.memory_allocated(torch_device)
    reserved_bytes = torch.cuda.memory_reserved(torch_device)
    max_allocated_bytes = torch.cuda.max_memory_allocated(torch_device)
    max_reserved_bytes = torch.cuda.max_memory_reserved(torch_device)
    mib = 1024.0 * 1024.0
    return (
        f"device={device_index} allocated_mb={allocated_bytes / mib:.1f} "
        f"reserved_mb={reserved_bytes / mib:.1f} free_mb={free_bytes / mib:.1f} "
        f"total_mb={total_bytes / mib:.1f} max_allocated_mb={max_allocated_bytes / mib:.1f} "
        f"max_reserved_mb={max_reserved_bytes / mib:.1f}"
    )


def _resolve_algorithm_name(agent_cfg_key: str) -> str:
    if agent_cfg_key == "skrl_cfg_entry_point":
        return "ppo"
    if agent_cfg_key.endswith("_cfg_entry_point"):
        agent_cfg_key = agent_cfg_key.removesuffix("_cfg_entry_point")
    if agent_cfg_key.startswith("skrl_"):
        agent_cfg_key = agent_cfg_key.removeprefix("skrl_")
    return agent_cfg_key.lower()


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
        "[train_gs] Overriding policy term history "
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
        "[train_gs] Overriding GS network endpoints "
        f"term={selected_name} previous={previous} current={overrides}",
        flush=True,
    )
    return True


def _run_ppo_rnn_training(env, agent_cfg: dict[str, Any], resume_path: str | None, startup_cuda_summary: str | None):
    if not args_cli.ml_framework.startswith("torch"):
        raise RuntimeError("PPO_RNN training in train_gs.py currently supports only the torch backend.")

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

    original_update = agent._update
    startup_suffix = f" startup[{startup_cuda_summary}]" if startup_cuda_summary else ""

    def _debug_update(timestep: int, timesteps: int):
        print(
            "[train_gs] PPO update start "
            f"timestep={timestep} rollout={getattr(agent, '_rollout', -1)} "
            f"current[{_cuda_memory_summary(env.device)}]{startup_suffix}",
            flush=True,
        )
        try:
            return original_update(timestep, timesteps)
        except Exception as exc:
            print(
                "[train_gs] PPO update exception "
                f"timestep={timestep} rollout={getattr(agent, '_rollout', -1)} "
                f"error={exc} current[{_cuda_memory_summary(env.device)}]{startup_suffix}",
                flush=True,
            )
            raise
        finally:
            print(
                "[train_gs] PPO update end "
                f"timestep={timestep} rollout={getattr(agent, '_rollout', -1)} "
                f"current[{_cuda_memory_summary(env.device)}]{startup_suffix}",
                flush=True,
            )

    agent._update = _debug_update

    trainer_cfg = copy.deepcopy(agent_cfg.get("trainer", {}))
    trainer_cfg.pop("class", None)
    trainer = SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg)

    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)
    trainer.train()


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    if not isinstance(agent_cfg, dict):
        raise TypeError(f"Expected a dict agent config from '{args_cli.agent}', but received: {type(agent_cfg)}")

    if _uses_recurrent_policy(args_cli.agent, agent_cfg):
        if not _override_policy_term_history_len(env_cfg, args_cli.policy_term, history_len=1):
            print(
                "[train_gs] Recurrent-policy history override skipped "
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
            "[train_gs] GS network endpoint override skipped "
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

    if args_cli.max_iterations is not None:
        if "rollouts" in agent_cfg.get("agent", {}):
            rollouts = int(agent_cfg["agent"]["rollouts"])
            agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * rollouts
        else:
            agent_cfg["trainer"]["timesteps"] = int(args_cli.max_iterations)
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    algorithm = _resolve_algorithm_name(args_cli.agent)
    experiment_cfg = agent_cfg.setdefault("agent", {}).setdefault("experiment", {})
    experiment_dir = experiment_cfg.get("directory", "jetauto_vrrobo_manager")
    if experiment_cfg.get("checkpoint_interval") == "auto":
        trainer_timesteps = int(agent_cfg.get("trainer", {}).get("timesteps", 0))
        experiment_cfg["checkpoint_interval"] = max(1, trainer_timesteps // 20)
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", experiment_dir))
    run_name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{algorithm}_{args_cli.ml_framework}"
    if experiment_cfg.get("experiment_name"):
        run_name += f"_{experiment_cfg['experiment_name']}"
    experiment_cfg["directory"] = log_root_path
    experiment_cfg["experiment_name"] = run_name

    log_dir = os.path.join(log_root_path, run_name)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    env_cfg.log_dir = log_dir

    print("[train_gs] About to call gym.make(...)", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print("[train_gs] gym.make(...) finished", flush=True)
    startup_cuda_summary = None
    if args_cli.ml_framework.startswith("torch"):
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        startup_cuda_summary = _cuda_memory_summary(getattr(base_env, "device", args_cli.device))
        setattr(base_env, "_startup_cuda_memory_summary", startup_cuda_summary)
        print(f"[train_gs] startup cuda {startup_cuda_summary}", flush=True)


    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    print("[train_gs] About to wrap env with GSEnvWrapper", flush=True)
    env = GSEnvWrapper(
        env,
        policy_term_name=args_cli.policy_term,
        fallback_to_full_policy=args_cli.full_policy_fallback,
    )
    print("[train_gs] GSEnvWrapper finished", flush=True)

    print("[train_gs] About to wrap env with SkrlVecEnvWrapper", flush=True)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    print("[train_gs] SkrlVecEnvWrapper finished", flush=True)


    start_time = time.time()
    agent_class = str(agent_cfg.get("agent", {}).get("class", "")).lower()
    if agent_class == "ppo_rnn":
        _run_ppo_rnn_training(env, agent_cfg, resume_path, startup_cuda_summary)
    else:
        runner = Runner(env, agent_cfg)
        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load(resume_path)
        runner.run()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
