from __future__ import annotations

import argparse
import os
import random
import time
from datetime import datetime

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
parser.add_argument("--max_iterations", type=int, default=None, help="Override the PPO iteration count.")
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
    from skrl.utils.runner.torch import Runner
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


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    if not isinstance(agent_cfg, dict):
        raise TypeError(f"Expected a dict agent config from '{args_cli.agent}', but received: {type(agent_cfg)}")

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed")
    agent_cfg["seed"] = seed
    env_cfg.seed = seed

    if args_cli.max_iterations is not None:
        rollouts = int(agent_cfg["agent"]["rollouts"])
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * rollouts
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    algorithm = _resolve_algorithm_name(args_cli.agent)
    experiment_cfg = agent_cfg.setdefault("agent", {}).setdefault("experiment", {})
    experiment_dir = experiment_cfg.get("directory", "jetauto_vrrobo_manager")
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

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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

    env = GSEnvWrapper(
        env,
        policy_term_name=args_cli.policy_term,
        fallback_to_full_policy=args_cli.full_policy_fallback,
    )
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    start_time = time.time()
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
