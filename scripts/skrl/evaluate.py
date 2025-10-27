# evaluate.py —— B 方案 + AppLauncher（pip 安装友好）
# 1) 先拉起 Kit/Isaac Sim 运行时
import sys, argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
# 你的评测参数
parser.add_argument("--task", type=str, default="Jetauto-Navigation-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=1000)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--out_dir", type=str, default="eval_outputs")
# parser.add_argument("--headless", action="store_true")
parser.add_argument("--record_dir", type=str, default="")
parser.add_argument("--max_steps", type=int, default=600)
# 合并 AppLauncher 的 CLI（如 --renderer, --gpu, --enable_cameras 等）
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_rest = parser.parse_known_args()
# 交给后续库的解析，避免不认识的参数报错
sys.argv = [sys.argv[0]] + hydra_rest

# 启动 Kit（这一步之后 omni.* 可用）
_app = AppLauncher(args_cli)
simulation_app = _app.app

# 2) 下面是原来的 B 方案逻辑：从注册表取 cfg entry point → 实例化 → gym.make
import os, csv, json, math, importlib, random
from collections import defaultdict
import numpy as np
import torch
import gymnasium as gym

# 触发你的任务注册（很关键）
importlib.import_module("jetauto_navigation.tasks.direct.jetauto_navigation")

def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_env_cfg_class(task_id: str):
    try:
        spec = gym.envs.registry[task_id]
    except KeyError as e:
        avail = [k for k in gym.envs.registry.keys() if "Jetauto" in k or "Direct" in k]
        raise RuntimeError(f"Task id '{task_id}' not found. Available: {avail}") from e
    cfg_ep = spec.kwargs.get("env_cfg_entry_point")
    if not cfg_ep or ":" not in cfg_ep:
        raise RuntimeError(f"Task '{task_id}' missing kwargs['env_cfg_entry_point']")
    mod_path, cls_name = cfg_ep.split(":")
    return getattr(importlib.import_module(mod_path), cls_name)

def make_env(task_id: str, headless: bool, overrides: dict | None = None):
    EnvCfg = get_env_cfg_class(task_id)
    env_cfg = EnvCfg()
    if overrides:
        for k, v in overrides.items():
            obj = env_cfg
            parts = k.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], v)
    # 传 cfg 构造环境；是否渲染由 headless 控制
    return gym.make(task_id, cfg=env_cfg, render_mode=None if headless else "rgb_array")

def infer_success_from_info(info: dict) -> bool:
    if "success" in info: return bool(info["success"])
    if "curr_vis" in info:
        try: return float(info["curr_vis"]) >= 0.99
        except: pass
    return False

def infer_collision_from_info(info: dict) -> bool:
    return bool(info.get("collision", False))

def set_fixed_init(env, pos=(0.0, 0.0, 0.05), yaw_deg=0.0):
    return  # 如需固定初始位姿，这里接你的自定义 API

def load_ppo_agent(env, checkpoint_path, device):
    from skrl.agents.torch.ppo import PPO
    from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
    import torch.nn as nn
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    class Policy(GaussianMixin, Model):
        def __init__(self):
            Model.__init__(self, obs_space=env.observation_space, act_space=env.action_space, device=device)
            GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True,
                                   min_log_std=-20, max_log_std=2, initial_log_std=0.0)
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 32), nn.ELU(),
                nn.Linear(32, 32), nn.ELU(),
                nn.Linear(32, act_dim)
            )
        def compute(self, inputs, role):
            x = inputs["states"]; mu = self.net(x); return mu, {}

    class Value(DeterministicMixin, Model):
        def __init__(self):
            Model.__init__(self, obs_space=env.observation_space, act_space=env.action_space, device=device)
            DeterministicMixin.__init__(self, clip_actions=False)
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 32), nn.ELU(),
                nn.Linear(32, 32), nn.ELU(),
                nn.Linear(32, 1)
            )
        def compute(self, inputs, role):
            x = inputs["states"]; v = self.net(x); return v, {}

    agent = PPO(models={"policy": Policy(), "value": Value()}, memory=None,
                cfg={"rollouts": 1, "learning_starts": 0})
    agent.init(); agent.load(checkpoint_path); agent.set_running_mode("eval")
    return agent

def run_fixed_rollouts(args):
    if not args.record_dir: return
    try:
        import imageio
    except Exception:
        print("⚠ 保存视频需要: pip install imageio imageio-ffmpeg"); return
    os.makedirs(args.record_dir, exist_ok=True)
    env = make_env(args.task, headless=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = load_ppo_agent(env, args.checkpoint, device)
    presets = [
        {"pos": (0.0, 0.0, 0.05), "yaw_deg": 0},
        {"pos": (0.8, -0.6, 0.05), "yaw_deg": 90},
        {"pos": (-0.8, 0.6, 0.05), "yaw_deg": -90},
        {"pos": (0.8, 0.8, 0.05), "yaw_deg": 180},
    ]
    for i, p in enumerate(presets):
        set_global_seed(1234 + i); set_fixed_init(env, **p)
        obs, info = env.reset(); frames = []; ep_ret = 0.0
        for _ in range(args.max_steps):
            with torch.no_grad():
                act, _ = agent.act({"states": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)})
            obs, rew, terminated, truncated, info = env.step(act.squeeze(0).cpu().numpy())
            ep_ret += float(rew)
            frame = None
            if hasattr(env, "render"):
                try: frame = env.render()
                except Exception: frame = None
            if frame is not None: frames.append(frame)
            if terminated or truncated: break
        if frames:
            import imageio
            path = os.path.join(args.record_dir, f"rollout_{i:02d}_ret{ep_ret:.1f}.mp4")
            imageio.mimsave(path, frames, fps=20); print(f"�� 保存：{path}")
    env.close()

def evaluate(args):
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["seed","episodes","success_rate","avg_return",
                                "collision_rate","terminated_rate","truncated_rate"])
    for seed in args.seeds:
        set_global_seed(seed)
        env = make_env(args.task, headless=args.headless)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = load_ppo_agent(env, args.checkpoint, device)

        succ_cnt = 0; total_ret = 0.0; n = args.episodes
        fail_collision = fail_term = fail_trunc = 0
        for _ in range(n):
            obs, info = env.reset(seed=seed); ep_ret = 0.0
            while True:
                with torch.no_grad():
                    act, _ = agent.act({"states": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)})
                obs, rew, terminated, truncated, info = env.step(act.squeeze(0).cpu().numpy())
                ep_ret += float(rew)
                if terminated or truncated:
                    success = infer_success_from_info(info)
                    collision = infer_collision_from_info(info)
                    succ_cnt += int(success)
                    if not success:
                        if collision: fail_collision += 1
                        elif terminated: fail_term += 1
                        elif truncated: fail_trunc += 1
                    total_ret += ep_ret; break
        success_rate = succ_cnt / n; avg_return = total_ret / n
        coll_rate = fail_collision / n; term_rate = fail_term / n; trunc_rate = fail_trunc / n
        print(f"[seed {seed}] success={success_rate:.3f}  avg_return={avg_return:.2f}  "
              f"collision={coll_rate:.3f} terminated={term_rate:.3f} truncated={trunc_rate:.3f}")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([seed, n, f"{success_rate:.6f}", f"{avg_return:.6f}",
                                    f"{coll_rate:.6f}", f"{term_rate:.6f}", f"{trunc_rate:.6f}"])
        env.close()
    print(f"✅ 完成，结果写入：{csv_path}")

if __name__ == "__main__":
    args = args_cli  # 直接复用上面解析的参数
    if args.record_dir: run_fixed_rollouts(args)
    evaluate(args)
    simulation_app.close()
