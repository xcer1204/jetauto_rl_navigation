# evaluate.py
import os, csv, json, math, argparse, random
from collections import defaultdict
import numpy as np
import torch
import gymnasium as gym

# ===== 你项目里的任务注册名（来自 jetauto_navigation/__init__.py） =====
TASK_NAME = "Jetauto-Navigation-Direct-v0"  # :contentReference[oaicite:3]{index=3}

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------- 这3个钩子按你的环境实现做了对齐 ----------
def infer_success_from_info(info: dict) -> bool:
    # 最佳方式：环境把 success 放进 info/extras；你可按下面补丁开启（见文末小补丁）
    if "success" in info:
        return bool(info["success"])
    # 兜底：没有 success 字段时，Gym 的 terminated 里既包含成功也包含失败，无法100%区分；
    # 我们退化为：若有 "curr_vis" 并 >=0.99 则记为成功。
    if "curr_vis" in info:
        try:
            return float(info["curr_vis"]) >= 0.99
        except Exception:
            pass
    return False

def infer_collision_from_info(info: dict) -> bool:
    # 你的环境内部有 collision_mask，可把它导出到 info（见文末小补丁）
    return bool(info.get("collision", False))

def set_fixed_init(env, pos=(0.0, 0.0, 0.05), yaw_deg=0.0):
    """固定初始位姿：若你的 env 提供自定义 API，可在这里调用。
       这里用 root pose 写入的通用方式（DirectRLEnv 兼容）。"""
    try:
        import isaaclab.utils.math as math_utils
        yaw = math.radians(yaw_deg)
        quat = np.array([0, 0, math.sin(yaw/2), math.cos(yaw/2)], dtype=np.float32)
        # 如果你的 env 暴露了 robot 名称，可以用 scene 接口设置；否则调用 reset 时由 env 自行采样
        # 这里不强行写 root state，避免和内部 reset 流程冲突。保持占位。
    except Exception:
        pass

# ---------- 载入SKRL PPO（按你训练时结构；obs/act尺寸自动从env读） ----------
def load_ppo_agent(env, checkpoint_path, device):
    from skrl.agents.torch.ppo import PPO
    from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
    import torch.nn as nn

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    class Policy(GaussianMixin, Model):
        def __init__(self):
            Model.__init__(self, obs_space=env.observation_space, act_space=env.action_space, device=device)
            GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, initial_log_std=0.0)
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 32), nn.ELU(),
                nn.Linear(32, 32), nn.ELU(),
                nn.Linear(32, act_dim)
            )
        def compute(self, inputs, role):
            x = inputs["states"]
            mu = self.net(x)
            return mu, {}

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
            x = inputs["states"]
            v = self.net(x)
            return v, {}

    agent = PPO(models={"policy": Policy(), "value": Value()}, memory=None, cfg={"rollouts": 1, "learning_starts": 0})
    agent.init()
    agent.load(checkpoint_path)
    agent.set_running_mode("eval")
    return agent

# ---------- 固定起点 rollout 并可保存视频 ----------
def run_fixed_rollouts(args):
    if not args.record_dir:
        return
    try:
        import imageio
    except Exception:
        print("⚠ 想保存视频请先安装: pip install imageio imageio-ffmpeg")
        return

    os.makedirs(args.record_dir, exist_ok=True)
    env = gym.make(TASK_NAME, render_mode="rgb_array" if not args.headless else None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = load_ppo_agent(env, args.checkpoint, device)

    presets = [
        {"pos": (0.0, 0.0, 0.05), "yaw_deg": 0},
        {"pos": (0.8, -0.6, 0.05), "yaw_deg": 90},
        {"pos": (-0.8, 0.6, 0.05), "yaw_deg": -90},
        {"pos": (0.8, 0.8, 0.05), "yaw_deg": 180},
    ]

    for i, p in enumerate(presets):
        set_global_seed(1234 + i)
        # 可按需把 p 应到 env；这里走默认 reset
        obs, info = env.reset()
        frames = []
        ep_ret = 0.0
        for t in range(args.max_steps):
            with torch.no_grad():
                act, _ = agent.act({"states": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)})
            obs, rew, terminated, truncated, info = env.step(act.squeeze(0).cpu().numpy())
            ep_ret += float(rew)
            frame = None
            if hasattr(env, "render"):
                try:
                    frame = env.render()
                except Exception:
                    frame = None
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                break
        if frames:
            path = os.path.join(args.record_dir, f"rollout_{i:02d}_ret{ep_ret:.1f}.mp4")
            imageio.mimsave(path, frames, fps=20)
            print(f"🎬 保存：{path}")
    env.close()

# ---------- 主评测：多 seed × 多 episode ----------
def evaluate(args):
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "seed","episodes","success_rate","avg_return",
            "collision_rate","terminated_rate","truncated_rate"
        ])

    for seed in args.seeds:
        set_global_seed(seed)
        env = gym.make(TASK_NAME, render_mode=None if args.headless else "rgb_array")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = load_ppo_agent(env, args.checkpoint, device)

        succ_cnt = 0
        total_ret = 0.0
        n = args.episodes

        fail_collision = 0
        fail_term = 0
        fail_trunc = 0

        for ep in range(n):
            obs, info = env.reset(seed=seed)
            ep_ret = 0.0
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
                        if collision:
                            fail_collision += 1
                        elif terminated:
                            fail_term += 1
                        elif truncated:
                            fail_trunc += 1
                    total_ret += ep_ret
                    break

        success_rate = succ_cnt / n
        avg_return = total_ret / n
        coll_rate = fail_collision / n
        term_rate = fail_term / n
        trunc_rate = fail_trunc / n

        print(f"[seed {seed}] success={success_rate:.3f}  avg_return={avg_return:.2f}  "
              f"collision={coll_rate:.3f} terminated={term_rate:.3f} truncated={trunc_rate:.3f}")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                seed, n, f"{success_rate:.6f}", f"{avg_return:.6f}",
                f"{coll_rate:.6f}", f"{term_rate:.6f}", f"{trunc_rate:.6f}"
            ])
        env.close()

    print(f"✅ 完成，结果写入：{csv_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="SKRL PPO 权重路径，如 logs/skrl/.../best_agent.pt")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    p.add_argument("--out_dir", type=str, default="eval_outputs")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--record_dir", type=str, default="", help="若填写则保存固定初始位姿的 mp4 到此目录")
    p.add_argument("--max_steps", type=int, default=600)
    args = p.parse_args()

    if args.record_dir:
        run_fixed_rollouts(args)
    evaluate(args)

