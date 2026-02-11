from __future__ import annotations

import atexit
import pickle
import socket
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rpyc
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import ObservationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@torch.jit.script
def euler_to_quaternion(euler_angles: torch.Tensor) -> torch.Tensor:
    cy = torch.cos(euler_angles[:, 2] * 0.5)
    sy = torch.sin(euler_angles[:, 2] * 0.5)
    cp = torch.cos(euler_angles[:, 1] * 0.5)
    sp = torch.sin(euler_angles[:, 1] * 0.5)
    cr = torch.cos(euler_angles[:, 0] * 0.5)
    sr = torch.sin(euler_angles[:, 0] * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack((w, x, y, z), dim=-1)


class GSServer:
    """Receives rendered RGB batches from vrrobo_renderer over TCP."""

    def __init__(self, host: str = "localhost", port: int = 12345, timeout_s: float = 10.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.thread = None
        self.running = False
        self.lock = threading.Lock()

    def init_data(self, env_num: int, h: int = 180, w: int = 320):
        self.h = h
        self.w = w
        self.data = np.zeros((env_num, 3 * h * w), dtype=np.uint8)
        self.last_data = np.zeros((env_num, 3 * h * w), dtype=np.uint8)
        self.latency = np.random.randint(0, 2, size=(env_num, 1), dtype=np.int32)

    def _receive_once(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(1)
        conn, _ = sock.accept()
        conn.settimeout(self.timeout_s)

        payload = b""
        try:
            while True:
                packet = conn.recv(40960000)
                if not packet:
                    break
                payload += packet
        except socket.timeout:
            pass
        finally:
            conn.close()
            sock.close()

        if not payload:
            return None
        return pickle.loads(payload)

    def start(self):
        atexit.register(self.close)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def close(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.5)

    def _run(self):
        while self.running:
            arr = self._receive_once()
            if arr is None:
                continue
            with self.lock:
                self.last_data = self.data
                self.data = arr

    def get_data(self):
        with self.lock:
            is_start = (self.last_data == 0).all(axis=1).reshape(-1, 1)
            out = (self.latency * self.last_data + (1 - self.latency) * self.data) * (1 - is_start) + self.data * is_start
            return np.clip(out, 0, 255).astype(np.uint8)

    def reset(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            return
        ids = env_ids.cpu().numpy()
        self.data[ids] = 0
        self.last_data[ids] = 0
        self.latency[ids] = np.random.randint(0, 2, size=(len(ids), 1), dtype=np.int32)


def rgb_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def goal_pos_multi(env: ManagerBasedRLEnv, base_height: float = 0.0) -> torch.Tensor:
    red = env.scene["cone_red"].data.object_pos_w[:, 0, :] - env.scene.env_origins
    green = env.scene["cone_green"].data.object_pos_w[:, 0, :] - env.scene.env_origins
    blue = env.scene["cone_blue"].data.object_pos_w[:, 0, :] - env.scene.env_origins
    goals = torch.stack([red, green, blue], dim=1)
    goals[:, :, 2] += base_height
    return goals.reshape(env.num_envs, -1)


def selected_goal_pos(env: ManagerBasedRLEnv, command_name: str, base_height: float = 0.0) -> torch.Tensor:
    commands = env.command_manager.get_command(command_name)
    red = env.scene["cone_red"].data.object_pos_w[:, 0, :] - env.scene.env_origins
    green = env.scene["cone_green"].data.object_pos_w[:, 0, :] - env.scene.env_origins
    blue = env.scene["cone_blue"].data.object_pos_w[:, 0, :] - env.scene.env_origins
    goals = torch.stack([red, green, blue], dim=1)
    goals[:, :, 2] += base_height
    return (goals * commands.unsqueeze(-1)).sum(dim=1)


def root_pos_e(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_pos_w - env.scene.env_origins


class gs_image_feature(ManagerTermBase):
    """RPC-based GS rendering + frozen vision encoder (VR-Robo style)."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.conn = rpyc.connect("localhost", 18861, config={"allow_pickle": True, "allow_public_attrs": True})
        self.image_server = GSServer()
        self.image_server.start()
        self.image_server.init_data(env.num_envs, h=180, w=320)

        self.encoder_model, self.output_dim, mean, std = self._build_encoder(env.device)
        self.encoder_model.eval()

        self.preprocess = T.Compose(
            [
                T.Resize((224, 224)),
                T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0))], p=0.6),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.camera_pos_noise_scale = torch.tensor([0.01, 0.01, 0.01], device=env.device)
        self.camera_rot_noise_scale = torch.tensor([1.0, 1.0, 2.0], device=env.device)

        # Optional debug image dump from external renderer.
        self.save_debug_images = bool(cfg.params.get("save_debug_images", False))
        self.save_every_n_steps = max(1, int(cfg.params.get("save_every_n_steps", 20)))
        self.save_max_images = max(1, int(cfg.params.get("save_max_images", 100)))
        self.save_env_index = int(cfg.params.get("save_env_index", 0))
        self._obs_step = 0
        self._saved_count = 0
        save_dir = cfg.params.get("save_dir", "logs/gs_render_debug")
        self.save_dir = Path(save_dir)
        if self.save_debug_images:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _build_encoder(self, device: str):
        try:
            import timm

            model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
            model.head = nn.Identity()
            model = model.to(device)
            return model, int(model.num_features), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        except Exception:
            from torchvision import models

            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model = nn.Sequential(*list(model.children())[:-1]).to(device)
            return model, 512, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def reset(self, env_ids: torch.Tensor | None = None):
        self.image_server.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        camera_pos: list[float] | tuple[float, float, float] = (0.25, 0.0, 0.15),
        camera_rot: list[float] | tuple[float, float, float] = (0.0, 20.0, 0.0),
        asset_offset_pos: list[float] | tuple[float, float, float] = (3.2, 0.0, -0.01),
        save_debug_images: bool | None = None,
        save_every_n_steps: int | None = None,
        save_max_images: int | None = None,
        save_env_index: int | None = None,
        save_dir: str | None = None,
    ) -> torch.Tensor:
        # Accept debug params from ObservationTermCfg to satisfy manager param validation.
        # Runtime overrides are optional and primarily for debugging.
        if save_debug_images is not None:
            self.save_debug_images = bool(save_debug_images)
        if save_every_n_steps is not None:
            self.save_every_n_steps = max(1, int(save_every_n_steps))
        if save_max_images is not None:
            self.save_max_images = max(1, int(save_max_images))
        if save_env_index is not None:
            self.save_env_index = int(save_env_index)
        if save_dir is not None and str(self.save_dir) != str(save_dir):
            self.save_dir = Path(save_dir)
            if self.save_debug_images:
                self.save_dir.mkdir(parents=True, exist_ok=True)

        robot: Articulation = env.scene[asset_cfg.name]
        pos_r = robot.data.root_pos_w - env.scene.env_origins
        quat_r = robot.data.root_quat_w

        cam_pos = torch.tensor(camera_pos, device=env.device).repeat(env.num_envs, 1)
        cam_pos += (2 * torch.rand_like(cam_pos) - 1) * self.camera_pos_noise_scale
        cam_rot = torch.tensor(camera_rot, device=env.device).repeat(env.num_envs, 1)
        cam_rot += (2 * torch.rand_like(cam_rot) - 1) * self.camera_rot_noise_scale
        cam_rot = euler_to_quaternion(torch.deg2rad(cam_rot))

        offset = torch.tensor(asset_offset_pos, device=env.device).repeat(env.num_envs, 1)
        cam_pos_w = pos_r + math_utils.quat_apply(quat_r, cam_pos) - offset
        cam_quat_w = math_utils.quat_mul(quat_r, cam_rot)
        cam_quat_ros = math_utils.convert_camera_frame_orientation_convention(cam_quat_w, "world", "ros")

        red: RigidObjectCollection = env.scene["cone_red"]
        green: RigidObjectCollection = env.scene["cone_green"]
        blue: RigidObjectCollection = env.scene["cone_blue"]
        red_pos = red.data.object_pos_w[:, 0, :] - env.scene.env_origins - offset
        green_pos = green.data.object_pos_w[:, 0, :] - env.scene.env_origins - offset
        blue_pos = blue.data.object_pos_w[:, 0, :] - env.scene.env_origins - offset

        self.conn.root.render(cam_pos_w, cam_quat_ros, red_pos, green_pos, blue_pos)
        images_np = self.image_server.get_data()
        self._maybe_save_debug_image(images_np)
        images = torch.tensor(images_np, dtype=torch.float32, device=env.device).reshape(env.num_envs, 3, 180, 320)
        images = images / 255.0
        images = self.preprocess(images)

        with torch.inference_mode():
            features = self.encoder_model(images)
            if features.dim() > 2:
                features = features.view(features.shape[0], -1)
        return features

    def _maybe_save_debug_image(self, images_np: np.ndarray):
        self._obs_step += 1
        if not self.save_debug_images:
            return
        if self._saved_count >= self.save_max_images:
            return
        if self._obs_step % self.save_every_n_steps != 0:
            return
        if images_np.ndim != 2 or images_np.shape[0] == 0:
            return

        env_idx = min(max(self.save_env_index, 0), images_np.shape[0] - 1)
        try:
            img = images_np[env_idx].reshape(3, 180, 320).transpose(1, 2, 0)
            out_path = self.save_dir / f"render_env{env_idx:02d}_step{self._obs_step:06d}.png"
            Image.fromarray(img).save(out_path)
            self._saved_count += 1
        except Exception:
            # Keep training running even if image dump fails.
            return
