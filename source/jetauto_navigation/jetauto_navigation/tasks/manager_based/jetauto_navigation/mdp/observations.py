from __future__ import annotations

import atexit
import pickle
import socket
import threading
import time
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
from .multitask_inference import (
    DEFAULT_MULTITASK_MODEL_PATH,
    DEFAULT_OCCLUSION_CLASS_NAMES,
    MultiTaskOcclusionPredictor,
)

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
            if payload:
                print(
                    f"[GSServer] socket timeout with partial payload bytes={len(payload)} host={self.host} port={self.port}",
                    flush=True,
                )
            pass
        finally:
            conn.close()
            sock.close()

        if not payload:
            return None
        try:
            return pickle.loads(payload)
        except Exception as exc:
            print(
                f"[GSServer] failed to decode payload bytes={len(payload)} host={self.host} port={self.port} error={exc}",
                flush=True,
            )
            raise

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
            try:
                arr = self._receive_once()
            except Exception as exc:
                print(f"[GSServer] receiver loop exception: {exc}", flush=True)
                raise
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
    """RPC-based GS rendering with a shared DeepLab backbone for rewards and policy features."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        print("[gs_image_feature] Connecting to render server on localhost:18861...")
        self.conn = rpyc.connect("localhost", 18861, config={"allow_pickle": True, "allow_public_attrs": True})
        print("[gs_image_feature] Render server connected.")
        self.image_server = GSServer()
        self.image_server.start()
        self.image_server.init_data(env.num_envs, h=180, w=320)
        print("[gs_image_feature] RGB socket receiver ready.")
        print("[gs_image_feature] Initializing multitask occlusion predictor...")
        self.occlusion_predictor = MultiTaskOcclusionPredictor(
            model_path=str(cfg.params.get("multitask_model_path", DEFAULT_MULTITASK_MODEL_PATH)),
            project_root=cfg.params.get("multitask_project_root"),
            device=env.device,
        )
        self.occlusion_class_names = tuple(self.occlusion_predictor.occlusion_class_names)
        self.success_occlusion_class = str(cfg.params.get("success_occlusion_class", "0-20%"))
        self.success_occlusion_index = self.occlusion_predictor.class_index(self.success_occlusion_class)
        self.randomize_occlusion_prediction = bool(cfg.params.get("randomize_occlusion_prediction", False))
        print("[gs_image_feature] Multitask occlusion predictor ready.")
        if self.randomize_occlusion_prediction:
            print("[gs_image_feature] Random occlusion labels enabled. Image features still come from renderer inputs.")

        print("[gs_image_feature] Reusing DeepLab backbone features for the policy observation.")
        self.output_dim = int(self.occlusion_predictor.feature_dim)
        print(f"[gs_image_feature] Shared backbone feature_dim={self.output_dim}")
        # Legacy ViT/ResNet policy encoder path retained for reference.
        # print("[gs_image_feature] Initializing policy encoder...")
        # self.encoder_model, self.output_dim, mean, std = self._build_encoder(env.device)
        # self.encoder_model.eval()
        # print(f"[gs_image_feature] Policy encoder ready. output_dim={self.output_dim}")
        self.history_len = max(1, int(cfg.params.get("history_len", 4)))
        self.feature_history = torch.zeros(
            (env.num_envs, self.history_len, self.output_dim),
            device=env.device,
            dtype=torch.float32,
        )
        # Legacy ViT/ResNet preprocessing retained for reference.
        # self.preprocess = T.Compose(
        #     [
        #         T.Resize((224, 224)),
        #         T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
        #         T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0))], p=0.6),
        #         T.Normalize(mean=mean, std=std),
        #     ]
        # )
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



        # Optional debug mask dump from external renderer.
        self.save_debug_masks = bool(cfg.params.get("save_debug_masks", False))
        self.save_mask_every_n_steps = max(1, int(cfg.params.get("save_mask_every_n_steps", 1)))
        self.save_mask_max_images = int(cfg.params.get("save_mask_max_images", -1))
        self.save_mask_env_index = int(cfg.params.get("save_mask_env_index", self.save_env_index))
        self.mask_occluded_dir = Path(cfg.params.get("mask_occluded_dir", "logs/gs_mask_debug/occluded"))
        self.mask_target_only_dir = Path(cfg.params.get("mask_target_only_dir", "logs/gs_mask_debug/target_only"))
        self.mask_target_from_command = bool(cfg.params.get("mask_target_from_command", True))
        self.mask_command_name = str(cfg.params.get("mask_command_name", "rgb_command"))
        self.mask_target_default = str(cfg.params.get("mask_target_default", "red"))
        self.mask_threshold = float(cfg.params.get("mask_threshold", 0.5))
        self.mask_binary = bool(cfg.params.get("mask_binary", True))
        self._saved_mask_count = 0
        self._mask_api_warned = False
        if self.save_debug_masks:
            self.mask_occluded_dir.mkdir(parents=True, exist_ok=True)
            self.mask_target_only_dir.mkdir(parents=True, exist_ok=True)

    # Legacy ViT/ResNet encoder path retained for reference.
    # def _build_encoder(self, device: str):
    #     try:
    #         import timm
    #
    #         print("[gs_image_feature] Loading timm vit_tiny_patch16_224 pretrained weights...")
    #         model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    #         model.head = nn.Identity()
    #         model = model.to(device)
    #         return model, int(model.num_features), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    #     except Exception:
    #         from torchvision import models
    #
    #         print("[gs_image_feature] Falling back to torchvision resnet18 pretrained weights...")
    #         model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    #         model = nn.Sequential(*list(model.children())[:-1]).to(device)
    #         return model, 512, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def reset(self, env_ids: torch.Tensor | None = None):
        self.image_server.reset(env_ids)
        if env_ids is None:
            self.feature_history = torch.zeros_like(self.feature_history)
            return
        feature_history = self.feature_history.clone()
        feature_history[env_ids] = 0.0
        self.feature_history = feature_history

    @staticmethod
    def _summarize_value(name: str, value, max_items: int = 4) -> str:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if getattr(tensor, "names", None) is not None and any(n is not None for n in tensor.names):
                tensor = tensor.rename(None)
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype)
            if tensor.numel() == 0:
                return f"{name}: tensor shape={shape} dtype={dtype} empty"
            tensor_cpu = tensor.float().cpu()
            flat = tensor_cpu.reshape(-1)
            finite = torch.isfinite(flat)
            finite_count = int(finite.sum().item())
            sample = flat[:max_items].tolist()
            if finite_count > 0:
                finite_values = flat[finite]
                min_value = float(finite_values.min().item())
                max_value = float(finite_values.max().item())
                mean_value = float(finite_values.mean().item())
            else:
                min_value = float("nan")
                max_value = float("nan")
                mean_value = float("nan")
            return (
                f"{name}: tensor shape={shape} dtype={dtype} "
                f"finite={finite_count}/{flat.numel()} min={min_value:.6g} "
                f"max={max_value:.6g} mean={mean_value:.6g} sample={sample}"
            )
        if isinstance(value, np.ndarray):
            array = value
            shape = tuple(array.shape)
            dtype = str(array.dtype)
            if array.size == 0:
                return f"{name}: ndarray shape={shape} dtype={dtype} empty"
            flat = array.reshape(-1)
            if np.issubdtype(array.dtype, np.number):
                finite = np.isfinite(flat)
                finite_count = int(finite.sum())
                sample = flat[:max_items].tolist()
                if finite_count > 0:
                    finite_values = flat[finite]
                    min_value = float(finite_values.min())
                    max_value = float(finite_values.max())
                    mean_value = float(finite_values.mean())
                else:
                    min_value = float("nan")
                    max_value = float("nan")
                    mean_value = float("nan")
                return (
                    f"{name}: ndarray shape={shape} dtype={dtype} "
                    f"finite={finite_count}/{flat.size} min={min_value:.6g} "
                    f"max={max_value:.6g} mean={mean_value:.6g} sample={sample}"
                )
            return f"{name}: ndarray shape={shape} dtype={dtype}"
        return f"{name}: type={type(value).__name__} value={value!r}"

    def _log_debug_context(self, stage: str, env: ManagerBasedRLEnv, **values) -> None:
        common_step = int(getattr(env, "common_step_counter", -1))
        sim_step = int(getattr(env, "_sim_step_counter", -1))
        try:
            episode_min = int(env.episode_length_buf.min().item())
            episode_max = int(env.episode_length_buf.max().item())
        except Exception:
            episode_min = -1
            episode_max = -1
        print(
            "[gs_image_feature] "
            f"debug stage={stage} common_step={common_step} sim_step={sim_step} "
            f"obs_step={self._obs_step} episode_len_min={episode_min} episode_len_max={episode_max}",
            flush=True,
        )
        print(f"[gs_image_feature] cuda {self._summarize_cuda_memory(env.device)}", flush=True)
        startup_cuda_summary = self._startup_cuda_memory_summary(env)
        if startup_cuda_summary is not None:
            print(f"[gs_image_feature] startup cuda {startup_cuda_summary}", flush=True)
        for name, value in values.items():
            print(f"[gs_image_feature] {self._summarize_value(name, value)}", flush=True)

    @staticmethod
    def _summarize_cuda_memory(device) -> str:
        if not torch.cuda.is_available():
            return "unavailable"
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

    @staticmethod
    def _startup_cuda_memory_summary(env: ManagerBasedRLEnv) -> str | None:
        summary = getattr(env, "_startup_cuda_memory_summary", None)
        if summary:
            return str(summary)
        return None

    def _warn_if_suspicious_inputs(
        self,
        env: ManagerBasedRLEnv,
        pos_r: torch.Tensor,
        quat_r: torch.Tensor,
        cam_pos_w: torch.Tensor,
        cam_quat_ros: torch.Tensor,
        red_pos: torch.Tensor,
        green_pos: torch.Tensor,
        blue_pos: torch.Tensor,
    ) -> None:
        warnings: list[str] = []
        robot_quat_norm = quat_r.norm(dim=-1)
        cam_quat_norm = cam_quat_ros.norm(dim=-1)
        robot_bad = (robot_quat_norm - 1.0).abs() > 1e-2
        cam_bad = (cam_quat_norm - 1.0).abs() > 1e-2
        if bool(robot_bad.any()):
            warnings.append(f"robot_quat_norm_bad_envs={robot_bad.nonzero(as_tuple=False)[:8].flatten().tolist()}")
        if bool(cam_bad.any()):
            warnings.append(f"cam_quat_norm_bad_envs={cam_bad.nonzero(as_tuple=False)[:8].flatten().tolist()}")
        for name, tensor in (
            ("robot_pos", pos_r),
            ("robot_quat", quat_r),
            ("cam_pos_w", cam_pos_w),
            ("cam_quat_ros", cam_quat_ros),
            ("red_pos", red_pos),
            ("green_pos", green_pos),
            ("blue_pos", blue_pos),
        ):
            if isinstance(tensor, torch.Tensor) and tensor.numel() > 0 and tensor.is_floating_point():
                if not bool(torch.isfinite(tensor).all()):
                    warnings.append(f"{name}=non_finite")
        if warnings:
            print("[gs_image_feature] suspicious render inputs: " + "; ".join(warnings), flush=True)
            self._log_debug_context(
                "pre_render_warning",
                env,
                robot_pos=pos_r,
                robot_quat=quat_r,
                cam_pos_w=cam_pos_w,
                cam_quat_ros=cam_quat_ros,
                red_pos=red_pos,
                green_pos=green_pos,
                blue_pos=blue_pos,
                robot_quat_norm=robot_quat_norm,
                cam_quat_norm=cam_quat_norm,
            )


    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        camera_pos: list[float] | tuple[float, float, float] = (0.25, 0.0, 0.15),
        camera_rot: list[float] | tuple[float, float, float] = (0.0, 20.0, 0.0),
        asset_offset_pos: list[float] | tuple[float, float, float] = (3.2, 0.0, -0.01),
        multitask_model_path: str | None = None,
        multitask_project_root: str | None = None,
        success_occlusion_class: str | None = None,
        save_debug_images: bool | None = None,
        save_every_n_steps: int | None = None,
        save_max_images: int | None = None,
        save_env_index: int | None = None,
        save_dir: str | None = None,
        save_debug_masks: bool | None = None,
        save_mask_every_n_steps: int | None = None,
        save_mask_max_images: int | None = None,
        save_mask_env_index: int | None = None,
        mask_occluded_dir: str | None = None,
        mask_target_only_dir: str | None = None,
        mask_target_from_command: bool | None = None,
        mask_command_name: str | None = None,
        mask_target_default: str | None = None,
        mask_threshold: float | None = None,
        mask_binary: bool | None = None,
        randomize_occlusion_prediction: bool | None = None,
        history_len: int | None = None,
    ) -> torch.Tensor:
        # print(f"[gs_image_feature] called, step={env.common_step_counter}")
        # Accept debug params from ObservationTermCfg to satisfy manager param validation.
        # Runtime overrides are optional and primarily for debugging.
        # These model-related params are configured during initialization; they are accepted here
        # to satisfy ObservationTermCfg validation and to allow a lightweight runtime override of
        # the success bucket without reloading the model every step.
        if multitask_model_path is not None:
            pass
        if multitask_project_root is not None:
            pass
        if success_occlusion_class is not None and success_occlusion_class != self.success_occlusion_class:
            self.success_occlusion_class = str(success_occlusion_class)
            self.success_occlusion_index = self.occlusion_predictor.class_index(self.success_occlusion_class)
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
        if save_debug_masks is not None:
            self.save_debug_masks = bool(save_debug_masks)
        if save_mask_every_n_steps is not None:
            self.save_mask_every_n_steps = max(1, int(save_mask_every_n_steps))
        if save_mask_max_images is not None:
            self.save_mask_max_images = int(save_mask_max_images)
        if save_mask_env_index is not None:
            self.save_mask_env_index = int(save_mask_env_index)
        if mask_occluded_dir is not None and str(self.mask_occluded_dir) != str(mask_occluded_dir):
            self.mask_occluded_dir = Path(mask_occluded_dir)
        if mask_target_only_dir is not None and str(self.mask_target_only_dir) != str(mask_target_only_dir):
            self.mask_target_only_dir = Path(mask_target_only_dir)
        if self.save_debug_masks:
            self.mask_occluded_dir.mkdir(parents=True, exist_ok=True)
            self.mask_target_only_dir.mkdir(parents=True, exist_ok=True)
        if mask_target_from_command is not None:
            self.mask_target_from_command = bool(mask_target_from_command)
        if mask_command_name is not None:
            self.mask_command_name = str(mask_command_name)
        if mask_target_default is not None:
            self.mask_target_default = str(mask_target_default)
        if mask_threshold is not None:
            self.mask_threshold = float(mask_threshold)
        if mask_binary is not None:
            self.mask_binary = bool(mask_binary)
        if randomize_occlusion_prediction is not None:
            self.randomize_occlusion_prediction = bool(randomize_occlusion_prediction)
        if history_len is not None and int(history_len) != self.history_len:
            raise ValueError(
                f"gs_image_feature was initialized with history_len={self.history_len}, "
                f"but received runtime history_len={int(history_len)}."
            )

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




        def _strip_names(t: torch.Tensor) -> torch.Tensor:
            # drop named-tensor metadata (cheap)
            if isinstance(t, torch.Tensor) and getattr(t, "names", None) is not None:
                # names is a tuple; any non-None means it's named
                if any(n is not None for n in t.names):
                    t = t.rename(None)
            return t

        cam_pos_w = _strip_names(cam_pos_w)
        cam_quat_ros = _strip_names(cam_quat_ros)
        red_pos = _strip_names(red_pos)
        green_pos = _strip_names(green_pos)
        blue_pos = _strip_names(blue_pos)

        self._warn_if_suspicious_inputs(env, pos_r, quat_r, cam_pos_w, cam_quat_ros, red_pos, green_pos, blue_pos)

        render_started = time.perf_counter()
        try:
            self.conn.root.render(cam_pos_w, cam_quat_ros, red_pos, green_pos, blue_pos)
        except Exception as exc:
            self._log_debug_context(
                "render_exception",
                env,
                error=str(exc),
                robot_pos=pos_r,
                robot_quat=quat_r,
                cam_pos_w=cam_pos_w,
                cam_quat_ros=cam_quat_ros,
                red_pos=red_pos,
                green_pos=green_pos,
                blue_pos=blue_pos,
            )
            raise
        render_elapsed = time.perf_counter() - render_started
        if render_elapsed > 5.0:
            print(
                "[gs_image_feature] slow render call "
                f"common_step={int(getattr(env, 'common_step_counter', -1))} "
                f"sim_step={int(getattr(env, '_sim_step_counter', -1))} "
                f"elapsed_s={render_elapsed:.3f}",
                flush=True,
            )

        try:
            images_np = self.image_server.get_data()
        except Exception as exc:
            self._log_debug_context(
                "get_data_exception",
                env,
                error=str(exc),
                cam_pos_w=cam_pos_w,
                cam_quat_ros=cam_quat_ros,
            )
            raise
        if not isinstance(images_np, np.ndarray) or images_np.shape != (env.num_envs, 3 * 180 * 320):
            self._log_debug_context("unexpected_image_buffer", env, images_np=images_np)
        self._maybe_save_debug_image(images_np)
        if self.save_debug_masks and not self._mask_api_warned:
            print("[gs_image_feature] save_debug_masks is ignored when using the RGB-only render server.")
            self._mask_api_warned = True

        try:
            reward_images = torch.tensor(images_np, dtype=torch.float32, device=env.device).reshape(
                env.num_envs, 3, 180, 320
            )
        except Exception as exc:
            self._log_debug_context("reward_image_tensor_exception", env, error=str(exc), images_np=images_np)
            raise
        reward_images = reward_images / 255.0
        common_step = int(getattr(env, "common_step_counter", -1))
        startup_cuda_summary = self._startup_cuda_memory_summary(env)
        # Disabled periodic predictor memory debug logs to reduce terminal noise during training.
        # if common_step > 0 and common_step % 40 == 0:
        #     print(
        #         "[gs_image_feature] predictor pre "
        #         f"common_step={common_step} sim_step={int(getattr(env, '_sim_step_counter', -1))} "
        #         f"current[{self._summarize_cuda_memory(env.device)}]"
        #         f"{f' startup[{startup_cuda_summary}]' if startup_cuda_summary else ''}",
        #         flush=True,
        #     )
        predict_started = time.perf_counter()
        try:
            occ_indices, occ_probs, features = self.occlusion_predictor.predict_with_features(reward_images)
        except Exception as exc:
            self._log_debug_context(
                "predict_with_features_exception",
                env,
                error=str(exc),
                reward_images=reward_images,
                images_np=images_np,
            )
            raise
        predict_elapsed = time.perf_counter() - predict_started
        # if common_step > 0 and common_step % 40 == 0:
        #     print(
        #         "[gs_image_feature] predictor post "
        #         f"common_step={common_step} sim_step={int(getattr(env, '_sim_step_counter', -1))} "
        #         f"current[{self._summarize_cuda_memory(env.device)}]"
        #         f"{f' startup[{startup_cuda_summary}]' if startup_cuda_summary else ''}",
        #         flush=True,
        #     )
        if predict_elapsed > 5.0:
            print(
                "[gs_image_feature] slow predictor call "
                f"common_step={int(getattr(env, 'common_step_counter', -1))} "
                f"sim_step={int(getattr(env, '_sim_step_counter', -1))} "
                f"elapsed_s={predict_elapsed:.3f}",
                flush=True,
            )
        if self.randomize_occlusion_prediction:
            occ_indices, occ_probs = self._sample_random_occlusion_prediction(env)
        if isinstance(occ_probs, torch.Tensor) and not bool(torch.isfinite(occ_probs).all()):
            self._log_debug_context("occ_probs_non_finite", env, occ_probs=occ_probs)
        if isinstance(features, torch.Tensor) and not bool(torch.isfinite(features).all()):
            self._log_debug_context("features_non_finite", env, features=features)
        env.extras["pred_occ_class"] = occ_indices.to(env.device)
        env.extras["pred_occ_probs"] = occ_probs.to(env.device)
        env.extras["pred_occ_class_names"] = list(self.occlusion_class_names)
        env.extras["pred_occ_success_index"] = int(self.success_occlusion_index)
        env.extras["pred_occ_success_class"] = self.success_occlusion_class
        env.extras["pred_occ_success_mask"] = occ_indices == self.success_occlusion_index
        env.extras["pred_occ_success_rate"] = float((occ_indices == self.success_occlusion_index).float().mean().item())

        with torch.inference_mode():
            features = features.to(env.device)
            # Legacy ViT/ResNet policy-feature extraction retained for reference.
            # images = self.preprocess(reward_images)
            # features = self.encoder_model(images)
            # if features.dim() > 2:
            #     features = features.view(features.shape[0], -1)
            reset_mask = env.episode_length_buf == 0
        if reset_mask.any():
            self.feature_history[reset_mask] = features[reset_mask].unsqueeze(1).repeat(1, self.history_len, 1)

        self.feature_history = torch.roll(self.feature_history, shifts=-1, dims=1)
        self.feature_history[:, -1, :] = features
        return self.feature_history.reshape(env.num_envs, self.history_len * self.output_dim)
        # return features

    def _sample_random_occlusion_prediction(self, env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.rand((env.num_envs, len(self.occlusion_class_names)), device=env.device, dtype=torch.float32)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        occ_indices = probs.argmax(dim=-1)
        return occ_indices, probs

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

    @staticmethod
    def _command_to_target(command_row: torch.Tensor, default_target: str = "red") -> str:
        if command_row.numel() < 3:
            return default_target
        target_idx = int(torch.argmax(command_row[:3]).item())
        return ["red", "green", "blue"][target_idx]

    def _maybe_save_debug_masks(
        self,
        env: ManagerBasedRLEnv,
        cam_pos_w: torch.Tensor,
        cam_quat_ros: torch.Tensor,
        red_pos: torch.Tensor,
        green_pos: torch.Tensor,
        blue_pos: torch.Tensor,
    ):
        if not self.save_debug_masks:
            return
        if self.save_mask_max_images >= 0 and self._saved_mask_count >= self.save_mask_max_images:
            return
        if self._obs_step % self.save_mask_every_n_steps != 0:
            return

        env_idx = min(max(self.save_mask_env_index, 0), cam_pos_w.shape[0] - 1)
        target_name = self.mask_target_default
        if self.mask_target_from_command:
            try:
                command = env.command_manager.get_command(self.mask_command_name)
                target_name = self._command_to_target(command[env_idx], self.mask_target_default)
            except Exception:
                target_name = self.mask_target_default

        try:
            cam_pos_1 = cam_pos_w[env_idx : env_idx + 1]
            cam_quat_1 = cam_quat_ros[env_idx : env_idx + 1]
            red_pos_1 = red_pos[env_idx : env_idx + 1]
            green_pos_1 = green_pos[env_idx : env_idx + 1]
            blue_pos_1 = blue_pos[env_idx : env_idx + 1]

            mask_occ = self.conn.root.render_mask(
                cam_pos_1,
                cam_quat_1,
                red_pos_1,
                green_pos_1,
                blue_pos_1,
                target=target_name,
                threshold=self.mask_threshold,
                binary=self.mask_binary,
                send_to_socket=False,
                occlusion_mode="full_scene",
            )
            mask_target_only = self.conn.root.render_mask(
                cam_pos_1,
                cam_quat_1,
                red_pos_1,
                green_pos_1,
                blue_pos_1,
                target=target_name,
                threshold=self.mask_threshold,
                binary=self.mask_binary,
                send_to_socket=False,
                occlusion_mode="target_only",
            )

            mask_occ_img = np.asarray(mask_occ, dtype=np.uint8)[0]
            mask_target_only_img = np.asarray(mask_target_only, dtype=np.uint8)[0]
            occ_path = self.mask_occluded_dir / f"mask_occ_env{env_idx:02d}_step{self._obs_step:06d}_{target_name}.png"
            target_only_path = (
                self.mask_target_only_dir / f"mask_target_only_env{env_idx:02d}_step{self._obs_step:06d}_{target_name}.png"
            )
            Image.fromarray(mask_occ_img, mode="L").save(occ_path)
            Image.fromarray(mask_target_only_img, mode="L").save(target_only_path)
            self._saved_mask_count += 1
        except Exception as exc:
            if not self._mask_api_warned:
                print(
                    "[gs_image_feature] mask dump failed once. "
                    f"error={exc}. Please run render_server_with_mask.py."
                )
                self._mask_api_warned = True
            # Keep training running even if mask dump fails.
            return


class random_occlusion_feature(ManagerTermBase):
    """Testing observation term that bypasses renderer and predictor."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.occlusion_class_names = tuple(
            str(name) for name in cfg.params.get("occlusion_class_names", DEFAULT_OCCLUSION_CLASS_NAMES)
        )
        if not self.occlusion_class_names:
            raise ValueError("random_occlusion_feature requires at least one occlusion class name.")

        self.success_occlusion_class = str(cfg.params.get("success_occlusion_class", self.occlusion_class_names[0]))
        self.success_occlusion_index = self._class_index(self.success_occlusion_class)
        self.output_dim = max(1, int(cfg.params.get("feature_dim", 320)))
        self.history_len = max(1, int(cfg.params.get("history_len", 4)))
        self.feature_mode = str(cfg.params.get("feature_mode", "zeros")).lower()
        self.feature_history = torch.zeros(
            (env.num_envs, self.history_len, self.output_dim),
            device=env.device,
            dtype=torch.float32,
        )
        print(
            "[random_occlusion_feature] Using synthetic occlusion labels. "
            f"classes={self.occlusion_class_names} feature_dim={self.output_dim} feature_mode={self.feature_mode}"
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.feature_history = torch.zeros_like(self.feature_history)
            return
        feature_history = self.feature_history.clone()
        feature_history[env_ids] = 0.0
        self.feature_history = feature_history

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        success_occlusion_class: str | None = None,
        occlusion_class_names: list[str] | tuple[str, ...] | None = None,
        feature_dim: int | None = None,
        history_len: int | None = None,
        feature_mode: str | None = None,
    ) -> torch.Tensor:
        if success_occlusion_class is not None and success_occlusion_class != self.success_occlusion_class:
            self.success_occlusion_class = str(success_occlusion_class)
            self.success_occlusion_index = self._class_index(self.success_occlusion_class)

        if occlusion_class_names is not None:
            class_names = tuple(str(name) for name in occlusion_class_names)
            if class_names and class_names != self.occlusion_class_names:
                self.occlusion_class_names = class_names
                self.success_occlusion_index = self._class_index(self.success_occlusion_class)

        if feature_dim is not None and int(feature_dim) != self.output_dim:
            raise ValueError(
                f"random_occlusion_feature was initialized with feature_dim={self.output_dim}, "
                f"but received runtime feature_dim={int(feature_dim)}."
            )

        if history_len is not None and int(history_len) != self.history_len:
            raise ValueError(
                f"random_occlusion_feature was initialized with history_len={self.history_len}, "
                f"but received runtime history_len={int(history_len)}."
            )

        if feature_mode is not None:
            self.feature_mode = str(feature_mode).lower()

        probs = torch.rand((env.num_envs, len(self.occlusion_class_names)), device=env.device, dtype=torch.float32)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        occ_indices = probs.argmax(dim=-1)
        features = self._make_features(env)

        env.extras["pred_occ_class"] = occ_indices
        env.extras["pred_occ_probs"] = probs
        env.extras["pred_occ_class_names"] = list(self.occlusion_class_names)
        env.extras["pred_occ_success_index"] = int(self.success_occlusion_index)
        env.extras["pred_occ_success_class"] = self.success_occlusion_class
        env.extras["pred_occ_success_mask"] = occ_indices == self.success_occlusion_index
        env.extras["pred_occ_success_rate"] = float((occ_indices == self.success_occlusion_index).float().mean().item())

        reset_mask = env.episode_length_buf == 0
        if reset_mask.any():
            self.feature_history[reset_mask] = features[reset_mask].unsqueeze(1).repeat(1, self.history_len, 1)

        self.feature_history = torch.roll(self.feature_history, shifts=-1, dims=1)
        self.feature_history[:, -1, :] = features
        return self.feature_history.reshape(env.num_envs, self.history_len * self.output_dim)

    def _make_features(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        if self.feature_mode == "normal":
            return torch.randn((env.num_envs, self.output_dim), device=env.device, dtype=torch.float32)
        if self.feature_mode == "uniform":
            return torch.rand((env.num_envs, self.output_dim), device=env.device, dtype=torch.float32)
        if self.feature_mode == "ones":
            return torch.ones((env.num_envs, self.output_dim), device=env.device, dtype=torch.float32)
        return torch.zeros((env.num_envs, self.output_dim), device=env.device, dtype=torch.float32)

    def _class_index(self, class_name: str) -> int:
        try:
            return self.occlusion_class_names.index(str(class_name))
        except ValueError as exc:
            raise ValueError(
                f"Unknown occlusion class '{class_name}'. Available classes: {self.occlusion_class_names}"
            ) from exc
