from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rpyc
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import ObservationTermCfg

from ..mdp.multitask_inference import DEFAULT_MULTITASK_MODEL_PATH
from ..mdp.observations import GSServer, euler_to_quaternion, gs_image_feature
from .multitask_inference import MidOccMultiTaskOcclusionPredictor


def _matrix_to_quaternion_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    m00 = matrix[:, 0, 0]
    m01 = matrix[:, 0, 1]
    m02 = matrix[:, 0, 2]
    m10 = matrix[:, 1, 0]
    m11 = matrix[:, 1, 1]
    m12 = matrix[:, 1, 2]
    m20 = matrix[:, 2, 0]
    m21 = matrix[:, 2, 1]
    m22 = matrix[:, 2, 2]

    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0))
    qx = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0))
    qy = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0))
    qz = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0))

    qx = torch.copysign(qx, m21 - m12)
    qy = torch.copysign(qy, m02 - m20)
    qz = torch.copysign(qz, m10 - m01)
    quat = torch.stack((qw, qx, qy, qz), dim=-1)
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _look_at_quaternion_for_render_server_torch(camera_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
    up_hint = torch.tensor([0.0, 0.0, 1.0], device=camera_pos.device, dtype=camera_pos.dtype).repeat(
        camera_pos.shape[0], 1
    )
    fallback_up = torch.tensor([0.0, 1.0, 0.0], device=camera_pos.device, dtype=camera_pos.dtype).repeat(
        camera_pos.shape[0], 1
    )

    forward = target_pos - camera_pos
    forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    near_parallel = (forward * up_hint).sum(dim=-1).abs() > 0.98
    up_hint = torch.where(near_parallel.unsqueeze(-1), fallback_up, up_hint)

    right = torch.cross(forward, up_hint, dim=-1)
    right = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    down = torch.cross(forward, right, dim=-1)
    down = down / down.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    r_w2c = torch.stack((right, down, forward), dim=1)
    return _matrix_to_quaternion_wxyz(r_w2c.transpose(1, 2))


class gs_look_at_target_image_feature(gs_image_feature):
    """Dataset-matched GS observation for the no-out-of-view mid-occlusion route."""

    def __init__(self, cfg: ObservationTermCfg, env):
        ManagerTermBase.__init__(self, cfg, env)
        self.render_server_host = str(cfg.params.get("render_server_host", "localhost"))
        self.render_server_port = int(cfg.params.get("render_server_port", 18862))
        self.rgb_socket_host = str(cfg.params.get("rgb_socket_host", "localhost"))
        self.rgb_socket_port = int(cfg.params.get("rgb_socket_port", 12345))
        print(
            "[midocc_gs_image_feature] Connecting to render server on "
            f"{self.render_server_host}:{self.render_server_port}..."
        )
        self.conn = rpyc.connect(
            self.render_server_host,
            self.render_server_port,
            config={"allow_pickle": True, "allow_public_attrs": True},
        )
        print("[midocc_gs_image_feature] Render server connected.")
        self.image_server = GSServer(host=self.rgb_socket_host, port=self.rgb_socket_port)
        self.image_server.start()
        self.image_server.init_data(env.num_envs, h=180, w=320)
        print("[midocc_gs_image_feature] RGB socket receiver ready.")

        print("[midocc_gs_image_feature] Initializing mid-occ multitask occlusion predictor...")
        self.occlusion_predictor = MidOccMultiTaskOcclusionPredictor(
            model_path=str(cfg.params.get("multitask_model_path", DEFAULT_MULTITASK_MODEL_PATH)),
            project_root=cfg.params.get("multitask_project_root"),
            device=env.device,
        )
        self.occlusion_class_names = tuple(self.occlusion_predictor.occlusion_class_names)
        self.success_occlusion_class = str(cfg.params.get("success_occlusion_class", "0-20%"))
        self.success_occlusion_index = self.occlusion_predictor.class_index(self.success_occlusion_class)
        self.randomize_occlusion_prediction = bool(cfg.params.get("randomize_occlusion_prediction", False))
        self.output_dim = int(self.occlusion_predictor.feature_dim)
        self.history_len = max(1, int(cfg.params.get("history_len", 4)))
        self.feature_history = torch.zeros(
            (env.num_envs, self.history_len, self.output_dim),
            device=env.device,
            dtype=torch.float32,
        )

        self.camera_sampling_mode = str(cfg.params.get("camera_sampling_mode", "look_at_target")).lower()
        if self.camera_sampling_mode not in {"look_at_target", "world_xy_yaw"}:
            raise ValueError(
                f"Unsupported camera_sampling_mode: {self.camera_sampling_mode!r}. "
                "Expected 'look_at_target' or 'world_xy_yaw'."
            )
        camera_rot_default = cfg.params.get("camera_rot_deg", cfg.params.get("camera_rot", [0.0, 18.0, 0.0]))
        self.camera_rot_deg = torch.tensor(camera_rot_default, device=env.device, dtype=torch.float32)
        self.look_at_target_name = str(cfg.params.get("look_at_target_name", "red")).lower()
        self.look_at_target_offset = torch.tensor(
            cfg.params.get("look_at_target_offset", [0.0, 0.0, 0.1]),
            device=env.device,
            dtype=torch.float32,
        )
        self.look_at_yaw_mode = str(cfg.params.get("look_at_yaw_mode", "offset_orientation")).lower()
        if self.camera_sampling_mode == "look_at_target":
            print(
                "[midocc_gs_image_feature] Using dataset-matched look-at camera. "
                f"target={self.look_at_target_name} yaw_mode={self.look_at_yaw_mode} "
                f"feature_dim={self.output_dim}"
            )
        else:
            print(
                "[midocc_gs_image_feature] Using dataset-matched world_xy_yaw camera. "
                f"camera_rot_deg={self.camera_rot_deg.tolist()} feature_dim={self.output_dim}"
            )

        self.save_debug_images = bool(cfg.params.get("save_debug_images", False))
        self.save_every_n_steps = max(1, int(cfg.params.get("save_every_n_steps", 20)))
        self.save_max_images = max(1, int(cfg.params.get("save_max_images", 100)))
        self.save_env_index = int(cfg.params.get("save_env_index", 0))
        self._obs_step = 0
        self._saved_count = 0
        self.latest_images_np: np.ndarray | None = None
        self.save_dir = Path(cfg.params.get("save_dir", "logs/midocc_gs_render_debug"))
        if self.save_debug_images:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_debug_masks = bool(cfg.params.get("save_debug_masks", False))
        self.save_mask_every_n_steps = max(1, int(cfg.params.get("save_mask_every_n_steps", 1)))
        self.save_mask_max_images = int(cfg.params.get("save_mask_max_images", -1))
        self.save_mask_env_index = int(cfg.params.get("save_mask_env_index", self.save_env_index))
        self.mask_occluded_dir = Path(cfg.params.get("mask_occluded_dir", "logs/midocc_gs_mask_debug/occluded"))
        self.mask_target_only_dir = Path(
            cfg.params.get("mask_target_only_dir", "logs/midocc_gs_mask_debug/target_only")
        )
        self.mask_target_from_command = bool(cfg.params.get("mask_target_from_command", False))
        self.mask_command_name = str(cfg.params.get("mask_command_name", "rgb_command"))
        self.mask_target_default = str(cfg.params.get("mask_target_default", "red"))
        self.mask_threshold = float(cfg.params.get("mask_threshold", 0.5))
        self.mask_binary = bool(cfg.params.get("mask_binary", True))
        self._saved_mask_count = 0
        self._mask_api_warned = False

    @staticmethod
    def _strip_names(t: torch.Tensor) -> torch.Tensor:
        if isinstance(t, torch.Tensor) and getattr(t, "names", None) is not None:
            if any(n is not None for n in t.names):
                return t.rename(None)
        return t

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        camera_pos: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
        camera_rot: list[float] | tuple[float, float, float] | None = None,
        camera_rot_deg: list[float] | tuple[float, float, float] | None = None,
        camera_sampling_mode: str | None = None,
        asset_offset_pos: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
        look_at_target_name: str | None = None,
        look_at_target_offset: list[float] | tuple[float, float, float] | None = None,
        look_at_yaw_mode: str | None = None,
        render_server_host: str | None = None,
        render_server_port: int | None = None,
        rgb_socket_host: str | None = None,
        rgb_socket_port: int | None = None,
        history_len: int | None = None,
        save_debug_images: bool | None = None,
        save_every_n_steps: int | None = None,
        save_max_images: int | None = None,
        save_env_index: int | None = None,
        save_dir: str | None = None,
        multitask_model_path: str | None = None,
        multitask_project_root: str | None = None,
        success_occlusion_class: str | None = None,
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
    ) -> torch.Tensor:
        del render_server_host, render_server_port, rgb_socket_host, rgb_socket_port
        del multitask_model_path, multitask_project_root

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
                f"gs_look_at_target_image_feature was initialized with history_len={self.history_len}, "
                f"but received runtime history_len={int(history_len)}."
            )
        if camera_sampling_mode is not None:
            normalized_mode = str(camera_sampling_mode).lower()
            if normalized_mode not in {"look_at_target", "world_xy_yaw"}:
                raise ValueError(
                    f"Unsupported camera_sampling_mode: {normalized_mode!r}. "
                    "Expected 'look_at_target' or 'world_xy_yaw'."
                )
            self.camera_sampling_mode = normalized_mode
        if camera_rot_deg is None and camera_rot is not None:
            camera_rot_deg = camera_rot
        if camera_rot_deg is not None:
            self.camera_rot_deg = torch.tensor(camera_rot_deg, device=env.device, dtype=torch.float32)
        if look_at_target_name is not None:
            self.look_at_target_name = str(look_at_target_name).lower()
        if look_at_target_offset is not None:
            self.look_at_target_offset = torch.tensor(look_at_target_offset, device=env.device, dtype=torch.float32)
        if look_at_yaw_mode is not None:
            self.look_at_yaw_mode = str(look_at_yaw_mode).lower()

        robot: Articulation = env.scene[asset_cfg.name]
        pos_r = robot.data.root_pos_w - env.scene.env_origins
        quat_r = robot.data.root_quat_w

        offset = torch.tensor(asset_offset_pos, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
        camera_offset = torch.tensor(camera_pos, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)

        red: RigidObjectCollection = env.scene["cone_red"]
        green: RigidObjectCollection = env.scene["cone_green"]
        blue: RigidObjectCollection = env.scene["cone_blue"]
        red_pos = red.data.object_pos_w[:, 0, :] - env.scene.env_origins - offset
        green_pos = green.data.object_pos_w[:, 0, :] - env.scene.env_origins - offset
        blue_pos = blue.data.object_pos_w[:, 0, :] - env.scene.env_origins - offset

        if self.camera_sampling_mode == "look_at_target":
            cam_pos_w = pos_r + camera_offset - offset
            target_positions = {"red": red_pos, "green": green_pos, "blue": blue_pos}
            target_pos = target_positions.get(self.look_at_target_name, red_pos) + self.look_at_target_offset
            cam_quat_ros = _look_at_quaternion_for_render_server_torch(cam_pos_w, target_pos)

            if self.look_at_yaw_mode == "offset_orientation":
                rot_m = math_utils.matrix_from_quat(quat_r)
                yaw = torch.atan2(rot_m[:, 1, 0], rot_m[:, 0, 0])
                zeros = torch.zeros_like(yaw)
                yaw_quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
                cam_quat_ros = math_utils.quat_mul(yaw_quat, cam_quat_ros)
                cam_quat_ros = cam_quat_ros / cam_quat_ros.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            elif self.look_at_yaw_mode != "none":
                raise ValueError(f"Unsupported look_at_yaw_mode: {self.look_at_yaw_mode!r}")
        else:
            cam_pos_w = pos_r + math_utils.quat_apply(quat_r, camera_offset) - offset
            camera_rot_local = self.camera_rot_deg.unsqueeze(0).repeat(env.num_envs, 1)
            camera_quat_local = euler_to_quaternion(torch.deg2rad(camera_rot_local))
            cam_quat_w = math_utils.quat_mul(quat_r, camera_quat_local)
            cam_quat_ros = math_utils.convert_camera_frame_orientation_convention(cam_quat_w, "world", "ros")

        cam_pos_w = self._strip_names(cam_pos_w)
        cam_quat_ros = self._strip_names(cam_quat_ros)
        red_pos = self._strip_names(red_pos)
        green_pos = self._strip_names(green_pos)
        blue_pos = self._strip_names(blue_pos)

        self._warn_if_suspicious_inputs(env, pos_r, quat_r, cam_pos_w, cam_quat_ros, red_pos, green_pos, blue_pos)

        render_started = time.perf_counter()
        try:
            self.conn.root.render(cam_pos_w, cam_quat_ros, red_pos, green_pos, blue_pos)
        except Exception as exc:
            self._log_debug_context(
                "midocc_render_exception",
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
                "[midocc_gs_image_feature] slow render call "
                f"common_step={int(getattr(env, 'common_step_counter', -1))} "
                f"sim_step={int(getattr(env, '_sim_step_counter', -1))} "
                f"elapsed_s={render_elapsed:.3f}",
                flush=True,
            )

        try:
            images_np = self.image_server.get_data()
        except Exception as exc:
            self._log_debug_context("midocc_get_data_exception", env, error=str(exc), cam_pos_w=cam_pos_w)
            raise
        if not isinstance(images_np, np.ndarray) or images_np.shape != (env.num_envs, 3 * 180 * 320):
            self._log_debug_context("midocc_unexpected_image_buffer", env, images_np=images_np)
        self.latest_images_np = images_np.copy()
        self._maybe_save_debug_image(images_np)
        if self.save_debug_masks and not self._mask_api_warned:
            print("[midocc_gs_image_feature] save_debug_masks is ignored by the RGB-only render server.")
            self._mask_api_warned = True

        try:
            reward_images = torch.tensor(images_np, dtype=torch.float32, device=env.device).reshape(
                env.num_envs, 3, 180, 320
            )
        except Exception as exc:
            self._log_debug_context("midocc_reward_image_tensor_exception", env, error=str(exc), images_np=images_np)
            raise
        reward_images = reward_images / 255.0

        predict_started = time.perf_counter()
        try:
            occ_indices, occ_probs, features = self.occlusion_predictor.predict_with_features(reward_images)
        except Exception as exc:
            self._log_debug_context(
                "midocc_predict_with_features_exception",
                env,
                error=str(exc),
                reward_images=reward_images,
                images_np=images_np,
            )
            raise
        predict_elapsed = time.perf_counter() - predict_started
        if predict_elapsed > 5.0:
            print(
                "[midocc_gs_image_feature] slow predictor call "
                f"common_step={int(getattr(env, 'common_step_counter', -1))} "
                f"sim_step={int(getattr(env, '_sim_step_counter', -1))} "
                f"elapsed_s={predict_elapsed:.3f}",
                flush=True,
            )
        if self.randomize_occlusion_prediction:
            occ_indices, occ_probs = self._sample_random_occlusion_prediction(env)

        env.extras["pred_occ_class"] = occ_indices.to(env.device)
        env.extras["pred_occ_probs"] = occ_probs.to(env.device)
        env.extras["pred_occ_class_names"] = list(self.occlusion_class_names)
        env.extras["pred_occ_success_index"] = int(self.success_occlusion_index)
        env.extras["pred_occ_success_class"] = self.success_occlusion_class
        env.extras["pred_occ_success_mask"] = occ_indices == self.success_occlusion_index
        env.extras["pred_occ_success_rate"] = float((occ_indices == self.success_occlusion_index).float().mean().item())

        with torch.inference_mode():
            features = features.to(env.device)
            reset_mask = env.episode_length_buf == 0
            if reset_mask.any():
                self.feature_history[reset_mask] = features[reset_mask].unsqueeze(1).repeat(
                    1, self.history_len, 1
                )
            self.feature_history = torch.roll(self.feature_history, shifts=-1, dims=1)
            self.feature_history[:, -1, :] = features
            return self.feature_history.reshape(env.num_envs, self.history_len * self.output_dim)
