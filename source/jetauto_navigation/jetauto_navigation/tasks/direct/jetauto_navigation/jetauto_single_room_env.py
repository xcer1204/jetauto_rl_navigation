from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.utils.math as math_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_term_cfg import ObservationTermCfg

from .custom_observations import ImageFeaturesNoHead
from .jetauto_single_room_env_cfg import JetautoSingleRoomEnvCfg


class JetautoSingleRoomEnv(DirectRLEnv):
    """Single-room navigation environment with a cylindrical target and box obstacle."""

    cfg: JetautoSingleRoomEnvCfg

    def __init__(self, cfg: JetautoSingleRoomEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._hist_len = 4
        self._feat_hist = torch.zeros(self.num_envs, self._hist_len, 512, device=self.device)

        self.v_xy_max = self.cfg.env_params.motion.v_xy_max
        self.w_z_max = self.cfg.env_params.motion.w_z_max
        self.v_smooth = self.cfg.env_params.motion.v_smooth
        self._v_cmd_body = torch.zeros((self.num_envs, 3), device=self.device)
        self._w_cmd = torch.zeros((self.num_envs,), device=self.device)

        self.dof_idx, _ = self.robot_a.find_joints(self.cfg.dof_names)

        self.prev_vis = torch.zeros(self.num_envs, device=self.device)
        self.curr_vis = torch.zeros(self.num_envs, device=self.device)
        self.collision_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        obs_term_cfg = ObservationTermCfg(
            func=ImageFeaturesNoHead,
            params={
                "sensor_cfg": SceneEntityCfg("camera_a"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )

        self.resnet_extractor = ImageFeaturesNoHead(obs_term_cfg, env=self)

    def _setup_scene(self):
        """Build a single room with walls, cylindrical target, and box obstacle."""

        self.robot_a = Articulation(self.cfg.robot_cfg.replace(prim_path="/World/envs/env_.*/Robot"))
        self.scene.articulations["robot_a"] = self.robot_a

        w = self.cfg.env_params.walls
        t = self.cfg.env_params.target
        o = self.cfg.env_params.obstacle
        c = self.cfg.env_params.camera

        # Background 3DGS (visual only)
        bg_scale = 0.23994040084788124
        bg_pos = (-0.206631, 0.343036, 0.754697)
        # Quaternion (x, y, z, w) derived from provided transform matrix
        bg_rot = (-0.05688028042359017, -0.00610525184353289, -0.0018002532294618672, 0.9983607157171052)

        bg_usd_cfg = sim_utils.UsdFileCfg(
            usd_path="/home/ubuntu/xc_isaac/video_data_process/results_lab1211/3dgs_output/point_cloud/iteration_30000/lab.usdz",
            scale=(bg_scale, bg_scale, bg_scale),
        )
        # Spawn under env_0; the InteractiveScene clone will replicate to other envs.
        sim_utils.spawn_from_usd(
            prim_path="/World/envs/env_0/Background",
            cfg=bg_usd_cfg,
            translation=bg_pos,
            orientation=(bg_rot[3], bg_rot[0], bg_rot[1], bg_rot[2]),
        )

        wall_z = w.size[2] / 2
        half_x = w.size[0] * 0.5
        half_y = w.size_vert[1] * 0.5
        walls = [
            {"name": "Room_Wall_N", "pos": (0.0, half_y, wall_z), "size": w.size, "color": (0.1, 0.4, 0.9)},
            {"name": "Room_Wall_S", "pos": (0.0, -half_y, wall_z), "size": w.size, "color": (0.95, 0.9, 0.5)},
            {"name": "Room_Wall_E", "pos": (half_x, 0.0, wall_z), "size": w.size_vert, "color": (0.8, 0.4, 0.05)},
            {"name": "Room_Wall_W", "pos": (-half_x, 0.0, wall_z), "size": w.size_vert, "color": (0.55, 0.1, 0.8)},
        ]
        for wall in walls:
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{wall['name']}",
                spawn=sim_utils.CuboidCfg(
                    size=wall["size"],
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=wall["color"]),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=wall["pos"], rot=(0.0, 0.0, 0.0, 1.0)),
            )
            self.scene.rigid_objects[wall["name"].lower()] = RigidObject(cfg)

        target_radius = t.size[0] * 0.5
        target_half_h = t.size[2] * 0.5
        target_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Target",
            spawn=sim_utils.CylinderCfg(
                radius=target_radius,
                height=t.size[2],
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=t.color),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                semantic_tags=[("class", "target")],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.0, target_half_h), rot=(0.0, 0.0, 0.0, 1.0)),
        )
        self._target_a = RigidObject(target_cfg)
        self.scene.rigid_objects["target_a"] = self._target_a

        half_height = o.size[2] / 2
        obstacle_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Obstacle",
            spawn=sim_utils.CuboidCfg(
                size=o.size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=o.color),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                semantic_tags=[("class", "obstacle")],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, half_height), rot=(0.0, 0.0, 0.0, 1.0)),
        )
        self._obstacles = [RigidObject(obstacle_cfg)]
        self.scene.rigid_objects["obstacle_a"] = self._obstacles[0]

        cam_cfg = CameraCfg(
            prim_path="/World/envs/env_.*/Robot/base_footprint/visuals/depth_camera_link/Camera",
            update_period=0.0167,
            height=c.height,
            width=c.width,
            data_types=["rgb", "semantic_segmentation"],
            colorize_semantic_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, -0.1, 0.0),
                rot=(0.0, 0.0, 0.60876, 0.79335),
                convention="parent",
            ),
        )
        self._camera_a = Camera(cam_cfg)
        self.scene.sensors["camera_a"] = self._camera_a

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _lock_planar(self, robot, z_lock: float = 0.01) -> None:
        """Lock robot on a plane at z=z_lock with yaw only."""
        if z_lock is None:
            z_lock = self.cfg.env_params.motion.planar_z_lock
        pos = robot.data.root_pos_w.clone()
        pos[:, 2] = z_lock

        R = math_utils.matrix_from_quat(robot.data.root_quat_w)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])

        zeros = torch.zeros_like(yaw)
        quat_yaw = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
        robot.write_root_pose_to_sim(torch.cat([pos, quat_yaw], dim=-1))

    def _apply_action(self) -> None:
        a = self.actions.clamp(-1.0, 1.0)
        vx = a[:, 0] * self.v_xy_max
        vy = a[:, 1] * self.v_xy_max
        wz = a[:, 2] * self.w_z_max

        alpha = self.v_smooth
        v_new = torch.stack([vx, vy, torch.zeros_like(vx)], dim=-1)
        self._v_cmd_body = (1 - alpha) * self._v_cmd_body + alpha * v_new
        self._w_cmd = (1 - alpha) * self._w_cmd + alpha * wz

        v_world = math_utils.quat_apply(self.robot_a.data.root_quat_w, self._v_cmd_body)
        vel6 = torch.cat(
            [
                v_world[:, :2],
                torch.zeros_like(v_world[:, :1]),
                torch.zeros_like(self._w_cmd[:, None]),
                torch.zeros_like(self._w_cmd[:, None]),
                self._w_cmd[:, None],
            ],
            dim=-1,
        )
        self.robot_a.write_root_velocity_to_sim(vel6)
        self._lock_planar(self.robot_a, 0.01)

    def _get_observations(self) -> dict:
        with torch.no_grad():
            resnet_features = self.resnet_extractor(
                env=self,
                sensor_cfg=SceneEntityCfg("camera_a"),
                data_type="rgb",
                model_name="resnet18",
            )

        seg = self._camera_a.data.output["semantic_segmentation"][..., 0]

        target_ids = []
        has_target = []
        for info in self._camera_a.data.info:
            mapping = info["semantic_segmentation"]["idToLabels"]
            found = None
            for k, v in mapping.items():
                label = v.get("class", v) if isinstance(v, dict) else v
                if str(label).lower() == "target":
                    found = int(k)
                    break
            if found is None:
                target_ids.append(-1)
                has_target.append(False)
            else:
                target_ids.append(found)
                has_target.append(True)

        target_ids_tensor = torch.as_tensor(target_ids, device=seg.device, dtype=seg.dtype).view(-1, 1, 1)
        has_target_tensor = torch.as_tensor(has_target, device=seg.device, dtype=torch.bool).view(-1, 1, 1)

        mask = (seg == target_ids_tensor) & has_target_tensor
        visible_pixels = mask.sum(dim=(1, 2)).to(torch.float32)

        h, w = seg.shape[-2:]
        visible_ratio = torch.zeros_like(visible_pixels)
        valid_envs = has_target_tensor.view(-1)
        visible_ratio[valid_envs] = visible_pixels[valid_envs] / float(h * w + 1e-6)
        self.curr_vis = visible_ratio

        self._feat_hist = torch.roll(self._feat_hist, shifts=-1, dims=1)
        self._feat_hist[:, -1, :] = resnet_features
        obs_2048 = self._feat_hist.reshape(self.num_envs, -1)

        robot_xy = self.robot_a.data.root_pos_w[:, :2]

        robot_x = robot_xy[:, 0]
        robot_y = robot_xy[:, 1]

        w_cfg = self.cfg.env_params.walls
        half_x = w_cfg.size[0] * 0.5
        half_y = w_cfg.size_vert[1] * 0.5
        margin = self.cfg.env_params.walls.wall_margin
        inside_x = (robot_x >= -half_x + margin) & (robot_x <= half_x - margin)
        inside_y = (robot_y >= -half_y + margin) & (robot_y <= half_y - margin)
        wall_collision = ~(inside_x & inside_y)

        obs_xy = torch.stack([obs.data.root_pos_w[:, :2] for obs in self._obstacles], dim=1)
        dist_to_obs = torch.norm(robot_xy.unsqueeze(1) - obs_xy, dim=-1)
        obs_collision = (dist_to_obs < (self.cfg.env_params.obstacle.r + self.cfg.env_params.robot_r)).any(dim=1)

        target_xy = self._target_a.data.root_pos_w[:, :2]
        dist_to_target = torch.norm(robot_xy - target_xy, dim=-1)
        target_collision = dist_to_target < (self.cfg.env_params.target.r + self.cfg.env_params.robot_r)

        self.collision_mask = wall_collision | obs_collision | target_collision

        return {
            "policy": obs_2048,
            "critic": obs_2048,
        }

    def _get_rewards(self) -> torch.Tensor:
        self.prev_vis = self.curr_vis.clone()
        self.extras["curr_vis"] = float(self.curr_vis.mean().item())
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        curr_vis = self.curr_vis.clamp(0.0, 1.0)
        success = curr_vis >= 0.99

        failed = self.collision_mask
        terminated = success | failed

        truncated = self.episode_length_buf >= self.max_episode_length - 1

        self.extras["success"] = bool(success.any().item())
        self.extras["collision"] = bool(self.collision_mask.any().item())

        return terminated, truncated

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if not hasattr(self, "_static_target_pos"):
            target_h = self.cfg.env_params.target.size[2] * 0.5
            obstacle_h = self.cfg.env_params.obstacle.size[2] * 0.5
            self._static_target_pos = torch.tensor([0.0, 0.0, target_h], device=self.device).repeat(self.num_envs, 1)
            self._static_obstacle_pos = torch.tensor([0.1, -0.385, obstacle_h], device=self.device).repeat(self.num_envs, 1)
            self.cfg.env_params.obstacle.r = float(0.5 * math.sqrt(self.cfg.env_params.obstacle.size[0] ** 2 + self.cfg.env_params.obstacle.size[1] ** 2))
            self.cfg.env_params.target.r = 0.05
            self.cfg.env_params.robot_r = 0.05

        target_pos = self._static_target_pos[env_ids].clone()
        target_pos[:, :2] += torch.randn(len(env_ids), 2, device=self.device) * self.cfg.env_params.reset.target_noise

        obstacle_pos = self._static_obstacle_pos[env_ids].clone()
        obstacle_pos[:, :2] += torch.randn(len(env_ids), 2, device=self.device) * self.cfg.env_params.reset.obstacle_noise

        def sample_in_rect(num_envs, x1, x2, y1, y2, device=None, dtype=torch.float32):
            rx = torch.rand(num_envs, device=device, dtype=dtype) * (x2 - x1) + x1
            ry = torch.rand(num_envs, device=device, dtype=dtype) * (y2 - y1) + y1
            return torch.stack((rx, ry), dim=-1)

        robot_xy = sample_in_rect(
            num_envs=len(env_ids),
            x1=-1.0,
            x2=1.0,
            y1=-1.0,
            y2=-0.65,
            device=self.device,
        )

        robot_pos = torch.cat([robot_xy, torch.zeros(len(env_ids), 1, device=self.device)], dim=-1)

        root_states_a = self.robot_a.data.default_root_state[env_ids].clone()
        root_states_a[:, :3] = self.scene.env_origins[env_ids] + robot_pos

        yaw = (torch.rand(len(env_ids), device=self.device) - 0.5) * 2 * math.pi
        quat = math_utils.quat_from_angle_axis(yaw, torch.tensor([0.0, 0.0, 1.0], device=self.device))
        root_states_a[:, 3:7] = quat

        self.robot_a.write_root_pose_to_sim(root_states_a[:, :7], env_ids)
        self.robot_a.write_root_velocity_to_sim(root_states_a[:, 7:], env_ids)

        joint_pos_a = self.robot_a.data.default_joint_pos[env_ids].clone()
        joint_vel_a = torch.zeros_like(joint_pos_a)
        self.robot_a.set_joint_position_target(joint_pos_a, env_ids=env_ids)
        self.robot_a.write_joint_state_to_sim(joint_pos_a, joint_vel_a, env_ids=env_ids)

        target_state = self._target_a.data.default_root_state[env_ids].clone()
        obstacle_state = self._obstacles[0].data.default_root_state[env_ids].clone()

        target_state[:, 0:3] = self.scene.env_origins[env_ids] + target_pos
        obstacle_state[:, 0:3] = self.scene.env_origins[env_ids] + obstacle_pos

        self._target_a.write_root_pose_to_sim(target_state[:, :7], env_ids)
        self._target_a.write_root_velocity_to_sim(target_state[:, 7:], env_ids)
        self._obstacles[0].write_root_pose_to_sim(obstacle_state[:, :7], env_ids)
        self._obstacles[0].write_root_velocity_to_sim(obstacle_state[:, 7:], env_ids)

        self.collision_mask[env_ids] = False
        self.curr_vis[env_ids] = 0.0
        self.prev_vis[env_ids] = 0.0

        # Visual feature extractor is stateless; nothing to reset.
