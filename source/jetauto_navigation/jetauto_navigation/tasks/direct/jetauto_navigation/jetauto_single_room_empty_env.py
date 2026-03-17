from __future__ import annotations

import math

import torch
import rpyc

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.utils.math as math_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
import isaacsim.core.utils.prims as prim_utils

from .custom_observations import ImageFeaturesNoHead
from .jetauto_single_room_env_cfg import JetautoSingleRoomEnvCfg


class JetautoSingleRoomEmptyEnv(DirectRLEnv):
    """Single-room environment with background only; walls/obstacle/target not spawned but logic preserved."""

    cfg: JetautoSingleRoomEnvCfg

    # alignment constants (real -> sim)
    ALIGN_SCALE = 0.3664808278887077
    ALIGN_ROT_XYZW = (0.04309, 0.03407, -0.02809, 0.99810)
    ALIGN_TRANSLATION = (0.089711, 0.740089, 0.192696)

    def __init__(self, cfg: JetautoSingleRoomEnvCfg, render_mode: str | None = None, **kwargs):

        import inspect, os
        # raise RuntimeError(f"[ENV-TRACE] Loaded env class from: {inspect.getfile(self.__class__)}")

        # path = inspect.getfile(self.__class__)
        # print("[ENV-REALPATH] class_file =", path, flush=True)
        # print("[ENV-REALPATH] realpath    =", os.path.realpath(path), flush=True)
        # print("[ENV-REALPATH] module      =", self.__class__.__module__, flush=True)

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
        self.curr_vis = torch.zeros(self.num_envs, device=self.device)       # 用于 rpyc ratio
        self.curr_vis_sem = torch.zeros(self.num_envs, device=self.device)  # 仅用于语义分割debug

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

        # Real->sim alignment (measured from box alignment results)
        self._align_scale = self.ALIGN_SCALE
        self._align_rot_xyzw = torch.tensor(self.ALIGN_ROT_XYZW, device=self.device)
        self._align_translation = torch.tensor(self.ALIGN_TRANSLATION, device=self.device)

        # Compute sim-space bounds for the real rectangle [(1.2,-0.6), (-0.6,-0.6), (1.2,1.8), (-0.6,1.8), z=0]
        sim_rect = torch.tensor(
            [
                [1.2, 1.1],
                [-1.0, 1.1],
                [1.2, 2.0],
                [-1.0, 2.0],
            ],
            device=self.device,
        )

        self._sim_rect = sim_rect

        # placeholders for logical target/obstacle positions (not spawned)
        self._target_pos_current = torch.zeros(self.num_envs, 3, device=self.device)
        self._obstacle_pos_current = torch.zeros(self.num_envs, 3, device=self.device)

        # --- ratio RPC client ---
        self._ratio_rpc = rpyc.connect("localhost", 18862, config={"allow_pickle": True})
        self._ratio_every = 1   # 每 8 step 更新一次（建议先大一点，避免太慢）
        self._ratio_step = 0
        print("[ENV] env file:", __file__, flush=True)

    def _real_to_sim(self, pts_real: torch.Tensor) -> torch.Tensor:
        """Apply the measured real->sim similarity transform to points."""
        quat_xyzw = self._align_rot_xyzw
        quat_wxyz = torch.stack((quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])).to(
            device=pts_real.device
        )
        quat_wxyz = quat_wxyz.expand(pts_real.shape[0], -1)
        rotated = math_utils.quat_apply(quat_wxyz, pts_real)
        return rotated * self._align_scale + self._align_translation

    def _setup_scene(self):
        """Build a single room with background only (no physical walls/targets/obstacles)."""

        self.robot_a = Articulation(self.cfg.robot_cfg.replace(prim_path="/World/envs/env_.*/Robot"))
        self.scene.articulations["robot_a"] = self.robot_a

        c = self.cfg.env_params.camera

        # Real->sim alignment (measured from box alignment results)y
        self._align_scale = self.ALIGN_SCALE
        self._align_rot_xyzw = torch.tensor(self.ALIGN_ROT_XYZW, device=self.device)
        self._align_translation = torch.tensor(self.ALIGN_TRANSLATION, device=self.device)

        # Background 3DGS
        bg_scale = self._align_scale
        bg_scale = self.ALIGN_SCALE
        # bg_scale = 0.23994040084788124
        bg_pos = tuple(self._align_translation.tolist())
        bg_pos = self.ALIGN_TRANSLATION
        # bg_pos = (-0.206631, 0.343036, 0.754697)
        bg_rot = tuple(self._align_rot_xyzw.tolist())
        bg_rot = self.ALIGN_ROT_XYZW
        # bg_rot = (-0.05688028042359017, -0.00610525184353289, -0.0018002532294618672, 0.9983607157171052)


        bg_usd_cfg = sim_utils.UsdFileCfg(
            # usd_path="/home/ubuntu/xc_isaac/jetauto_rl_navigation-main/source/jetauto_navigation/jetauto_navigation/tasks/direct/jetauto_navigation/source/corridor.usdz",
            usd_path="/home/ubuntu/xc_isaac/video_data_process/results_corridor/3dgs_output/point_cloud/iteration_30000/point_cloud.usdz",
            scale=(bg_scale, bg_scale, bg_scale),
        )
        sim_utils.spawn_from_usd(
            prim_path="/World/envs/env_0/Background",
            cfg=bg_usd_cfg,
            translation=bg_pos,
            orientation=(bg_rot[3], bg_rot[0], bg_rot[1], bg_rot[2]),
        )

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
                # rot=(0.0, 0.0, 0.60876, 0.79335),
                rot=(0.0, 0.0, 0.6820, 0.7314),  # TODO check the real camera pose on jetauto

                convention="parent",
            ),
        )
        self._camera_a = Camera(cam_cfg)
        self.scene.sensors["camera_a"] = self._camera_a

        ground_cfg = GroundPlaneCfg(color=None)
        spawn_ground_plane(prim_path="/World/ground", cfg=ground_cfg)
        # Hide the visual mesh to avoid affecting rendering while keeping the collision plane.
        if prim_utils.is_prim_path_valid("/World/ground/Environment"):
            prim_utils.set_prim_property("/World/ground/Environment", "visibility", "invisible")
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _lock_planar(self, robot, z_lock: float = 0.01) -> None:
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
        self._ratio_step += 1
        self._query_ratio_from_rpc()


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
        # self.curr_vis = visible_ratio
        self.curr_vis_sem = visible_ratio


        self._feat_hist = torch.roll(self._feat_hist, shifts=-1, dims=1)
        self._feat_hist[:, -1, :] = resnet_features
        obs_2048 = self._feat_hist.reshape(self.num_envs, -1)

        robot_xy = self.robot_a.data.root_pos_w[:, :2]
        robot_x = robot_xy[:, 0]
        robot_y = robot_xy[:, 1]

        # margin = self.cfg.env_params.walls.wall_margin
        min_xy = self._sim_rect.min(dim=0).values
        max_xy = self._sim_rect.max(dim=0).values
        min_x, min_y = min_xy
        max_x, max_y = max_xy
        inside_x = (robot_x >= min_x) & (robot_x <= max_x)
        inside_y = (robot_y >= min_y) & (robot_y <= max_y)
        wall_collision = ~(inside_x & inside_y)

        # obs_xy = self._obstacle_pos_current[:, :2].unsqueeze(1)
        # dist_to_obs = torch.norm(robot_xy.unsqueeze(1) - obs_xy, dim=-1)
        # obs_collision = (dist_to_obs < (self.cfg.env_params.obstacle.r + self.cfg.env_params.robot_r)).any(dim=1)
        #
        # target_xy = self._target_pos_current[:, :2]
        # dist_to_target = torch.norm(robot_xy - target_xy, dim=-1)
        # target_collision = dist_to_target < (self.cfg.env_params.target.r + self.cfg.env_params.robot_r)

        self.collision_mask = wall_collision

        return {
            "policy": obs_2048,
            "critic": obs_2048,
        }

    def _get_rewards(self) -> torch.Tensor:
        """
        Composite reward:
            O_t = occlusion ratio = 1 - visible_ratio
            ΔO  = O_{t-1} - O_t  (positive when occlusion decreases)
            P_t = +5 if O_t == 0 else -0.1
            R   = ΔO + P_t
        """

        # 1) current / previous occlusion ratio
        vis_t = self.curr_vis.clamp(0.0, 1.0)      # rpyc returned visible ratio
        vis_prev = self.prev_vis.clamp(0.0, 1.0)

        O_t = (1.0 - vis_t).clamp(0.0, 1.0)
        O_prev = (1.0 - vis_prev).clamp(0.0, 1.0)

        # 2) dense reward: occlusion decrease is positive
        delta_O = O_prev - O_t

        # 3) additional term P_t (success if visible ratio > 0.7)
        P_t = torch.where(vis_t > 0.7, torch.full_like(O_t, 5.0), torch.full_like(O_t, -0.01))

        # 4) collision penalty
        collision_penalty = torch.where(
            self.collision_mask,
            torch.full_like(O_t, -5.0),
            torch.zeros_like(O_t),
        )

        # 5) final reward
        reward = delta_O + P_t + collision_penalty

        # 6) update prev for next step (very important)
        self.prev_vis = self.curr_vis.clone()

        # 7) logging
        self.extras["vis_ratio"] = float(vis_t.mean().item())
        self.extras["occ_ratio"] = float(O_t.mean().item())
        self.extras["delta_O"] = float(delta_O.mean().item())
        self.extras["P_t"] = float(P_t.mean().item())
        self.extras["collision_penalty"] = float(collision_penalty.mean().item())
        self.extras["success"] = bool((vis_t > 0.7).any().item())

        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        curr_vis = self.curr_vis.clamp(0.0, 1.0)
        failed = self.collision_mask
        success = (curr_vis > 0.7) & (~failed)

        # terminated = success
        terminated = success | failed

        truncated = self.episode_length_buf >= self.max_episode_length - 1

        if success.any().item():
            print("Episode success!")
        if failed.any().item():
            print("Episode failed due to collision!")

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
            self._static_obstacle_pos = torch.tensor([0.1, -0.385, obstacle_h], device=self.device).repeat(
                self.num_envs, 1)
            self.cfg.env_params.obstacle.r = float(
                0.5 * math.sqrt(self.cfg.env_params.obstacle.size[0] ** 2 + self.cfg.env_params.obstacle.size[1] ** 2))
            self.cfg.env_params.target.r = 0.1
            self.cfg.env_params.robot_r = 0.2

        target_pos = self._static_target_pos[env_ids].clone()
        target_pos[:, :2] += torch.randn(len(env_ids), 2, device=self.device) * self.cfg.env_params.reset.target_noise

        obstacle_pos = self._static_obstacle_pos[env_ids].clone()
        obstacle_pos[:, :2] += torch.randn(len(env_ids), 2,
                                           device=self.device) * self.cfg.env_params.reset.obstacle_noise

        def sample_in_rect(num_envs, x1, x2, y1, y2, device=None, dtype=torch.float32):
            rx = torch.rand(num_envs, device=device, dtype=dtype) * (x2 - x1) + x1
            ry = torch.rand(num_envs, device=device, dtype=dtype) * (y2 - y1) + y1
            return torch.stack((rx, ry), dim=-1)

        min_xy = self._sim_rect.min(dim=0).values
        max_xy = self._sim_rect.max(dim=0).values
        min_x, min_y = min_xy
        max_x, max_y = max_xy

        robot_xy = sample_in_rect(
            num_envs=len(env_ids),
            x1=min_x,
            x2=max_x,
            y1=min_y,
            y2=max_y,
            device=self.device,
        )

        robot_pos = torch.cat([robot_xy, torch.zeros(len(env_ids), 1, device=self.device)], dim=-1)

        root_states_a = self.robot_a.data.default_root_state[env_ids].clone()
        root_states_a[:, :3] = self.scene.env_origins[env_ids] + robot_pos

        # x = robot_xy[:, 0]
        # y = robot_xy[:, 1]
        #
        # yaw = torch.atan2(-y, -x)  # 指向原点的朝向（z-up，只算yaw）
        # 然后四元数（w,x,y,z）
        yaw = (torch.rand(len(env_ids), device=self.device) - 0.5) * 2 * math.pi
        half = 0.5 * yaw
        quat = torch.stack([torch.cos(half),
                            torch.zeros_like(half),
                            torch.zeros_like(half),
                            torch.sin(half)], dim=-1)

        root_states_a[:, 3:7] = quat

        self.robot_a.write_root_pose_to_sim(root_states_a[:, :7], env_ids)
        self.robot_a.write_root_velocity_to_sim(root_states_a[:, 7:], env_ids)

        joint_pos_a = self.robot_a.data.default_joint_pos[env_ids].clone()
        joint_vel_a = torch.zeros_like(joint_pos_a)
        self.robot_a.set_joint_position_target(joint_pos_a, env_ids=env_ids)
        self.robot_a.write_joint_state_to_sim(joint_pos_a, joint_vel_a, env_ids=env_ids)

        # store logical positions for collision checks
        self._target_pos_current[env_ids] = target_pos
        self._obstacle_pos_current[env_ids] = obstacle_pos

        self.collision_mask[env_ids] = False
        self.curr_vis[env_ids] = 0.0
        self.prev_vis[env_ids] = 0.0
        self.curr_vis_sem[env_ids] = 0.0



    def _quat_xyzw_to_wxyz(self, q_xyzw: torch.Tensor) -> torch.Tensor:
        # q_xyzw: (...,4)
        return torch.stack((q_xyzw[..., 3], q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2]), dim=-1)

    def _quat_mul_wxyz(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        # Hamilton product, both wxyz
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack((
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ), dim=-1)

    def _query_ratio_from_rpc(self) -> None:
        cam_pos_w = getattr(self._camera_a.data, "pos_w", None)
        cam_quat_w = getattr(self._camera_a.data, "quat_w", None)  # usually wxyz

        if cam_pos_w is None or cam_quat_w is None:
            root_pos = self.robot_a.data.root_pos_w
            root_q_wxyz = self.robot_a.data.root_quat_w

            off_pos = torch.tensor([0.0, -0.1, 0.0], device=self.device).view(1, 3).repeat(self.num_envs, 1)
            off_q_xyzw = torch.tensor([0.0, 0.0, 0.6820, 0.7314], device=self.device).view(1, 4).repeat(self.num_envs, 1)
            off_q_wxyz = self._quat_xyzw_to_wxyz(off_q_xyzw)

            cam_pos_w = root_pos + math_utils.quat_apply(root_q_wxyz, off_pos)
            cam_quat_w = self._quat_mul_wxyz(root_q_wxyz, off_q_wxyz)

        # Service expects Isaac poses: [px, py, pz, qw, qx, qy, qz]
        E = cam_pos_w.shape[0]
        poses = torch.cat([cam_pos_w, cam_quat_w], dim=-1)  # (E,7) wxyz

        # print(f"[RPC] step={self._ratio_step} calling ratio for {E} envs", flush=True)
        poses_np = poses.detach().cpu().numpy()
        ratios_np = self._ratio_rpc.root.visible_ratio(poses_np)
        # print(f"[RPC] step={self._ratio_step} got ratios shape={ratios_np.shape}", flush=True)
        # self.curr_vis = torch.tensor(ratios_np, device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        ratios_list = ratios_np.tolist() if hasattr(ratios_np, "tolist") else ratios_np
        if not isinstance(ratios_list, (list, tuple)):
            ratios_list = [float(ratios_list)]
        self.curr_vis = torch.tensor(ratios_list, device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
