# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from sympy.diffgeom.rn import theta
from transformers.models.xcodec.modeling_xcodec import SemanticEncoder

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg, TiledCameraCfg
from isaaclab.managers import SceneEntityCfg

from .custom_observations import ImageFeaturesNoHead
from isaaclab.envs.mdp import * 
from isaaclab.managers.manager_term_cfg import ObservationTermCfg

from .jetauto_navigation_env_cfg import JetautoNavigationEnvCfg


class JetautoNavigationEnv(DirectRLEnv):
    cfg: JetautoNavigationEnvCfg

    def __init__(self, cfg: JetautoNavigationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # self.cfg.env_params = self.cfg.env_params

        self._hist_len = 4
        self._feat_hist = torch.zeros(self.num_envs, self._hist_len, 512, device=self.device)

        self.v_xy_max = self.cfg.env_params.motion.v_xy_max  # m/s
        self.w_z_max = self.cfg.env_params.motion.w_z_max  # rad/s
        self.v_smooth = self.cfg.env_params.motion.v_smooth  # [0~1] 越大越稳
        self._v_cmd_body = torch.zeros((self.num_envs, 3), device=self.device)  # [vx,vy,0]
        self._w_cmd = torch.zeros((self.num_envs,), device=self.device)  # wz

        self.dof_idx, _ = self.robot_a.find_joints(self.cfg.dof_names)

        self.prev_vis = torch.zeros(self.num_envs, device=self.device) #缓存上一帧的可见率
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
        # feat = ImageFeaturesNoHead(obs_term_cfg, env=self)
        # print(feat._model)
        # 手动初始化 ResNet18 特征提取模块
        # init constants
        self._const_success = torch.tensor(20.0, device=self.device)
        self._const_zero = torch.tensor(0.0, device=self.device)
        self._const_penalty = torch.tensor(-1.0, device=self.device)
        print("Torch mem:", torch.cuda.memory_allocated()/1024**2, "MB")


    def _setup_scene(self):
        """构建场景：房间A(机器人+目标+障碍物+围墙) + 房间B(机器人+目标+围墙)。"""

        self.robot_a = Articulation(self.cfg.robot_cfg.replace(prim_path="/World/envs/env_.*/Robot_A"))
        self.scene.articulations["robot_a"] = self.robot_a
        # print("robot A bbox size:", (self.robot_a.data.bounding_box_max - self.robot_a.data.bounding_box_min)[0])

        # === 读配置 ===
        w = self.cfg.env_params.walls
        t = self.cfg.env_params.target
        o = self.cfg.env_params.obstacle
        c = self.cfg.env_params.camera
        offset_x = self.cfg.env_params.reset.room_b_offset_x
        # 房间 A 围墙
        # wall_size = (3.0, 0.1, 0.4)
        # wall_size_vert = (0.1, 3.0, 0.4)
        # wall_z = wall_size[2] / 2
        wall_z = w.size[2] / 2
        walls_a = [
            {"name": "RoomA_Wall_N", "pos": (0.0, 1.5, wall_z), "size": w.size,       "color": (0.1, 0.4, 0.9)},   # 蓝
            {"name": "RoomA_Wall_S", "pos": (0.0, -1.5, wall_z), "size": w.size,      "color": (0.95, 0.9, 0.5)},  # 浅黄
            {"name": "RoomA_Wall_E", "pos": (1.5, 0.0, wall_z), "size": w.size_vert,  "color": (0.8, 0.4, 0.05)},  # 深橙
            {"name": "RoomA_Wall_W", "pos": (-1.5, 0.0, wall_z), "size": w.size_vert, "color": (0.55, 0.1, 0.8)},  # 紫
        ]
        for wall in walls_a:
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{wall['name']}",
                spawn=sim_utils.CuboidCfg(
                    size=wall["size"],
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=wall["color"]),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg()
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=wall["pos"], rot=(0.0, 0.0, 0.0, 1.0)
                )
            )
            self.scene.rigid_objects[wall['name'].lower()] = RigidObject(cfg)


        # if not hasattr(self, "_static_target_pos"):
        #     self.target_size = (0.2, 0.2, 0.2)
        #     self.obstacle_size = (0.1, 0.1, 0.1)

        # 房间 A 目标
        target_half_h = t.size[2] / 2
        target_cfg_a = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Target_A",
            spawn=sim_utils.CuboidCfg(
                size=t.size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                semantic_tags=[("class", "target")]
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.8, 0.0, target_half_h), rot=(0.0, 0.0, 0.0, 1.0)
            )
        )
        self._target_a = RigidObject(target_cfg_a)
        self.scene.rigid_objects["target_a"] = self._target_a


        # 房间 A 障碍物
        # obstacle_size = (0.1, 0.1, 0.1)
        half_height = o.size[2] / 2
        obstacle_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Obstacle",
            spawn=sim_utils.CuboidCfg(
                size=o.size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                semantic_tags=[("class", "obstacle")]
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.4, 0.0, half_height), rot=(0.0, 0.0, 0.0, 1.0)
            )
        )
        self._obstacles = [RigidObject(obstacle_cfg)]
        self.scene.rigid_objects["obstacle_a"] = self._obstacles[0]

        # 相机 A
        cam_cfg_a = CameraCfg(
            # prim_path="/World/envs/env_.*/Robot_A/base_link/visuals/Kaya_Body_Collapsed/kaya_camera",
            prim_path="/World/envs/env_.*/Robot_A/base_footprint/visuals/depth_camera_link/Camera",
            update_period=0.0167,
            height=c.height, width=c.width,
            # height=320, width=320,
            data_types=["rgb","semantic_segmentation"],
            colorize_semantic_segmentation=False,
            spawn = sim_utils.PinholeCameraCfg(),
            # spawn = None,
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, -0.1, 0.0),
                # rot=(0.0, 0.0, 0.7071, 0.7071),
                rot=(0.0, 0.0, 0.60876, 0.79335),
                convention="parent"
            )
        )
        self._camera_a = Camera(cam_cfg_a)
        self.scene.sensors["camera_a"] = self._camera_a

        # ---------------- 房间 B: 向 X 轴偏移 3 米 ----------------
        # offset_x = 3.0

        # 机器人 B
        self.robot_b = Articulation(self.cfg.robot_cfg.replace(prim_path="/World/envs/env_.*/Robot_B"))
        self.scene.articulations["robot_b"] = self.robot_b

        # 房间 B 围墙
        walls_b = [
            {"name": "RoomB_Wall_N", "pos": (offset_x, 1.5, wall_z), "size": w.size},
            {"name": "RoomB_Wall_S", "pos": (offset_x, -1.5, wall_z), "size": w.size},
            {"name": "RoomB_Wall_E", "pos": (offset_x + 1.5, 0.0, wall_z), "size": w.size_vert},
            {"name": "RoomB_Wall_W", "pos": (offset_x - 1.5, 0.0, wall_z), "size": w.size_vert},
        ]
        for wall in walls_b:
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{wall['name']}",
                spawn=sim_utils.CuboidCfg(
                    size=wall["size"],
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.8)),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg()
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=wall["pos"], rot=(0.0, 0.0, 0.0, 1.0)
                )
            )
            self.scene.rigid_objects[wall['name'].lower()] = RigidObject(cfg)

        # 房间 B 目标
        target_cfg_b = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Target_B",
            spawn=sim_utils.CuboidCfg(
                size=t.size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                semantic_tags=[("class", "target")]
                # semantic_tags=SemanticTagsCfg(class_name="target")
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(offset_x + 0.8, 0.0, target_half_h), rot=(0.0, 0.0, 0.0, 1.0)
            )
        )
        self._target_b = RigidObject(target_cfg_b)
        self.scene.rigid_objects["target_b"] = self._target_b


        # 相机 B
        cam_cfg_b = CameraCfg(
            # prim_path="/World/envs/env_.*/Robot_B/base_link/visuals/Kaya_Body_Collapsed/kaya_camera",
            prim_path="/World/envs/env_.*/Robot_B/base_footprint/visuals/depth_camera_link/Camera",
            update_period=0.0167,
            height=c.height, width=c.width,
            # height=320, width=320,
            data_types=["rgb", "semantic_segmentation"],
            colorize_semantic_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(),
            # spawn = None,
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, -0.1, 0.0),
                # rot=(0.0, 0.0, 0.7071, 0.7071),
                rot=(0.0, 0.0, 0.60876, 0.79335),
                convention="parent"
            )
        )
        self._camera_b = Camera(cam_cfg_b)
        self.scene.sensors["camera_b"] = self._camera_b

        # ---------------- 公共部分 ----------------
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    # def _apply_action(self) -> None:
    #     self.robot_a.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)
    #     self.robot_b.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    # 放在类里，与 _apply_action 平级
    def _lock_planar(self, robot, z_lock: float = 0.09105) -> None:
        """锁定在平面：z=z_lock，去掉 pitch/roll，仅保留 yaw。"""
        if z_lock is None:
            z_lock = self.cfg.env_params.motion.planar_z_lock
        # 改高度
        pos = robot.data.root_pos_w.clone()  # (N,3)
        pos[:, 2] = z_lock

        # quat -> yaw（你的版本没有 yaw_from_quat，用矩阵最稳）
        R = math_utils.matrix_from_quat(robot.data.root_quat_w)  # (N,3,3)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])  # (N,)

        zeros = torch.zeros_like(yaw)
        quat_yaw = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)  # (N,4)

        # 写回 7 维根姿态（[x,y,z,qx,qy,qz,qw]）
        robot.write_root_pose_to_sim(torch.cat([pos, quat_yaw], dim=-1))


    # 覆盖 _apply_action
    def _apply_action(self) -> None:
        a = self.actions.clamp(-1.0, 1.0)
        vx = a[:, 0] * self.v_xy_max
        vy = a[:, 1] * self.v_xy_max
        wz = a[:, 2] * self.w_z_max

        # 指数平滑（更稳）
        alpha = self.v_smooth
        v_new = torch.stack([vx, vy, torch.zeros_like(vx)], dim=-1)
        self._v_cmd_body = (1 - alpha) * self._v_cmd_body + alpha * v_new  # (N,3)
        self._w_cmd = (1 - alpha) * self._w_cmd + alpha * wz  # (N,)

        # 机体系 -> 世界系，然后写根速度
        def _apply(robot):
            v_world = math_utils.quat_apply(robot.data.root_quat_w, self._v_cmd_body)  # (N,3)
            vel6 = torch.cat(
                [v_world[:, :2], torch.zeros_like(v_world[:, :1]),
                 torch.zeros_like(self._w_cmd[:, None]), torch.zeros_like(self._w_cmd[:, None]),
                 self._w_cmd[:, None]],
                dim=-1
            )  # (N,6) = [Vx,Vy,Vz,Wx,Wy,Wz]
            robot.write_root_velocity_to_sim(vel6)

        _apply(self.robot_a)
        _apply(self.robot_b)

        # 仅在最后做一次平面锁定
        self._lock_planar(self.robot_a,0.01)
        # self._lock_planar(self.robot_a, 0.01)
        # self._lock_planar(self.robot_a, 0.09105)
        self._lock_planar(self.robot_b,0.01)
        # self._lock_planar(self.robot_b, 0.01)
        # self._lock_planar(self.robot_b, 0.09105)

    def _get_observations(self) -> dict:

        # rgb_a = self._camera_a.data.output["rgb"]   #([num_envs, 320, 320, 3])
        # rgb_b = self._camera_b.data.output["rgb"]

        # 语义分割输出（取单通道 ID 图）
        seg_a = self._camera_a.data.output["semantic_segmentation"][..., 0]  # [N,H,W]
        seg_b = self._camera_b.data.output["semantic_segmentation"][..., 0]  # [N,H,W]

        # ---- 安全地为“每个env”查找 target 的语义ID（可能不存在）----
        # 有的相机会把 info["semantic_segmentation"]["idToLabels"] 的 v 写成 dict，有的直接是字符串
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
                target_ids.append(-1)          # -1 作为“无目标”的占位ID，seg不会等于-1
                has_target.append(False)
            else:
                target_ids.append(found)
                has_target.append(True)

        target_ids_tensor = torch.as_tensor(target_ids, device=seg_a.device, dtype=seg_a.dtype).view(-1, 1, 1)
        has_target_tensor = torch.as_tensor(has_target, device=seg_a.device, dtype=torch.bool).view(-1, 1, 1)

        # 只有当该env真的找到了 target_id 时才启用比较；否则整帧 mask 为 False
        mask_a = (seg_a == target_ids_tensor) & has_target_tensor
        mask_b = (seg_b == target_ids_tensor) & has_target_tensor

        visible_pixels_a = mask_a.sum(dim=(1, 2)).to(torch.float32)  # [N]
        visible_pixels_b = mask_b.sum(dim=(1, 2)).to(torch.float32)  # [N]

        has_target_b = visible_pixels_b > 0

        is_on_edge_b = self._mask_touches_edge(mask_b, border_width=5)

        visible_ratio = torch.zeros_like(visible_pixels_a)

        # only when has target in view & not touching edge, we compute visible ratio
        valid_envs = has_target_b & (~is_on_edge_b)
        visible_ratio[valid_envs] = (
            visible_pixels_a[valid_envs] /
            (visible_pixels_b[valid_envs] + 1e-6)
        )

        self.curr_vis = visible_ratio

        # ---- ResNet 特征 ----
        with torch.no_grad():
            resnet_features = self.resnet_extractor(
                env=self,
                sensor_cfg=SceneEntityCfg("camera_a"),
                data_type="rgb",
                model_name="resnet18",
            )

        # 历史缓存（右移一格，把新帧放最后一格）
        self._feat_hist = torch.roll(self._feat_hist, shifts=-1, dims=1)
        self._feat_hist[:, -1, :] = resnet_features

        obs_2048 = self._feat_hist.reshape(self.num_envs, -1)  # [N, 2048]

        # ---- 碰撞检测 ----
        robot_xy = self.robot_a.data.root_pos_w[:, :2]
        env_xy = self.scene.env_origins[:, :2]
        robot_local = robot_xy - env_xy

        # print("Robot local XYZ:", self.robot_a.data.root_pos_w[:, :3]-self.scene.env_origins[:, :3])

        room_half = 1.5  #TODO change to cfg
        wall_margin = self.cfg.env_params.walls.wall_margin
        wall_collision = (torch.abs(robot_local) > (room_half - wall_margin)).any(dim=-1)


        # 所有障碍物在 self._obstacles 列表中
        obs_xy = torch.stack([obs.data.root_pos_w[:, :2] for obs in self._obstacles], dim=1)  # [num_envs, num_obs, 2]
        dist_to_obs = torch.norm(robot_xy.unsqueeze(1) - obs_xy, dim=-1)  # [num_envs, num_obs]
        obs_collision = (dist_to_obs < (self.cfg.env_params.obstacle.r + self.cfg.env_params.robot_r)).any(dim=1)

        target_xy = self._target_a.data.root_pos_w[:, :2]
        dist_to_target = torch.norm(robot_xy - target_xy, dim=-1)
        target_collision = dist_to_target < (self.cfg.env_params.target.r + self.cfg.env_params.robot_r)

        # if wall_collision.sum().item() > 0:
        #     print("Wall collision detected!")
        # if obs_collision.sum().item() > 0:
        #     print("Obstacle collision detected!")
        
        # print("Wall collisions:", wall_collision.sum().item(), "Obstacle collisions:", obs_collision.sum().item())

        self.collision_mask = wall_collision | obs_collision | target_collision

        # return {"policy": resnet_features, "critic": resnet_features, "rgb":rgb_a}
        return {
            "policy": obs_2048,
            "critic": obs_2048,
            # "rgb": rgb_a  # 如需调试可保留，但 skrl 只会取 policy/critic
        }

    def _mask_touches_edge(self, mask, border_width=None):
        """判断掩码是否触碰到图像边缘"""
        if border_width is None:
            border_width = int(self.cfg.env_params.reset.mask_edge_width)
        h, w = mask.shape[-2:]
        top = mask[..., :border_width, :].any(dim=(-2, -1))
        bottom = mask[..., -border_width:, :].any(dim=(-2, -1))
        left = mask[..., :, :border_width].any(dim=(-2, -1))
        right = mask[..., :, -border_width:].any(dim=(-2, -1))
        return top | bottom | left | right


    def _get_rewards(self) -> torch.Tensor:
        """
        奖励函数：
        - 鼓励提高目标的可见比例
        - 到达无遮挡状态给高奖励
        - 轻微时间步惩罚防止磨叽
        """

        # 避免首帧产生虚假奖励：仅在每个 episode 的第一步执行一次
        first_step = self.episode_length_buf == 0
        self.prev_vis[first_step] = self.curr_vis[first_step]
        # print('Current Visibility:', self.curr_vis)

        # 可见率 self.curr_vis always in [0,1]
        vis_gain = torch.clamp(self.curr_vis - self.prev_vis, -1.0, 1.0)  # 可见率变化量

        # 稀疏奖励：完全无遮挡（成功）
        success = self.curr_vis >= 0.99
        success_reward = self._const_success * success
        # 稀疏惩罚：撞墙或撞障碍
        collision_penalty = self._const_penalty * self.collision_mask

        # 复合dense奖励
        alpha_1, alpha_2, step_penalty = 0.1, 0.05, 0.01
        dense_reward = alpha_1 * vis_gain + alpha_2 * self.curr_vis - step_penalty

        # dense_reward = -step_penalty
        # dense_reward =  alpha_2 * self.curr_vis - step_penalty

        reward = dense_reward + success_reward + collision_penalty

        # 记录奖励组成部分，方便调试
        # self.extras["reward_vis_gain"] = (alpha_1 * vis_gain).mean()
        # self.extras["reward_vis_abs"] = (alpha_2 * self.curr_vis).mean()
        # self.extras["reward_collision"] = (self._const_penalty * self.collision_mask).mean()

        # 记录上一帧可见度
        self.prev_vis = self.curr_vis.clone()

        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Episode 结束逻辑：
        - 成功：目标完全可见（可见率 >= 0.99）
        - 失败：撞墙或撞障碍
        - 截断：达到最大步数
        """
        curr_vis = self.curr_vis.clamp(0.0, 1.0)
        success = curr_vis >= 0.99

        failed = self.collision_mask
        terminated = success | failed

        # 截断（超时）
        truncated = self.episode_length_buf >= self.max_episode_length - 1


        if truncated.sum().item() > 0:
            print("Episode truncated!")
        if success.sum().item() > 0:
            print("Episode success!")
        if failed.sum().item() > 0:
            print("Episode failed due to collision!")
        

        # print("Terminated:", terminated.sum().item(), "Success:", success.sum().item(), "Failed:", failed.sum().item(), "Truncated:", truncated.sum().item())


        return terminated, truncated



    def _reset_idx(self, env_ids: Sequence[int] | None = None):

        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        

        # jetauto目测长34厘米 宽22厘米
        # 机器人中心到四个角 = (0.17^2 + 0.11^2) ^ 0.5  # 约20cm

        if not hasattr(self, "_static_target_pos"):
            # 只在第一次初始化时设置
            self._static_target_pos = torch.tensor([1.0, 0.0, self.cfg.env_params.target.size[2]*0.5], device=self.device).repeat(self.num_envs, 1)
            self._static_obstacle_pos = torch.tensor([0.0, 0.0, self.cfg.env_params.obstacle.size[2]*0.5], device=self.device).repeat(self.num_envs, 1)
            self.cfg.env_params.obstacle.r = float(0.5 * math.sqrt(self.cfg.env_params.obstacle.size[0] ** 2 + self.cfg.env_params.obstacle.size[1] ** 2))
            self.cfg.env_params.target.r = float(0.5 * math.sqrt(self.cfg.env_params.target.size[0] ** 2 + self.cfg.env_params.target.size[1] ** 2))
            self.cfg.env_params.robot_r = 0.2  # 20cm 半径

        # 1) 给target obstacle (x,y) 微小扰动（5cm 量级）

        target_pos = self._static_target_pos[env_ids].clone()
        target_pos[:, :2] += torch.randn(len(env_ids), 2, device=self.device) * self.cfg.env_params.reset.target_noise

        obstacle_pos = self._static_obstacle_pos[env_ids].clone()
        obstacle_pos[:, :2] += torch.randn(len(env_ids), 2, device=self.device) * self.cfg.env_params.reset.obstacle_noise

        target_xy = target_pos[:, :2]
        obstacle_xy = obstacle_pos[:, :2]

        # 2) 从配置读房间长宽 / 半径等
        room_size = torch.tensor([3.0, 3.0], device=self.device)    #TODO change to cfg

        # 3) 九宫格格子索引（可配置）
        # 格子编号规则（ix,iy）：
        # ↑ y
        # |
        # |
        # 6 | 7 | 8    ← top row (iy = 2)
        # ---+---+---
        # 3 | 4 | 5    ← middle row (iy = 1)
        # ---+---+---
        # 0 | 1 | 2    ← bottom row (iy = 0)
        # |
        # +--------→ x

        tgt_cell = int(5)
        obs_cell = int(4)

        # 4) 批量采样机器人 XY（不需要 env_ids 内容本身，只需要 num_envs）
        robot_xy = sample_safe_rect_positions_grid_torch(
            num_envs=len(env_ids),
            target_cell_idx=tgt_cell,
            obstacle_cell_idx=obs_cell,
            room_size=room_size,
            robot_radius= self.cfg.env_params.robot_r,
            target_radius=self.cfg.env_params.obstacle.r,
            obstacle_radius=self.cfg.env_params.target.r,
            target_xy=target_xy,
            obstacle_xy=obstacle_xy,
            margin=0.05,
            max_tries=10,
        )

        # 5) 组装 root_states 写回（用 env_ids 进行索引）
        robot_pos = torch.cat([robot_xy, torch.zeros(len(env_ids), 1, device=self.device)], dim=-1)

        root_states_a = self.robot_a.data.default_root_state[env_ids].clone()
        root_states_b = self.robot_b.data.default_root_state[env_ids].clone()

        root_states_a[:, :3] = self.scene.env_origins[env_ids] + robot_pos
        root_states_b[:, :3] = root_states_a[:, :3].clone()
        root_states_b[:, 0] += 3.0  # 房间B偏移

        # 随机朝向（yaw）
        yaw = (torch.rand(len(env_ids), device=self.device) - 0.5) * 2 * math.pi
        quat = math_utils.quat_from_angle_axis(yaw, torch.tensor([0.0, 0.0, 1.0], device=self.device))

        root_states_a[:, 3:7] = quat
        root_states_b[:, 3:7] = quat

        self.robot_a.write_root_pose_to_sim(root_states_a[:, :7], env_ids)
        self.robot_a.write_root_velocity_to_sim(root_states_a[:, 7:], env_ids)
        self.robot_b.write_root_pose_to_sim(root_states_b[:, :7], env_ids)
        self.robot_b.write_root_velocity_to_sim(root_states_b[:, 7:], env_ids)

        # 重置机器人的关节（轮子）位置和速度
        joint_pos_a = self.robot_a.data.default_joint_pos[env_ids].clone()
        joint_vel_a = torch.zeros_like(joint_pos_a)
        self.robot_a.set_joint_position_target(joint_pos_a, env_ids=env_ids)
        self.robot_a.write_joint_state_to_sim(joint_pos_a, joint_vel_a, env_ids=env_ids)

        joint_pos_b = self.robot_b.data.default_joint_pos[env_ids].clone()
        joint_vel_b = torch.zeros_like(joint_pos_b)
        self.robot_b.set_joint_position_target(joint_pos_b, env_ids=env_ids)
        self.robot_b.write_joint_state_to_sim(joint_pos_b, joint_vel_b, env_ids=env_ids)

        # --------------------
        # 写入 target / obstacle
        # --------------------
        target_state = self._target_a.data.default_root_state[env_ids].clone()
        obstacle_state = self._obstacles[0].data.default_root_state[env_ids].clone()

        target_state[:, 0:3] = self.scene.env_origins[env_ids] + target_pos
        obstacle_state[:, 0:3] = self.scene.env_origins[env_ids] + obstacle_pos

        self._target_a.write_root_pose_to_sim(target_state[:, :7], env_ids)
        self._target_a.write_root_velocity_to_sim(target_state[:, 7:], env_ids)
        self._obstacles[0].write_root_pose_to_sim(obstacle_state[:, :7], env_ids)
        self._obstacles[0].write_root_velocity_to_sim(obstacle_state[:, 7:], env_ids)

        # B房间同步
        target_state_b = target_state.clone()
        target_state_b[:, 0] += self.cfg.env_params.reset.room_b_offset_x
        self._target_b.write_root_pose_to_sim(target_state_b[:, :7], env_ids)
        self._target_b.write_root_velocity_to_sim(target_state_b[:, 7:], env_ids)

        # --------------------
        # 清零状态
        # --------------------
        self.collision_mask[env_ids] = False
        self.curr_vis[env_ids] = 0.0
        self.prev_vis[env_ids] = 0.0






@torch.jit.script
def sample_safe_rect_positions_grid_torch(
    num_envs: int,                               # 本次要重置的环境个数 (len(env_ids))
    target_cell_idx: int,                 # 0..8
    obstacle_cell_idx: int,               # 0..8
    room_size: torch.Tensor,             # [2] -> (L, W)
    robot_radius: float,
    target_radius: float,
    obstacle_radius: float,
    target_xy: torch.Tensor,              # [M, 2] 每个 env 的目标中心 (x,y)
    obstacle_xy: torch.Tensor,            # [M, 2] 每个 env 的障碍中心 (x,y)
    margin: float = 0.05,
    max_tries: int = 10,
) -> torch.Tensor:
    
    """
    3x3 九宫格采样（长方形房间；只对贴墙侧内缩；基于半径的碰撞判定）。
    - 仅当格子的某一侧恰好是房间外墙时，才在该侧内缩 (robot_radius + margin)。
    - 中央格不缩，边格缩一侧，角格缩两侧。
    - 如果整体房间在任何轴向上都放不下机器人：抛 ValueError。
    - 与目标/障碍的碰撞：dist > (r_robot + r_obj + margin)。
    """

    # 基本量
    L, W = room_size[0], room_size[1]
    half_L = L * 0.5
    half_W = W * 0.5
    pad = robot_radius + margin

    # 房间过小：抛异常（TorchScript 允许 RuntimeError）
    if (L <= 2.0 * pad) or (W <= 2.0 * pad):
        raise RuntimeError("Room too small for robot size + margin.")

    # 3x3 网格边界（形状 [4]）
    x_edges = torch.linspace(-half_L, half_L, 4, device=room_size.device, dtype=room_size.dtype)
    y_edges = torch.linspace(-half_W, half_W, 4, device=room_size.device, dtype=room_size.dtype)

    # 原始 cell 边界：cell_bounds[9,4] = [x1,x2,y1,y2]
    cell_bounds = torch.empty((9, 4), device=room_size.device, dtype=room_size.dtype)
    # 行优先: 0 1 2 / 3 4 5 / 6 7 8
    # ix = 0,1,2 (列) ; iy = 0,1,2 (行, 0=下,2=上)
    idx = 0
    for iy in range(3):
        for ix in range(3):
            cell_bounds[idx, 0] = x_edges[ix]
            cell_bounds[idx, 1] = x_edges[ix + 1]
            cell_bounds[idx, 2] = y_edges[iy]
            cell_bounds[idx, 3] = y_edges[iy + 1]
            idx += 1

    # 只对贴墙侧内缩 pad
    ix = torch.arange(9, device=room_size.device) % 3
    iy = torch.arange(9, device=room_size.device) // 3

    # shrink 左/右/下/上侧
    # 左墙: ix==0 → x1+=pad
    cell_bounds[:, 0] = cell_bounds[:, 0] + torch.where(ix == 0, torch.as_tensor(pad, device=room_size.device, dtype=room_size.dtype), torch.as_tensor(0.0, device=room_size.device, dtype=room_size.dtype))
    # 右墙: ix==2 → x2-=pad
    cell_bounds[:, 1] = cell_bounds[:, 1] - torch.where(ix == 2, torch.as_tensor(pad, device=room_size.device, dtype=room_size.dtype), torch.as_tensor(0.0, device=room_size.device, dtype=room_size.dtype))
    # 下墙: iy==0 → y1+=pad
    cell_bounds[:, 2] = cell_bounds[:, 2] + torch.where(iy == 0, torch.as_tensor(pad, device=room_size.device, dtype=room_size.dtype), torch.as_tensor(0.0, device=room_size.device, dtype=room_size.dtype))
    # 上墙: iy==2 → y2-=pad
    cell_bounds[:, 3] = cell_bounds[:, 3] - torch.where(iy == 2, torch.as_tensor(pad, device=room_size.device, dtype=room_size.dtype), torch.as_tensor(0.0, device=room_size.device, dtype=room_size.dtype))

    # 禁用目标/障碍所在的两个格子
    keep_mask = torch.ones(9, dtype=torch.bool, device=room_size.device)
    keep_mask[target_cell_idx] = False
    keep_mask[obstacle_cell_idx] = False

    usable_cells = cell_bounds[keep_mask]        # [7,4]
    if usable_cells.shape[0] == 0:
        raise RuntimeError("No usable grid cells after wall-side shrinking.")

    # 碰撞安全距离
    safe_dist_t = robot_radius + target_radius + margin
    safe_dist_o = robot_radius + obstacle_radius + margin

    # 输出与状态
    robot_xy = torch.zeros((num_envs, 2), device=room_size.device, dtype=room_size.dtype)
    placed = torch.zeros(num_envs, dtype=torch.bool, device=room_size.device)

    # 反复尝试（最多 max_tries 次）
    for _ in range(max_tries):
        # 为每个 env 随机选择一个可用格子
        idx7 = torch.randint(low=0, high=usable_cells.shape[0], size=(num_envs,), device=room_size.device)
        sel = usable_cells.index_select(0, idx7)  # [M,4]
        x1, x2, y1, y2 = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]

        # 在该格子的有效矩形内均匀采样
        rx = torch.rand(num_envs, device=room_size.device, dtype=room_size.dtype) * (x2 - x1) + x1
        ry = torch.rand(num_envs, device=room_size.device, dtype=room_size.dtype) * (y2 - y1) + y1
        pos = torch.stack((rx, ry), dim=-1)  # [M,2]

        # 基于半径的碰撞检测（每 env 与各自 target/obstacle）
        dist_t = torch.linalg.norm(pos - target_xy, dim=-1)
        dist_o = torch.linalg.norm(pos - obstacle_xy, dim=-1)
        ok = (dist_t > safe_dist_t) & (dist_o > safe_dist_o) & (~placed)

        robot_xy = torch.where(ok.unsqueeze(-1), pos, robot_xy)
        placed = placed | ok

        if bool(placed.all()):
            break

    # 兜底：未放置成功者 → 把其选中格子的中心作为位置
    if not bool(placed.all()):
        idx7 = torch.randint(low=0, high=usable_cells.shape[0], size=(num_envs,), device=room_size.device)
        sel = usable_cells.index_select(0, idx7)
        cx = (sel[:, 0] + sel[:, 1]) * 0.5
        cy = (sel[:, 2] + sel[:, 3]) * 0.5
        center_pos = torch.stack((cx, cy), dim=-1)
        robot_xy = torch.where((~placed).unsqueeze(-1), center_pos, robot_xy)

    return robot_xy