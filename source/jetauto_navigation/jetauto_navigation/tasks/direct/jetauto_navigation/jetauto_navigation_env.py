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

        self._hist_len = 4
        self._feat_hist = torch.zeros(self.num_envs, self._hist_len, 512, device=self.device)

        self.v_xy_max = 0.6  # m/s
        self.w_z_max = 1.5  # rad/s
        self.v_smooth = 0.2  # [0~1] 越大越稳
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


        # 房间 A 围墙
        wall_size = (3.0, 0.1, 0.4)
        wall_size_vert = (0.1, 3.0, 0.4)
        wall_z = wall_size[2] / 2
        walls_a = [
            {"name": "RoomA_Wall_N", "pos": (0.0, 1.5, wall_z), "size": wall_size},
            {"name": "RoomA_Wall_S", "pos": (0.0, -1.5, wall_z), "size": wall_size},
            {"name": "RoomA_Wall_E", "pos": (1.5, 0.0, wall_z), "size": wall_size_vert},
            {"name": "RoomA_Wall_W", "pos": (-1.5, 0.0, wall_z), "size": wall_size_vert},
        ]
        for wall in walls_a:
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


        if not hasattr(self, "_static_target_pos"):
            self.target_size = (0.2, 0.2, 0.2)
            self.obstacle_size = (0.1, 0.1, 0.1)

        # 房间 A 目标
        target_half_h = self.target_size[2] / 2
        target_cfg_a = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Target_A",
            spawn=sim_utils.CuboidCfg(
                size=self.target_size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
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
        obstacle_size = (0.1, 0.1, 0.1)
        half_height = obstacle_size[2] / 2
        obstacle_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Obstacle",
            spawn=sim_utils.CuboidCfg(
                size=obstacle_size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
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
            height=320, width=320,
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
        offset_x = 3.0

        # 机器人 B
        self.robot_b = Articulation(self.cfg.robot_cfg.replace(prim_path="/World/envs/env_.*/Robot_B"))
        self.scene.articulations["robot_b"] = self.robot_b

        # 房间 B 围墙
        walls_b = [
            {"name": "RoomB_Wall_N", "pos": (offset_x, 1.5, wall_z), "size": wall_size},
            {"name": "RoomB_Wall_S", "pos": (offset_x, -1.5, wall_z), "size": wall_size},
            {"name": "RoomB_Wall_E", "pos": (offset_x + 1.5, 0.0, wall_z), "size": wall_size_vert},
            {"name": "RoomB_Wall_W", "pos": (offset_x - 1.5, 0.0, wall_z), "size": wall_size_vert},
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
                size=self.target_size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
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
            height=320, width=320,
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


    # def _visualize_markers(self):
    #     self.marker_locations = self.robot_a.data.root_pos_w
    #     self.forward_marker_orientations = self.robot_a.data.root_quat_w
    #     self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

    #     loc = self.marker_locations + self.marker_offset
    #     loc = torch.vstack((loc, loc))
    #     rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

    #     all_envs = torch.arange(self.cfg.scene.num_envs)
    #     indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))

    #     self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    # def _apply_action(self) -> None:
    #     self.robot_a.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)
    #     self.robot_b.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    # 放在类里，与 _apply_action 平级
    def _lock_planar(self, robot, z_lock: float = 0.09105) -> None:
        """锁定在平面：z=z_lock，去掉 pitch/roll，仅保留 yaw。"""
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
        self._lock_planar(self.robot_a, 0.09105)
        self._lock_planar(self.robot_b, 0.09105)

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

        print("Robot local XYZ:", self.robot_a.data.root_pos_w[:, :3]-self.scene.env_origins[:, :3])

        arena_half = getattr(self, "_arena_half", 1.25)
        wall_collision = (torch.abs(robot_local) > (arena_half)).any(dim=-1)

        if hasattr(self, "_obstacles") and len(self._obstacles) > 0:
            obs_xy = torch.stack([obs.data.root_pos_w[:, :2] for obs in self._obstacles], dim=1)
            dist_to_obs = torch.norm(robot_xy.unsqueeze(1) - obs_xy, dim=-1)
            obs_collision = (dist_to_obs < 0.20).any(dim=1)
        else:
            obs_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # print("Wall collisions:", wall_collision.sum().item(), "Obstacle collisions:", obs_collision.sum().item())

        self.collision_mask = wall_collision | obs_collision

        # return {"policy": resnet_features, "critic": resnet_features, "rgb":rgb_a}
        return {
            "policy": obs_2048,
            "critic": obs_2048,
            # "rgb": rgb_a  # 如需调试可保留，但 skrl 只会取 policy/critic
        }

    def _mask_touches_edge(self, mask, border_width=5):
        """判断掩码是否触碰到图像边缘"""
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

        # print("Terminated:", terminated.sum().item(), "Success:", success.sum().item(), "Failed:", failed.sum().item(), "Truncated:", truncated.sum().item())


        return terminated, truncated


    # def _reset_idx(self, env_ids: Sequence[int] | None = None):

    #     super()._reset_idx(env_ids)
    #     if env_ids is None:
    #         env_ids = torch.arange(self.num_envs, device=self.device)

    #     # 参数：最小安全距离,考虑到障碍物具体的大小
    #     min_dist = 0.5
    #     # 缓冲清零
    #     self._feat_hist[env_ids] = 0.0

    #     # 重置机器人位置和朝向（放置到环境中心，速度清零）
    #     root_states = self.robot_a.data.default_root_state[env_ids].clone()
    #     root_states[:, :3] += self.scene.env_origins[env_ids]  # 设置位置为环境原点
    #     self.robot_a.write_root_pose_to_sim(root_states[:, :7], env_ids=env_ids)
    #     self.robot_a.write_root_velocity_to_sim(root_states[:, 7:], env_ids=env_ids)

    #     root_states_b = self.robot_b.data.default_root_state[env_ids].clone()
    #     root_states_b[:, :3] += self.scene.env_origins[env_ids]  # 设置位置为环境原点
    #     root_states_b[:, 0] += 3.0
    #     self.robot_b.write_root_pose_to_sim(root_states_b[:, :7], env_ids=env_ids)
    #     self.robot_b.write_root_velocity_to_sim(root_states_b[:, 7:], env_ids=env_ids)

    #     # 重置机器人的关节（轮子）位置和速度
    #     joint_pos_a = self.robot_a.data.default_joint_pos[env_ids].clone()
    #     joint_vel_a = torch.zeros_like(joint_pos_a)
    #     self.robot_a.set_joint_position_target(joint_pos_a, env_ids=env_ids)
    #     self.robot_a.write_joint_state_to_sim(joint_pos_a, joint_vel_a, env_ids=env_ids)

    #     joint_pos_b = self.robot_b.data.default_joint_pos[env_ids].clone()
    #     joint_vel_b = torch.zeros_like(joint_pos_b)
    #     self.robot_b.set_joint_position_target(joint_pos_b, env_ids=env_ids)
    #     self.robot_b.write_joint_state_to_sim(joint_pos_b, joint_vel_b, env_ids=env_ids)
    #     # 随机重置目标


    #     # 获取机器人位置和朝向
    #     root_states = self.robot_a.data.root_state_w[env_ids].clone()
    #     robot_pos = root_states[:, 0:3]  # (N,3)
    #     robot_quat = root_states[:, 3:7]  # (N,4)

    #     # 计算“前方偏移”（这里 0.8m）
    #     forward_offset = torch.tensor([0.8, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)

    #     # 把 offset 从机器人坐标系旋转到世界系
    #     forward_world = math_utils.quat_apply(robot_quat, forward_offset)

    #     # 设置目标位置：机器人位置 + 偏移 + 高度
    #     # ---------------------
    #     # 目标：放在 Jetbot 前方 + 随机扰动
    #     # ---------------------
    #     target_state = self._target_a.data.default_root_state[env_ids].clone()
    #     target_state_b = self._target_a.data.default_root_state[env_ids].clone()

    #     # Jetbot 初始位置
    #     robot_pos = root_states[:, 0:3]

    #     # 基础前方位置 (比如 0.8 m 前方)
    #     base_forward = torch.tensor([0.8, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)

    #     # 加入随机扰动（左右 ±0.2m，前后 ±0.1m，高度不变）
    #     noise = torch.zeros_like(base_forward)
    #     noise[:, 0] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.2  # 前后
    #     noise[:, 1] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.4  # 左右

    #     target_pos = robot_pos + base_forward + noise
    #     target_pos[:, 2] = target_state[:, 2]  # 保持原始高度

    #     target_state[:, 0:3] = target_pos
    #     target_state_b[:, 0:3] = target_pos
    #     target_state_b[:, 0] += 3.0

    #     self._target_a.write_root_pose_to_sim(target_state[:, :7], env_ids=env_ids)
    #     self._target_a.write_root_velocity_to_sim(target_state[:, 7:], env_ids=env_ids)

    #     self._target_b.write_root_pose_to_sim(target_state_b[:, :7], env_ids=env_ids)
    #     self._target_b.write_root_velocity_to_sim(target_state_b[:, 7:], env_ids=env_ids)


    #     # target_state = self._target.data.default_root_state[env_ids].clone()
    #     # target_state[:, 0:3] = robot_pos + forward_world
    #     # self._target.write_root_pose_to_sim(target_state[:, :7], env_ids=env_ids)
    #     # self._target.write_root_velocity_to_sim(target_state[:, 7:], env_ids=env_ids)

    #     # target_state = self._target.data.default_root_state[env_ids].clone()
    #     # rand_xy = torch.rand((len(env_ids), 2), device=self.device) * 2.0 - 1.0
    #     # rand_xy *= 0.8
    #     # target_xy = self.scene.env_origins[env_ids][:, 0:2] + rand_xy
    #     # # target_state[:, 0:2] = target_xy
    #     # self._target.write_root_pose_to_sim(target_state[:, :7], env_ids=env_ids)
    #     # self._target.write_root_velocity_to_sim(target_state[:, 7:], env_ids=env_ids)

    #     # ---------------------
    #     # 障碍物：放在 Jetbot -> Target 方向上，带扰动
    #     # ---------------------

    #     # ---- 3. 放置障碍物 ----
    #     blocker = self._obstacles[0]
    #     blocker_state = blocker.data.default_root_state[env_ids].clone()

    #     direction = target_pos - robot_pos
    #     direction_xy = direction[:, :2]
    #     dist = torch.norm(direction_xy, dim=-1, keepdim=True)
    #     unit_dir = direction_xy / (dist + 1e-6)

    #     # 障碍位于 30%~70% 之间
    #     alpha = 0.3 + 0.4 * torch.rand(len(env_ids), 1, device=self.device)
    #     base_blocker_pos = robot_pos[:, :2] + alpha * direction_xy

    #     # 横向扰动 ±0.1m
    #     perp = torch.stack([-unit_dir[:, 1], unit_dir[:, 0]], dim=1)
    #     lateral_offset = (torch.rand(len(env_ids), 1, device=self.device) - 0.5) * 0.2 * perp
    #     base_blocker_pos += lateral_offset.squeeze(1)

    #     blocker_state[:, 0:2] = base_blocker_pos
    #     blocker.write_root_pose_to_sim(blocker_state[:, :7], env_ids)
    #     blocker.write_root_velocity_to_sim(blocker_state[:, 7:], env_ids)

    #     # ---- 4. 清零状态 ----
    #     self.collision_mask[env_ids] = False
    #     self.curr_vis[env_ids] = 0.0
    #     self.prev_vis[env_ids] = 0.0



    def _reset_idx(self, env_ids: Sequence[int] | None = None):

        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # 参数：最小安全距离,考虑到障碍物具体的大小
        min_dist = 0.1
        arena_half = 1.4 


        if not hasattr(self, "_static_target_pos"):
            # 只在第一次初始化时设置
            self._static_target_pos = torch.tensor([0.8, 0.0, 0.1], device=self.device).repeat(self.num_envs, 1)
            self._static_obstacle_pos = torch.tensor([0.4, 0.0, 0.05], device=self.device).repeat(self.num_envs, 1)

        # 给target obstacle微小扰动（2cm 量级）
        target_pos = self._static_target_pos[env_ids] + torch.randn(len(env_ids), 3, device=self.device) * 0.02
        obstacle_pos = self._static_obstacle_pos[env_ids] + torch.randn(len(env_ids), 3, device=self.device) * 0.02

        target_center = target_pos[0, :2]
        obstacle_center = obstacle_pos[0, :2]

        robot_xy = self.sample_safe_rect_positions(
            num_envs=len(env_ids),
            target_center=target_center,
            target_size=self.target_size[:2],
            obstacle_center=obstacle_center,
            obstacle_size=self.obstacle_size[:2],
            arena_half=1.4,
            margin=0.1
        )

        robot_pos = torch.cat([robot_xy, torch.zeros(len(env_ids), 1, device=self.device)], dim=-1)
    

        # 写入机器人和对应的 robot_b
        # --------------------
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


        # # 重置机器人位置和朝向（放置到环境中心，速度清零）
        # root_states = self.robot_a.data.default_root_state[env_ids].clone()
        # root_states[:, :3] += self.scene.env_origins[env_ids]  # 设置位置为环境原点
        # self.robot_a.write_root_pose_to_sim(root_states[:, :7], env_ids=env_ids)
        # self.robot_a.write_root_velocity_to_sim(root_states[:, 7:], env_ids=env_ids)

        # root_states_b = self.robot_b.data.default_root_state[env_ids].clone()
        # root_states_b[:, :3] += self.scene.env_origins[env_ids]  # 设置位置为环境原点
        # root_states_b[:, 0] += 3.0
        # self.robot_b.write_root_pose_to_sim(root_states_b[:, :7], env_ids=env_ids)
        # self.robot_b.write_root_velocity_to_sim(root_states_b[:, 7:], env_ids=env_ids)

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
        target_state_b[:, 0] += 3.0
        self._target_b.write_root_pose_to_sim(target_state_b[:, :7], env_ids)
        self._target_b.write_root_velocity_to_sim(target_state_b[:, 7:], env_ids)

        # --------------------
        # 清零状态
        # --------------------
        self.collision_mask[env_ids] = False
        self.curr_vis[env_ids] = 0.0
        self.prev_vis[env_ids] = 0.0


    def rect_bounds(self, center, size):
        # 自动处理 tuple / list / numpy 类型
        if not torch.is_tensor(center):
            center = torch.tensor(center, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(size):
            size = torch.tensor(size, dtype=torch.float32, device=self.device)
        half = size / 2
        return center[0] - half[0], center[0] + half[0], center[1] - half[1], center[1] + half[1]


    def sample_safe_rect_positions(
        self,
        num_envs: int,
        target_center: torch.Tensor, target_size: torch.Tensor,
        obstacle_center: torch.Tensor, obstacle_size: torch.Tensor,
        arena_half: float = 1.4, margin: float = 0.1
    ):
        device = self.device

        # 房间范围
        x_min, x_max = -arena_half, arena_half
        y_min, y_max = -arena_half, arena_half

        # 得到目标/障碍的边界
        tx1, tx2, ty1, ty2 = self.rect_bounds(target_center, target_size)
        bx1, bx2, by1, by2 = self.rect_bounds(obstacle_center, obstacle_size)

        # 加 margin（安全距离）
        tx1 -= margin; tx2 += margin; ty1 -= margin; ty2 += margin
        bx1 -= margin; bx2 += margin; by1 -= margin; by2 += margin

        # 合并禁区 (union)
        x_forbidden_min = min(tx1, bx1)
        x_forbidden_max = max(tx2, bx2)
        y_forbidden_min = min(ty1, by1)
        y_forbidden_max = max(ty2, by2)

        # ---- X 方向可采样区间 ----
        left_len = max(0.0, x_forbidden_min - x_min)
        right_len = max(0.0, x_max - x_forbidden_max)
        total_len_x = left_len + right_len

        # 采样决策：落在左边还是右边
        choice_x = torch.rand(num_envs, device=device)
        in_left_x = choice_x < (left_len / (total_len_x + 1e-6))
        x = torch.empty(num_envs, device=device)

        if left_len > 0:
            x[in_left_x] = torch.rand(in_left_x.sum(), device=device) * left_len + x_min
        if right_len > 0:
            x[~in_left_x] = torch.rand((~in_left_x).sum(), device=device) * right_len + x_forbidden_max

        # ---- Y 方向同理 ----
        bottom_len = max(0.0, y_forbidden_min - y_min)
        top_len = max(0.0, y_max - y_forbidden_max)
        total_len_y = bottom_len + top_len

        choice_y = torch.rand(num_envs, device=device)
        in_bottom_y = choice_y < (bottom_len / (total_len_y + 1e-6))
        y = torch.empty(num_envs, device=device)
        if bottom_len > 0:
            y[in_bottom_y] = torch.rand(in_bottom_y.sum(), device=device) * bottom_len + y_min
        if top_len > 0:
            y[~in_bottom_y] = torch.rand((~in_bottom_y).sum(), device=device) * top_len + y_forbidden_max

        return torch.stack([x, y], dim=-1)
