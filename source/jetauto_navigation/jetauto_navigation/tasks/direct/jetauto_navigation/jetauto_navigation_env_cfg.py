# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from jetauto_navigation.robots.jetauto import JETAUTO_CONFIG

# =========================
# 可配置的环境参数（新增）
# =========================

@configclass
class WallsCfg:
    """围墙与场地参数"""
    # 北/南墙尺寸：长x宽x高（沿 X 方向延伸）
    size = (2.5, 0.1, 1.0)
    # 东/西墙尺寸：长x宽x高（沿 Y 方向延伸）
    size_vert = (0.1, 2.65, 1.0)
    # 场地半径（每个房间内 xy 的绝对边界 |x|,|y| <= arena_half）
    arena_half = 1.5
    # 撞墙判定余量
    wall_margin = 0.25


@configclass
class TargetCfg:
    """目标方块参数"""
    size = (0.13, 0.22, 0.16) #22 13 16
    color = (1.0, 0.0, 0.0)
    init_pos = (0.0, 0.0, 0.0)
    # 重置时，目标放在机器人前方的基准距离（米）
    forward_base = 0.8
    # 前后/左右扰动范围（±）
    noise_x = 0.1
    noise_y = 0.2
    collision_radius = 0.26


@configclass
class ObstacleCfg:
    """障碍物参数"""
    size = (0.22, 0.335, 0.26)  #33.5 22 26
    color = (0.0, 1.0, 0.0)
    # 将障碍物放在 机器人->目标 连线上的比例区间 [alpha_min, alpha_max]
    alpha_min = 0.3
    alpha_max = 0.7
    # 横向扰动（与连线垂直方向，±）
    lateral = 0.1
    # 碰撞判定半径（米）
    collision_radius = 0.4


@configclass
class CameraMountCfg:
    """相机安装与成像参数"""
    pos = (0.0, 0.1, 0.05)                # 相对父节点的位置
    rot = (0.7071, 0.7071, 0.0, 0.0)      # 四元数
    height = 320
    width = 320


@configclass
class ResetCfg:
    """重置与掩码判定相关参数"""
    room_b_offset_x = 3.0                 # 房间B沿X轴整体偏移
    mask_edge_width = 5                   # 目标贴边判定像素宽度
    target_noise = 0.03                  # 目标位置微扰范围（米）
    obstacle_noise = 0.03                # 障碍物位置微扰范围（米）


@configclass
class MotionCfg:
    """运动学与平滑参数"""
    v_xy_max = 0.6
    w_z_max = 1.5
    v_smooth = 0.2
    planar_z_lock = 0.01               # 平面锁定高度（m）


@configclass
class EnvParamsCfg:
    """环境参数总汇"""
    walls: WallsCfg = WallsCfg()
    target: TargetCfg = TargetCfg()
    obstacle: ObstacleCfg = ObstacleCfg()
    camera: CameraMountCfg = CameraMountCfg()
    reset: ResetCfg = ResetCfg()
    motion: MotionCfg = MotionCfg()


# =========================
# 任务环境配置（调整后）
# =========================

@configclass
class JetautoNavigationEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 4
    observation_space = 512*4
    state_space = 512*4

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation * 4)
    # robot(s)
    robot_cfg: ArticulationCfg = JETAUTO_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=10.0, replicate_physics=True)

    # custom parameters/scales
    dof_names = ["wheel_right_front_joint", "wheel_right_back_joint", "wheel_left_front_joint", "wheel_left_back_joint"]
    env_params: EnvParamsCfg = EnvParamsCfg()

    # ====== 新增：环境参数 ======