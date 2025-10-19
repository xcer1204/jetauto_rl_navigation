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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=10.0, replicate_physics=True)

    # custom parameters/scales
    dof_names = ["wheel_right_front_joint", "wheel_right_back_joint", "wheel_left_front_joint", "wheel_left_back_joint"]