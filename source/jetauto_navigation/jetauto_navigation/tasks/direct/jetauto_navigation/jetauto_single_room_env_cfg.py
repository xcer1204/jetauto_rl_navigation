# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .jetauto_navigation_env_cfg import (
    EnvParamsCfg,
    JetautoNavigationEnvCfg,
    ObstacleCfg,
    TargetCfg,
)


@configclass
class SingleRoomTargetCfg(TargetCfg):
    """Cylinder target: diameter 25cm, height 30cm."""

    size = (0.25, 0.25, 0.30)
    collision_radius = 0.05


@configclass
class SingleRoomObstacleCfg(ObstacleCfg):
    """Box obstacle: 30cm x 13cm x 40cm."""

    size = (0.30, 0.13, 0.40)
    collision_radius = 0.05


@configclass
class SingleRoomEnvParamsCfg(EnvParamsCfg):
    """Reuse base params but swap target/obstacle shapes."""

    target: SingleRoomTargetCfg = SingleRoomTargetCfg()
    obstacle: SingleRoomObstacleCfg = SingleRoomObstacleCfg()


@configclass
class JetautoSingleRoomEnvCfg(JetautoNavigationEnvCfg):
    """Configuration for single-room navigation."""

    env_params: SingleRoomEnvParamsCfg = SingleRoomEnvParamsCfg()


@configclass
class JetautoSingleRoomEmptyEnvCfg(JetautoSingleRoomEnvCfg):
    """Configuration for single-room navigation without spawning walls/obstacles/target."""
