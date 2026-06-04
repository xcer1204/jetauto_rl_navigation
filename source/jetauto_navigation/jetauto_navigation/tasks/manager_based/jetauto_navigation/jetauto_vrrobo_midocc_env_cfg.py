from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from jetauto_navigation.robots.jetauto import JETAUTO_CONFIG

from . import mdp
from . import midocc_mdp


MIDOCC_MODEL_PATH = (
    "/home/ubuntu/xc_isaac/deepLabSegment/logs_multitask/"
    "multitask_2026_05_03_01_27_37/best_epoch_weights.pth"
)
MIDOCC_PROJECT_ROOT = os.environ.get("MULTITASK_PROJECT_ROOT", "/home/ubuntu/xc_isaac/deepLabSegment")

MIDOCC_X_SAMPLING_RANGES = ((0.52, 0.76),)
MIDOCC_Y_SAMPLING_RANGES = ((0.85, 1.40),)
MIDOCC_ROBOT_Z = 0.22
MIDOCC_ASSET_OFFSET = (0.0, 0.0, 0.0)
MIDOCC_X_LIMITS = (0.0, 3.0)
MIDOCC_Y_LIMITS = (0.0, 2.32)
MIDOCC_YAW_LIMITS = (-math.radians(90.0), math.radians(90.0))
MIDOCC_CAMERA_SAMPLING_MODE = "world_xy_yaw"
MIDOCC_CAMERA_ROT_DEG = (0.0, 0.0, 0.0)
MIDOCCD_LIN_XY_STEP = (0.02, 0.02)
MIDOCCD_YAW_STEP = math.radians(5.0)
MIDOCCD_EPISODE_LENGTH_S = 25.0
MIDOCC_INITIAL_X = 0.5 * (MIDOCC_X_SAMPLING_RANGES[0][0] + MIDOCC_X_SAMPLING_RANGES[0][1])
MIDOCC_INITIAL_Y = 0.5 * (MIDOCC_Y_SAMPLING_RANGES[0][0] + MIDOCC_Y_SAMPLING_RANGES[0][1])
MIDOCC_WALL_THICKNESS = 0.01
MIDOCC_WALL_HEIGHT = 1.2
MIDOCC_WALL_Z = 0.5 * MIDOCC_WALL_HEIGHT
MIDOCC_WALL_X_SPAN = MIDOCC_X_LIMITS[1] - MIDOCC_X_LIMITS[0]
MIDOCC_WALL_Y_SPAN = MIDOCC_Y_LIMITS[1] - MIDOCC_Y_LIMITS[0]
MIDOCC_WALL_CENTER_X = 0.5 * (MIDOCC_X_LIMITS[0] + MIDOCC_X_LIMITS[1])
MIDOCC_WALL_CENTER_Y = 0.5 * (MIDOCC_Y_LIMITS[0] + MIDOCC_Y_LIMITS[1])

MIDOCC_RED_POS = (2.4, 1.3, 0.0)
MIDOCC_GREEN_POS = (1.3, 1.2, 0.0)
MIDOCC_BLUE_POS = (99.0, 99.0, 0.0)


@configclass
class JetautoVrRoboMidOccSceneCfg(InteractiveSceneCfg):
    """Isaac Lab proxy scene matched to the no-out-of-view mid-occlusion dataset."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(120.0, 120.0), color=None, physics_material=None),
    )

    robot: ArticulationCfg = JETAUTO_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=JETAUTO_CONFIG.init_state.replace(
            pos=(MIDOCC_INITIAL_X, MIDOCC_INITIAL_Y, MIDOCC_ROBOT_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.8, 0.8, 0.8), intensity=3500.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.15, 0.15, 0.15), intensity=900.0),
    )

    walls: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "wall_y_min": RigidObjectCfg(
                prim_path="/World/envs/env_.*/MidOcc_Wall_Y_Min",
                spawn=sim_utils.CuboidCfg(
                    size=(MIDOCC_WALL_X_SPAN, MIDOCC_WALL_THICKNESS, MIDOCC_WALL_HEIGHT),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(MIDOCC_WALL_CENTER_X, MIDOCC_Y_LIMITS[0], MIDOCC_WALL_Z),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
            "wall_y_max": RigidObjectCfg(
                prim_path="/World/envs/env_.*/MidOcc_Wall_Y_Max",
                spawn=sim_utils.CuboidCfg(
                    size=(MIDOCC_WALL_X_SPAN, MIDOCC_WALL_THICKNESS, MIDOCC_WALL_HEIGHT),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(MIDOCC_WALL_CENTER_X, MIDOCC_Y_LIMITS[1], MIDOCC_WALL_Z),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
            "wall_x_min": RigidObjectCfg(
                prim_path="/World/envs/env_.*/MidOcc_Wall_X_Min",
                spawn=sim_utils.CuboidCfg(
                    size=(MIDOCC_WALL_THICKNESS, MIDOCC_WALL_Y_SPAN, MIDOCC_WALL_HEIGHT),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(MIDOCC_X_LIMITS[0], MIDOCC_WALL_CENTER_Y, MIDOCC_WALL_Z),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
            "wall_x_max": RigidObjectCfg(
                prim_path="/World/envs/env_.*/MidOcc_Wall_X_Max",
                spawn=sim_utils.CuboidCfg(
                    size=(MIDOCC_WALL_THICKNESS, MIDOCC_WALL_Y_SPAN, MIDOCC_WALL_HEIGHT),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(MIDOCC_X_LIMITS[1], MIDOCC_WALL_CENTER_Y, MIDOCC_WALL_Z),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
        }
    )

    cone_red: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "cone_red": RigidObjectCfg(
                prim_path="/World/envs/env_.*/MidOcc_RedTarget",
                spawn=sim_utils.CuboidCfg(
                    size=(0.145, 0.174, 0.350),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=MIDOCC_RED_POS, rot=(1.0, 0.0, 0.0, 0.0)),
            )
        }
    )

    cone_green: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "cone_green": RigidObjectCfg(
                prim_path="/World/envs/env_.*/MidOcc_GreenOccluder",
                spawn=sim_utils.CuboidCfg(
                    size=(0.140, 0.270, 0.350),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=MIDOCC_GREEN_POS, rot=(1.0, 0.0, 0.0, 0.0)),
            )
        }
    )

    cone_blue: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "cone_blue": RigidObjectCfg(
                prim_path="/World/envs/env_.*/MidOcc_BlueHidden",
                spawn=sim_utils.CuboidCfg(
                    size=(0.05, 0.05, 0.05),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=MIDOCC_BLUE_POS, rot=(1.0, 0.0, 0.0, 0.0)),
            )
        }
    )


@configclass
class CommandsCfg:
    rgb_command = mdp.RGBCommandCfg(
        resampling_time_range=(1e5, 1e5),
        RGB_prob=[1.0, 0.0, 0.0],
    )


@configclass
class ActionsCfg:
    planar_vel = midocc_mdp.PlanarPoseStepActionCfg(
        asset_name="robot",
        lin_xy_step=(0.08, 0.08),
        yaw_step=math.radians(15.0),
        threshold=0.33,
        z_lock=MIDOCC_ROBOT_Z,
        x_limits=MIDOCC_X_LIMITS,
        y_limits=MIDOCC_Y_LIMITS,
        yaw_limits=MIDOCC_YAW_LIMITS,
    )


@configclass
class ContinuousActionsCfg:
    planar_vel = midocc_mdp.PlanarPoseContinuousActionCfg(
        asset_name="robot",
        lin_xy_step=(0.08, 0.08),
        yaw_step=math.radians(15.0),
        z_lock=MIDOCC_ROBOT_Z,
        x_limits=MIDOCC_X_LIMITS,
        y_limits=MIDOCC_Y_LIMITS,
        yaw_limits=MIDOCC_YAW_LIMITS,
    )


@configclass
class ContinuousDDActionsCfg:
    planar_vel = midocc_mdp.PlanarPoseContinuousActionCfg(
        asset_name="robot",
        lin_xy_step=MIDOCCD_LIN_XY_STEP,
        yaw_step=MIDOCCD_YAW_STEP,
        z_lock=MIDOCC_ROBOT_Z,
        x_limits=MIDOCC_X_LIMITS,
        y_limits=MIDOCC_Y_LIMITS,
        yaw_limits=MIDOCC_YAW_LIMITS,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        gs_image = ObsTerm(
            func=midocc_mdp.gs_look_at_target_image_feature,
            params={
                "camera_sampling_mode": MIDOCC_CAMERA_SAMPLING_MODE,
                "camera_pos": [0.0, 0.0, 0.0],
                "camera_rot_deg": list(MIDOCC_CAMERA_ROT_DEG),
                "asset_offset_pos": list(MIDOCC_ASSET_OFFSET),
                "render_server_host": "localhost",
                "render_server_port": 18862,
                "rgb_socket_host": "localhost",
                "rgb_socket_port": 12345,
                "history_len": 4,
                "save_debug_images": False,
                "save_every_n_steps": 1,
                "save_max_images": 100,
                "save_env_index": 0,
                "save_dir": "logs/midocc_gs_render_debug",
                "multitask_model_path": MIDOCC_MODEL_PATH,
                "multitask_project_root": MIDOCC_PROJECT_ROOT,
                "success_occlusion_class": "0-20%",
                "save_debug_masks": False,
                "save_mask_every_n_steps": 1,
                "save_mask_max_images": -1,
                "save_mask_env_index": 0,
                "mask_occluded_dir": "logs/midocc_gs_mask_debug/occluded",
                "mask_target_only_dir": "logs/midocc_gs_mask_debug/target_only",
                "mask_target_from_command": False,
                "mask_command_name": "rgb_command",
                "mask_target_default": "red",
                "mask_threshold": 0.5,
                "mask_binary": True,
            },
            noise=None,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        robot_pos = ObsTerm(func=mdp.root_pos_e)
        robot_quat = ObsTerm(func=mdp.root_quat_w)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_episode = EventTerm(
        func=midocc_mdp.randomize_robot_and_cones,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "red_cfg": SceneEntityCfg("cone_red"),
            "green_cfg": SceneEntityCfg("cone_green"),
            "blue_cfg": SceneEntityCfg("cone_blue"),
            "robot_x_ranges": list(MIDOCC_X_SAMPLING_RANGES),
            "robot_y_ranges": list(MIDOCC_Y_SAMPLING_RANGES),
            "robot_yaw_range": MIDOCC_YAW_LIMITS,
            "cone_pose_ranges": {
                "x": [(MIDOCC_RED_POS[0], MIDOCC_RED_POS[0]), (MIDOCC_GREEN_POS[0], MIDOCC_GREEN_POS[0]), (MIDOCC_BLUE_POS[0], MIDOCC_BLUE_POS[0])],
                "y": [(MIDOCC_RED_POS[1], MIDOCC_RED_POS[1]), (MIDOCC_GREEN_POS[1], MIDOCC_GREEN_POS[1]), (MIDOCC_BLUE_POS[1], MIDOCC_BLUE_POS[1])],
                "z": [(MIDOCC_RED_POS[2], MIDOCC_RED_POS[2]), (MIDOCC_GREEN_POS[2], MIDOCC_GREEN_POS[2]), (MIDOCC_BLUE_POS[2], MIDOCC_BLUE_POS[2])],
            },
            "z_lock": MIDOCC_ROBOT_Z,
        },
    )


@configclass
class RewardsCfg:
    visibility_progress = RewTerm(
        func=mdp.visibility_progress_reward,
        weight=1.0,
        params={
            "success_threshold": 0.9,
            "success_bonus": 5.0,
            "success_class_name": "0-20%",
            "idle_penalty": -0.01,
            "collision_penalty": -5.0,
            "x_limits": MIDOCC_X_LIMITS,
            "y_limits": MIDOCC_Y_LIMITS,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    visibility_success = DoneTerm(
        func=mdp.visibility_success,
        params={"threshold": 0.9, "success_class_name": "0-20%", "x_limits": MIDOCC_X_LIMITS, "y_limits": MIDOCC_Y_LIMITS},
    )
    out_of_bounds = DoneTerm(func=mdp.robot_out_of_bounds, params={"x_limits": MIDOCC_X_LIMITS, "y_limits": MIDOCC_Y_LIMITS})


@configclass
class JetautoVrRoboMidOccEnvCfg(ManagerBasedRLEnvCfg):
    scene: JetautoVrRoboMidOccSceneCfg = JetautoVrRoboMidOccSceneCfg(num_envs=48, env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 20
        self.episode_length_s = 15.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.viewer.eye = (0.65, 1.10, 6.0)
        self.viewer.lookat = (1.5, 1.2, 0.2)


@configclass
class JetautoVrRoboMidOccEnvCfg_PLAY(JetautoVrRoboMidOccEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False
        self.observations.policy.gs_image.params["save_debug_images"] = True
        self.observations.policy.gs_image.params["save_every_n_steps"] = 1
        self.observations.policy.gs_image.params["save_max_images"] = 200
        self.observations.policy.gs_image.params["save_env_index"] = 0
        self.observations.policy.gs_image.params["save_dir"] = "logs/midocc_gs_render_debug_play"


@configclass
class JetautoVrRoboMidOccContinuousEnvCfg(JetautoVrRoboMidOccEnvCfg):
    actions: ContinuousActionsCfg = ContinuousActionsCfg()


@configclass
class JetautoVrRoboMidOccContinuousEnvCfg_PLAY(JetautoVrRoboMidOccContinuousEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False
        self.observations.policy.gs_image.params["save_debug_images"] = True
        self.observations.policy.gs_image.params["save_every_n_steps"] = 1
        self.observations.policy.gs_image.params["save_max_images"] = 200
        self.observations.policy.gs_image.params["save_env_index"] = 0
        self.observations.policy.gs_image.params["save_dir"] = "logs/midocc_continuous_gs_render_debug_play"


@configclass
class JetautoVrRoboMidOccDEnvCfg(JetautoVrRoboMidOccEnvCfg):
    actions: ContinuousDDActionsCfg = ContinuousDDActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = MIDOCCD_EPISODE_LENGTH_S


@configclass
class JetautoVrRoboMidOccDEnvCfg_PLAY(JetautoVrRoboMidOccDEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False
        self.observations.policy.gs_image.params["save_debug_images"] = True
        self.observations.policy.gs_image.params["save_every_n_steps"] = 1
        self.observations.policy.gs_image.params["save_max_images"] = 300
        self.observations.policy.gs_image.params["save_env_index"] = 0
        self.observations.policy.gs_image.params["save_dir"] = "logs/midocc_d_gs_render_debug_play"
