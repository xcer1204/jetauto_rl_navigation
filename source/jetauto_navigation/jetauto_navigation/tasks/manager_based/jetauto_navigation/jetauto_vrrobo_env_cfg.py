from __future__ import annotations

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
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from jetauto_navigation.robots.jetauto import JETAUTO_CONFIG

from . import mdp

VRROBO_SCENE_DATA_ROOT = os.environ.get(
    "VRROBO_SCENE_DATA_ROOT",
    "/home/ubuntu/xc_isaac/VR-Robo/vrrobo_isaaclab/exts/scene_data",
)
VRROBO_MULTITASK_MODEL_PATH = os.environ.get(
    "VRROBO_MULTITASK_MODEL_PATH",
    (
        "/home/ubuntu/xc_isaac/deepLabSegment/logs_multitask/"
        "multitask_2026_05_03_01_27_37/best_epoch_weights.pth"
    ),
)
VRROBO_MULTITASK_PROJECT_ROOT = os.environ.get("MULTITASK_PROJECT_ROOT", "/home/ubuntu/xc_isaac/deepLabSegment")
ASSET_OFFSET = (3.2, 0.0, -0.01)


@configclass
class JetautoVrRoboSceneCfg(InteractiveSceneCfg):
    """Scene with Jetauto robot and VR-Robo background/target objects."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(40.0, 40.0), color=None, physics_material=None),
    )

    # robot: ArticulationCfg = JETAUTO_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")


    # Align spawn pose with the reset event so the robot does not appear at the USD default pose
    # for the first rendered frames before the first environment reset is applied.
    robot: ArticulationCfg = JETAUTO_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=JETAUTO_CONFIG.init_state.replace(
            pos=(2.0, -3.0, 0.01),
            rot=(0.70710678, 0.0, 0.0, 0.70710678),
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

    object: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "object": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=os.path.join(VRROBO_SCENE_DATA_ROOT, "scene.usd"),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=ASSET_OFFSET, rot=(1.0, 0.0, 0.0, 0.0)),
            ),
            "wall_left": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Wall_1",
                spawn=sim_utils.CuboidCfg(
                    size=(3.4, 0.01, 1.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 1.9, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
            "wall_right": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Wall_2",
                spawn=sim_utils.CuboidCfg(
                    size=(3.4, 0.01, 1.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, -3.5, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
            "wall_front": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Wall_3",
                spawn=sim_utils.CuboidCfg(
                    size=(0.01, 5.4, 1.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(3.2, -0.8, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
            "wall_back": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Wall_4",
                spawn=sim_utils.CuboidCfg(
                    size=(0.01, 5.4, 1.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, -0.8, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        }
    )

    cone_red: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "cone_red": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cone_red",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=os.path.join(VRROBO_SCENE_DATA_ROOT, "red.usd"),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(2.4, 1.4, 0.03), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        }
    )

    cone_green: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "cone_green": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cone_green",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=os.path.join(VRROBO_SCENE_DATA_ROOT, "green.usd"),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(2.4, 0.0, 0.03), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        }
    )

    cone_blue: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "cone_blue": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cone_blue",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=os.path.join(VRROBO_SCENE_DATA_ROOT, "blue.usd"),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(2.4, -1.4, 0.03), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        }
    )



    # Local Isaac camera for RGB feature extraction (no external renderer dependency)
    # front_camera = CameraCfg(
    #     prim_path="/World/envs/env_.*/Robot/base_footprint/visuals/depth_camera_link/Camera",
    #     update_period=0.02,
    #     height=180,
    #     width=320,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(),
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.0, -0.1, 0.0),
    #         rot=(0.0, 0.0, 0.6820, 0.7314),
    #         convention="parent",
    #     ),
    # )


@configclass
class CommandsCfg:
    rgb_command = mdp.RGBCommandCfg(
        resampling_time_range=(1e5, 1e5),
        RGB_prob=[0.34, 0.33, 0.33],
    )


@configclass
class ActionsCfg:
    # planar_vel = mdp.PlanarVelocityActionCfg(
    #     asset_name="robot",
    #     lin_xy_step=(0.08, 0.08),
    #     yaw_step=0.2617993877991494,
    #     threshold=0.33,
    #     z_lock=0.01,
    #     lin_xy_scale=(0.6, 0.6),
    #     yaw_scale=1.5,
    #     smoothing=0.2,
    #     z_lock=None,
    # )
    planar_vel = mdp.PlanarPoseStepActionCfg(
    asset_name="robot",
    lin_xy_step=(0.08, 0.08),
    yaw_step=0.2617993877991494,
    threshold=0.33,
    z_lock=0.01,
    )



@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        gs_image = ObsTerm(
            func=mdp.gs_image_feature,
            params={
                "camera_pos": [0.0, -0.1, 0.2],
                "camera_rot": [0.0, 23.0, 0.0],
                "asset_offset_pos": list(ASSET_OFFSET),
                "save_debug_images": False,
                "save_every_n_steps": 1,
                "save_max_images": 100,
                "save_env_index": 0,
                "save_dir": "logs/gs_render_debug",
                "multitask_model_path": VRROBO_MULTITASK_MODEL_PATH,
                "multitask_project_root": VRROBO_MULTITASK_PROJECT_ROOT,
                "success_occlusion_class": "0-20%",
                "save_debug_masks": False,
                "save_mask_every_n_steps": 1,
                "save_mask_max_images": -1,
                "save_mask_env_index": 0,
                "mask_occluded_dir": "logs/gs_mask_debug/occluded",
                "mask_target_only_dir": "logs/gs_mask_debug/target_only",
                "mask_target_from_command": False,
                "mask_command_name": "rgb_command",
                "mask_target_default": "red",
                "mask_threshold": 0.5,
                "mask_binary": True,
            },
            noise=None,
        )
        # goal_command = ObsTerm(func=mdp.rgb_command, params={"command_name": "rgb_command"})
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # projected_gravity = ObsTerm(func=mdp.projected_gravity)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # goal_command = ObsTerm(func=mdp.rgb_command, params={"command_name": "rgb_command"})
        # goal_pos = ObsTerm(func=mdp.goal_pos_multi, params={"base_height": 0.0})
        robot_pos = ObsTerm(func=mdp.root_pos_e)
        robot_quat = ObsTerm(func=mdp.root_quat_w)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # projected_gravity = ObsTerm(func=mdp.projected_gravity)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RandomOcclusionObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        gs_image = ObsTerm(
            func=mdp.random_occlusion_feature,
            params={
                "feature_dim": 320,
                "history_len": 4,
                "feature_mode": "zeros",
                "occlusion_class_names": ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
                "success_occlusion_class": "0-20%",
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
class RendererRandomOcclusionObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        gs_image = ObsTerm(
            func=mdp.gs_image_feature,
            params={
                "camera_pos": [0.0, -0.1, 0.2],
                "camera_rot": [0.0, 23.0, 0.0],
                "asset_offset_pos": list(ASSET_OFFSET),
                "save_debug_images": False,
                "save_every_n_steps": 1,
                "save_max_images": 100,
                "save_env_index": 0,
                "save_dir": "logs/gs_render_debug",
                "multitask_model_path": VRROBO_MULTITASK_MODEL_PATH,
                "multitask_project_root": VRROBO_MULTITASK_PROJECT_ROOT,
                "success_occlusion_class": "0-20%",
                "save_debug_masks": False,
                "save_mask_every_n_steps": 1,
                "save_mask_max_images": -1,
                "save_mask_env_index": 0,
                "mask_occluded_dir": "logs/gs_mask_debug/occluded",
                "mask_target_only_dir": "logs/gs_mask_debug/target_only",
                "mask_target_from_command": False,
                "mask_command_name": "rgb_command",
                "mask_target_default": "red",
                "mask_threshold": 0.5,
                "mask_binary": True,
                "randomize_occlusion_prediction": True,
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


# @configclass
# class IsaacRgbObservationsCfg:
#     @configclass
#     class PolicyCfg(ObsGroup):
#         # Uses built-in frozen image encoder (ResNet18 logits) on Isaac camera RGB
#         rgb_feature = ObsTerm(
#             func=mdp.image_features,
#             params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb", "model_name": "resnet18"},
#             noise=None,
#         )
#         goal_command = ObsTerm(func=mdp.rgb_command, params={"command_name": "rgb_command"})
#         actions = ObsTerm(func=mdp.last_action)
#         base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
#         base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
#         projected_gravity = ObsTerm(func=mdp.projected_gravity)

#         def __post_init__(self):
#             self.enable_corruption = False
#             self.concatenate_terms = True

#     @configclass
#     class CriticCfg(ObsGroup):
#         rgb_feature = ObsTerm(
#             func=mdp.image_features,
#             params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb", "model_name": "resnet18"},
#             noise=None,
#         )
#         goal_command = ObsTerm(func=mdp.rgb_command, params={"command_name": "rgb_command"})
#         goal_pos = ObsTerm(func=mdp.goal_pos_multi, params={"base_height": 0.0})
#         robot_pos = ObsTerm(func=mdp.root_pos_e)
#         robot_quat = ObsTerm(func=mdp.root_quat_w)
#         actions = ObsTerm(func=mdp.last_action)
#         base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
#         base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
#         projected_gravity = ObsTerm(func=mdp.projected_gravity)

#         def __post_init__(self):
#             self.enable_corruption = False
#             self.concatenate_terms = True

#     policy: PolicyCfg = PolicyCfg()
#     critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_episode = EventTerm(
        func=mdp.randomize_robot_and_cones,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "red_cfg": SceneEntityCfg("cone_red"),
            "green_cfg": SceneEntityCfg("cone_green"),
            "blue_cfg": SceneEntityCfg("cone_blue"),
            # Match VR-Robo reset_base ranges.
            "robot_x_range": (0.0, 0.7),
            # "robot_x_range": (0.4, 0.4),
            "robot_y_range": (-3.0, 1.0),
            # "robot_y_range": (-2.5, -2.5),
            "robot_yaw_range": (-3.141592653589793, 3.141592653589793),

            # Match VR-Robo reset_cones ranges, including elevated table region (z=0.3456).
            "cone_pose_ranges": {
                "x": [(1.3, 1.3), (3.0, 3.0), (2.5, 2.5)],
                "y": [(0.2, 0.2), (-2.5, -2.5), (-0.2, -0.2)],
                "z": [(0.0, 0.0), (0.0, 0.0), (0.3456, 0.3456)],
            },
            "z_lock": 0.01,
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
            "x_limits": (0.0, 0.7),
            "y_limits": (-3.0, 1.0),
        },
    )

    # reaching = RewTerm(
    #     func=mdp.goal_distance_tanh,
    #     weight=2.0,
    #     params={"command_name": "rgb_command", "std": 1.2},
    # )
    # heading = RewTerm(
    #     func=mdp.goal_heading_alignment,
    #     weight=0.2,
    #     params={"command_name": "rgb_command"},
    # )
    # reach_bonus = RewTerm(
    #     func=mdp.goal_reach_bonus,
    #     weight=4.0,
    #     params={"command_name": "rgb_command", "threshold": 0.35},
    # )
    # action_penalty = RewTerm(func=mdp.action_l2, weight=-0.02)

    # visibility = RewTerm(
    #     func=mdp.target_visibility_reward,
    #     weight=2.0,   # 你可以调这个
    # )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # goal_reached = DoneTerm(func=mdp.goal_reached, params={"command_name": "rgb_command", "threshold": 0.35})
    visibility_success = DoneTerm(
        func=mdp.visibility_success,
        params={"threshold": 0.9, "success_class_name": "0-20%", "x_limits": (0.0, 0.7), "y_limits": (-3.0, 1.0)},
    )
    out_of_bounds = DoneTerm(func=mdp.robot_out_of_bounds, params={"x_limits": (0.0, 0.7), "y_limits": (-3.0, 1.0)})


@configclass
class JetautoVrRoboEnvCfg(ManagerBasedRLEnvCfg):
    scene: JetautoVrRoboSceneCfg = JetautoVrRoboSceneCfg(num_envs=48, env_spacing=8.0)
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
        self.viewer.eye = (0.5, 0.0, 8.2)
        self.viewer.lookat = (1.2, -0.8, 0.2)


@configclass
class JetautoVrRoboEnvCfg_PLAY(JetautoVrRoboEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


@configclass
class JetautoVrRoboRandomOccEnvCfg(JetautoVrRoboEnvCfg):
    observations: RandomOcclusionObservationsCfg = RandomOcclusionObservationsCfg()


@configclass
class JetautoVrRoboRandomOccEnvCfg_PLAY(JetautoVrRoboRandomOccEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


@configclass
class JetautoVrRoboRendererRandomOccEnvCfg(JetautoVrRoboEnvCfg):
    observations: RendererRandomOcclusionObservationsCfg = RendererRandomOcclusionObservationsCfg()


@configclass
class JetautoVrRoboRendererRandomOccEnvCfg_PLAY(JetautoVrRoboRendererRandomOccEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


# @configclass
# class JetautoVrRoboIsaacRgbEnvCfg(JetautoVrRoboEnvCfg):
#     observations: IsaacRgbObservationsCfg = IsaacRgbObservationsCfg()

#     def __post_init__(self) -> None:
#         super().__post_init__()
#         self.scene.num_envs = 48
#         self.scene.env_spacing = 8.0


# @configclass
# class JetautoVrRoboIsaacRgbEnvCfg_PLAY(JetautoVrRoboIsaacRgbEnvCfg):
#     def __post_init__(self) -> None:
#         super().__post_init__()
#         self.scene.num_envs = 1
#         self.scene.env_spacing = 4.0
#         self.observations.policy.enable_corruption = False
