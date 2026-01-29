import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR



# https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/NVIDIA/Kaya/props/Kaya_Body.usd
# https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Kaya/kaya.usd
JETAUTO_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Kaya/kaya.usd",
        usd_path="/home/ubuntu/xc_isaac/jetauto_rl_navigation-main/source/jetauto_driveable/jetauto_driveable/jetauto_driveable.usd",
        
        # 可选：配置刚体和关节属性，如禁用自碰撞等
        # articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False), 
        # rigid_props=sim_utils.RigidBodyPropertiesCfg(max_linear_velocity=1000.0, max_angular_velocity=1000.0)
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), 
        joint_pos={"wheel_right_front_joint": 0.0, "wheel_right_back_joint": 0.0, "wheel_left_front_joint": 0.0, "wheel_left_back_joint": 0.0} 
    ),
    # 配置执行器：为每个轮子关节设置一个隐式速度控制执行器
    actuators={
        "wheel_actuators": ImplicitActuatorCfg(
            
            joint_names_expr=["wheel_right_front_joint", "wheel_right_back_joint", "wheel_left_front_joint", "wheel_left_back_joint"],
            velocity_limit=100.0, stiffness=0.0, damping=10000000.0
        )
    }
)
