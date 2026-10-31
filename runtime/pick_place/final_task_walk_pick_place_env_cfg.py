from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from pxr import UsdGeom

import envs.mdp as base_mdp
from envs import ManagerBasedRLEnvCfg
from envs.mdp.actions.actions_cfg import JointPositionActionCfg

from .mdp.events import reset_rigid_object_pair_without_overlap


FINAL_WALK_PICK_PLACE_TASK_ID = "Isaac-Final-Walk-PickPlace-Abs-v0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK1_ASSETS_ROOT = Path(os.environ.get("TASK1_ASSETS_ROOT", PROJECT_ROOT / "assets/task1")).expanduser()
ASSETS_USD_ROOT = TASK1_ASSETS_ROOT / "assets" / "usd"
SCENE_USD_PATH = Path(
    os.environ.get(
        "TASK1_SCENE_USD_PATH",
        str(PROJECT_ROOT / "assets/shared/scenes/task12.usdc"),
    )
).expanduser()

ROBOT_USD_PATH = Path(os.environ.get("TASK1_ROBOT_VISUAL_USD_PATH", str(TASK1_ASSETS_ROOT / "robot.usdc"))).expanduser()
OBJECT_USD_PATH = ASSETS_USD_ROOT / "assets" / "two_banana" / "two_banana_scaled_centered.usd"
PLACEMENT_USD_PATH = (
    ASSETS_USD_ROOT
    / "assets"
    / "2_floor_basket"
    / "basket_color_matched_real_zscale_1p15.usd"
)
LOCAL_GROUND_USD_PATH = ASSETS_USD_ROOT / "assets" / "Isaac" / "Grid" / "default_environment.usd"

ROBOT_INIT_POS = (-0.79251, 0.00837, 0.8)
ROBOT_INIT_ROT = (0.0, 0.0, 0.0, 1.0)  # yaw +180 deg, quaternion wxyz
OBJECT_INIT_POS = (-1.24429, 0.07554, 0.74729)
OBJECT_INIT_ROT = (0.70710678, 0.0, 0.0, 0.70710678)  # yaw +90 deg, quaternion wxyz
PLACEMENT_INIT_POS = (-1.26797, -0.17057, 0.77887)
PLACEMENT_INIT_ROT = (0.70710678, 0.70710678, 0.0, 0.0)  # roll +90 deg, quaternion wxyz

OBJECT_STATIC_FRICTION = 0.5
OBJECT_DYNAMIC_FRICTION = 0.5
OBJECT_MASS_KG = 0.02


def evaluate_walk_pick_place_replay_success(payload, cfg):
    """Evaluate Task1 from tracked object/placement state histories."""

    object_name = str(cfg["object_asset_name"])
    placement_name = str(cfg["placement_asset_name"])
    object_pose = np.asarray(payload[f"{object_name}_root_pose"], dtype=np.float64)
    object_velocity = np.asarray(payload[f"{object_name}_root_velocity"], dtype=np.float64)
    placement_pose = np.asarray(payload[f"{placement_name}_root_pose"], dtype=np.float64)
    if object_pose.ndim != 2 or object_pose.shape[1] < 3 or len(object_pose) == 0:
        raise ValueError(f"{object_name}_root_pose must have shape (T, >=3)")
    if placement_pose.ndim != 2 or placement_pose.shape[1] < 3 or len(placement_pose) != len(object_pose):
        raise ValueError(f"{placement_name}_root_pose must match {object_name}_root_pose length")
    if object_velocity.ndim != 2 or object_velocity.shape[1] < 3 or len(object_velocity) != len(object_pose):
        raise ValueError(f"{object_name}_root_velocity must match {object_name}_root_pose length")

    delta = object_pose[-1, :3] - placement_pose[-1, :3]
    speed = float(np.linalg.norm(object_velocity[-1, :3]))
    max_lift = float(np.max(object_pose[:, 2]) - object_pose[0, 2])
    checks = {
        "x": float(cfg["min_x"]) <= float(delta[0]) <= float(cfg["max_x"]),
        "y": float(cfg["min_y"]) <= float(delta[1]) <= float(cfg["max_y"]),
        "height": float(delta[2]) < float(cfg["max_height"]),
        "speed": speed < float(cfg["max_object_speed"]),
        "lift": max_lift >= float(cfg["min_object_lift"]),
    }
    return {
        "configured": True,
        "ok": all(checks.values()),
        "checks": checks,
        "reasons": [name for name, passed in checks.items() if not passed],
        "evaluated_frame": int(len(object_pose) - 1),
        "final_object_placement_delta": [float(value) for value in delta],
        "final_object_speed": speed,
        "max_object_lift": max_lift,
    }


# Evaluated after the full SONIC replay and terminal hold. This is deliberately
# not an IsaacLab termination term: reaching the basket must not stop replay.
REPLAY_SUCCESS_CFG = {
    "evaluator": evaluate_walk_pick_place_replay_success,
    "required_assets": ("object", "placement"),
    "object_asset_name": "object",
    "placement_asset_name": "placement",
    "min_x": -0.15,
    "max_x": 0.15,
    "min_y": -0.10,
    "max_y": 0.10,
    "max_height": 0.13,
    "max_object_speed": 0.10,
    "min_object_lift": 0.03,
}

SIM_EVALUATION_CFG = {
    "task_description": "walk to the table, pick up the object and put it to the basket",
    "tracked_assets": ("object", "placement"),
    "success_state_assets": ("object", "placement"),
    "reset_event_name": "reset_object_and_placement",
    "robot_init": {
        "x_range": (0.52625, 0.72625),
        "y_range": (-0.30, -0.10),
        "z": 0.8,
        "yaw_center_deg": 180.0,
        "yaw_jitter_deg": (0.0, 0.0),
    },
    "reset_pose_ranges": {
        "object_pose_range": {
            "x": (-0.10, 0.10),
            "y": (-0.10, 0.10),
            "yaw": (-math.radians(10.0), math.radians(10.0)),
        },
        "placement_pose_range": {
            "x": (-0.10, 0.10),
            "y": (-0.10, 0.10),
        },
    },
}

# Reset offsets are relative to the configured initial poses. Bounds were read
# from the source USDs and are used to reject colliding samples before write.
OBJECT_RESET_POSE_RANGE = {
    "x": (-0.17, 0.17),
    "y": (-0.1, 0.4),
    "yaw": (-math.radians(10.0), math.radians(10.0)),
}
PLACEMENT_RESET_POSE_RANGE = {
    "x": (-0.06, 0.06),
    "y": (-0.1, 0.1),
}
OBJECT_LOCAL_BOUNDS_MIN = (-0.090023, -0.048796, 0.0)
OBJECT_LOCAL_BOUNDS_MAX = (0.090023, 0.048796, 0.063783)
PLACEMENT_LOCAL_BOUNDS_MIN = (-0.151082, -0.041922, -0.125082)
PLACEMENT_LOCAL_BOUNDS_MAX = (0.150950, 0.039636, 0.137140)
OBJECT_PLACEMENT_CLEARANCE_M = 0.015


def set_object_mass_on_reset(
    env,
    env_ids,
    mass_kg: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    tag_success: bool = False,
    now_attempts: int = 0,
) -> None:
    """Set a fixed object mass while accepting this project's reset metadata."""

    del tag_success, now_attempts
    base_mdp.randomize_rigid_body_mass(
        env,
        env_ids,
        asset_cfg=asset_cfg,
        mass_distribution_params=(mass_kg, mass_kg),
        operation="abs",
        distribution="uniform",
        recompute_inertia=True,
    )


def spawn_invisible_ground_plane(prim_path, cfg, translation=None, orientation=None):
    """Spawn the standard collision plane and explicitly hide its visual hierarchy."""

    prim = sim_utils.spawn_ground_plane(prim_path, cfg, translation, orientation)
    UsdGeom.Imageable(prim).MakeInvisible()
    return prim


ENV_ACTION_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "xl330_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "d455_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_index_1_joint",
    "left_little_1_joint",
    "left_middle_1_joint",
    "left_ring_1_joint",
    "left_thumb_1_joint",
    "right_index_1_joint",
    "right_little_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_thumb_1_joint",
    "left_index_2_joint",
    "left_little_2_joint",
    "left_middle_2_joint",
    "left_ring_2_joint",
    "left_thumb_2_joint",
    "right_index_2_joint",
    "right_little_2_joint",
    "right_middle_2_joint",
    "right_ring_2_joint",
    "right_thumb_2_joint",
    "left_thumb_3_joint",
    "right_thumb_3_joint",
    "left_thumb_4_joint",
    "right_thumb_4_joint",
]

HAND_JOINT_NAMES = [
    "left_thumb_1_joint",
    "left_thumb_2_joint",
    "left_thumb_3_joint",
    "left_thumb_4_joint",
    "left_index_1_joint",
    "left_index_2_joint",
    "left_middle_1_joint",
    "left_middle_2_joint",
    "left_ring_1_joint",
    "left_ring_2_joint",
    "left_little_1_joint",
    "left_little_2_joint",
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_thumb_3_joint",
    "right_thumb_4_joint",
    "right_index_1_joint",
    "right_index_2_joint",
    "right_middle_1_joint",
    "right_middle_2_joint",
    "right_ring_1_joint",
    "right_ring_2_joint",
    "right_little_1_joint",
    "right_little_2_joint",
]

def make_robot_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_USD_PATH),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
                fix_root_link=False,
            ),
            copy_from_source=False,
            semantic_tags=[("class", "robot")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_INIT_POS,
            rot=ROBOT_INIT_ROT,
            joint_pos={
                "left_hip_yaw_joint": 0.0,
                "left_hip_roll_joint": 0.0,
                "left_hip_pitch_joint": -0.1,
                "left_knee_joint": 0.3,
                "left_ankle_pitch_joint": -0.2,
                "left_ankle_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "right_hip_pitch_joint": -0.1,
                "right_knee_joint": 0.3,
                "right_ankle_pitch_joint": -0.2,
                "right_ankle_roll_joint": 0.0,
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
                "left_shoulder_pitch_joint": 0.6,
                "left_shoulder_roll_joint": 0.9,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": -0.3,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                "right_shoulder_pitch_joint": 0.6,
                "right_shoulder_roll_joint": -0.9,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": -0.3,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                "xl330_joint": 0.0,
                "d455_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.98,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    ".*waist.*",
                ],
                effort_limit_sim={
                    ".*_hip_yaw_joint": 88.0,
                    ".*_hip_roll_joint": 139.0,
                    ".*_hip_pitch_joint": 88.0,
                    ".*_knee_joint": 139.0,
                    ".*waist_yaw_joint": 88.0,
                    ".*waist_roll_joint": 35.0,
                    ".*waist_pitch_joint": 35.0,
                },
                velocity_limit_sim={
                    ".*_hip_yaw_joint": 32.0,
                    ".*_hip_roll_joint": 20.0,
                    ".*_hip_pitch_joint": 32.0,
                    ".*_knee_joint": 20.0,
                    ".*waist_yaw_joint": 32.0,
                    ".*waist_roll_joint": 30.0,
                    ".*waist_pitch_joint": 30.0,
                },
                stiffness={
                    ".*_hip_yaw_joint": 120.0,
                    ".*_hip_roll_joint": 120.0,
                    ".*_hip_pitch_joint": 120.0,
                    ".*_knee_joint": 160.0,
                    ".*waist.*": 200.0,
                },
                damping={
                    ".*_hip_yaw_joint": 3.0,
                    ".*_hip_roll_joint": 3.0,
                    ".*_hip_pitch_joint": 3.0,
                    ".*_knee_joint": 4.0,
                    ".*waist.*": 5.0,
                },
                armature=0.01,
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit_sim=35.0,
                velocity_limit_sim=30.0,
                stiffness=40.0,
                damping=1.0,
                armature=0.01,
            ),
            "shoulders": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
                effort_limit_sim=25.0,
                velocity_limit_sim=37.0,
                stiffness=40.0,
                damping=1.0,
                armature=0.01,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"],
                effort_limit_sim=25.0,
                velocity_limit_sim=37.0,
                stiffness=40.0,
                damping=1.0,
                armature=0.01,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=[".*_wrist_yaw_joint", ".*_wrist_pitch_joint"],
                effort_limit_sim=5.0,
                velocity_limit_sim=22.0,
                stiffness=20.0,
                damping=0.5,
                armature=0.01,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=HAND_JOINT_NAMES,
                stiffness=None,
                damping=None,
                effort_limit_sim=None,
                velocity_limit_sim=None,
                armature=None,
                friction=None,
            ),
            "head": ImplicitActuatorCfg(
                joint_names_expr=["(?:^|.*_)xl330_joint$", "(?:^|.*_)d455_joint$"],
                stiffness=20.0,
                damping=4.0,
                effort_limit_sim=100.0,
                armature=0.001,
            ),
        },
    )


@configclass
class FinalWalkPickPlaceSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(
            func=spawn_invisible_ground_plane,
            usd_path=str(LOCAL_GROUND_USD_PATH),
            visible=False,
        ),
    )
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/b1_0711",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(usd_path=str(SCENE_USD_PATH)),
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_INIT_POS, rot=OBJECT_INIT_ROT),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(OBJECT_USD_PATH),
            semantic_tags=[("class", "target_object")],
        ),
    )
    placement = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/placement",
        init_state=RigidObjectCfg.InitialStateCfg(pos=PLACEMENT_INIT_POS, rot=PLACEMENT_INIT_ROT),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PLACEMENT_USD_PATH),
            semantic_tags=[("class", "target_placement")],
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    robot = make_robot_cfg()


@configclass
class FinalWalkPickPlaceActionsCfg:
    joint_position_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=ENV_ACTION_JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class FinalWalkPickPlaceObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=base_mdp.last_action)
        joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=base_mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        root_quat = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FinalWalkPickPlaceTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class FinalWalkPickPlaceEventsCfg:
    object_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (OBJECT_STATIC_FRICTION, OBJECT_STATIC_FRICTION),
            "dynamic_friction_range": (OBJECT_DYNAMIC_FRICTION, OBJECT_DYNAMIC_FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )
    object_mass = EventTerm(
        func=set_object_mass_on_reset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_kg": OBJECT_MASS_KG,
        },
    )
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_object_and_placement = EventTerm(
        func=reset_rigid_object_pair_without_overlap,
        mode="reset",
        params={
            "object_pose_range": OBJECT_RESET_POSE_RANGE,
            "placement_pose_range": PLACEMENT_RESET_POSE_RANGE,
            "object_local_bounds_min": OBJECT_LOCAL_BOUNDS_MIN,
            "object_local_bounds_max": OBJECT_LOCAL_BOUNDS_MAX,
            "placement_local_bounds_min": PLACEMENT_LOCAL_BOUNDS_MIN,
            "placement_local_bounds_max": PLACEMENT_LOCAL_BOUNDS_MAX,
            "clearance": OBJECT_PLACEMENT_CLEARANCE_M,
            "max_sampling_attempts": 64,
            "object_cfg": SceneEntityCfg("object"),
            "placement_cfg": SceneEntityCfg("placement"),
        },
    )


@configclass
class FinalWalkPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: FinalWalkPickPlaceSceneCfg = FinalWalkPickPlaceSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: FinalWalkPickPlaceObservationsCfg = FinalWalkPickPlaceObservationsCfg()
    actions: FinalWalkPickPlaceActionsCfg = FinalWalkPickPlaceActionsCfg()
    terminations: FinalWalkPickPlaceTerminationsCfg = FinalWalkPickPlaceTerminationsCfg()
    events: FinalWalkPickPlaceEventsCfg = FinalWalkPickPlaceEventsCfg()
    replay_success: dict = REPLAY_SUCCESS_CFG
    sim_evaluation: dict = SIM_EVALUATION_CFG
    env_action_joint_names: list[str] = ENV_ACTION_JOINT_NAMES

    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 60.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.0, -1.6, 1.45)
        self.viewer.lookat = (-0.6, 0.05, 0.8)
        self.viewer.asset_name = "robot"
