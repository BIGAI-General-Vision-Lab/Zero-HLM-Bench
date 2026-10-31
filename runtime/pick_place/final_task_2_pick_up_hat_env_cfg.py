from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import envs.mdp as base_mdp

from .final_task_walk_pick_place_env_cfg import (
    FinalWalkPickPlaceActionsCfg,
    FinalWalkPickPlaceEnvCfg,
    FinalWalkPickPlaceObservationsCfg,
    FinalWalkPickPlaceTerminationsCfg,
    make_robot_cfg,
    spawn_invisible_ground_plane,
)
from .task2_evaluation_contract import (
    TASK2_CABINET_FIXED_POS,
    TASK2_CABINET_FIXED_ROT_WXYZ,
    TASK2_CABINET_REPLAY_SUCCESS_CFG,
    TASK2_CABINET_SIM_EVALUATION_CFG,
)


FINAL_TASK_2_PICK_UP_HAT_TASK_ID = "Isaac-Final-Task2-PickUpHat-Abs-v0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_USD_ROOT = Path(
    os.environ.get("TASK2_EVAL_ASSETS_ROOT", str(PROJECT_ROOT / "assets/task2/usd"))
).expanduser()
ROBOT_USD_PATH = ASSETS_USD_ROOT / "robots" / "final_robot_usd_align_sonic.usd"

SCENE_USD_PATH = Path(
    os.environ.get(
        "TASK2_SCENE_USD_PATH",
        str(PROJECT_ROOT / "assets/shared/scenes/task12.usdc"),
    )
).expanduser()
OBJECT_USD_PATH = ASSETS_USD_ROOT / "assets" / "B1" / "hat_bottom_heavy_0p1kg_baked_centered.usd"
PLACEMENT_USD_PATH = ASSETS_USD_ROOT / "assets" / "B1" / "little_desk_light.usd"
HANGER_USD_PATH = ASSETS_USD_ROOT / "assets" / "B1" / "task2_guizi_rigid.usdc"

ROBOT_INIT_POS = (-0.41627, -0.33125, 0.8082)
ROBOT_INIT_ROT = (0.70710678, 0.0, 0.0, 0.70710678)  # yaw +90 deg, quaternion wxyz
OBJECT_INIT_POS = (-0.51813, 0.20505, 0.62121)
OBJECT_INIT_ROT = (0.70710678, 0.0, 0.0, 0.70710678)  # yaw +90 deg, quaternion wxyz
PLACEMENT_INIT_POS = (-0.52845, 0.23074, 0.0)
PLACEMENT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
HANGER_INIT_POS = (-0.26992, 0.18739, 0.0)
HANGER_INIT_ROT = (0.9515756268, 0.0, 0.0, -0.3074147468)  # yaw -35.807 deg, quaternion wxyz

OBJECT_STATIC_FRICTION = 1.2
OBJECT_DYNAMIC_FRICTION = 1.0

OBJECT_RESET_POSE_RANGE = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.02),
    "yaw": (-math.radians(20.0), math.radians(20.0)),
}
PLACEMENT_RESET_POSE_RANGE = {
    "yaw": (-math.radians(0.0), math.radians(0.0)),
}
HANGER_RESET_POSE_RANGE = {
    "x": (0.0, 0.20),
    "y": (0.0, 0.15),
    "yaw": (-math.radians(20.0), math.radians(20.0)),
}
ZERO_VELOCITY_RANGE: dict[str, tuple[float, float]] = {}

# Local-space bounds read from the committed task assets. Keeping them in the
# task cfg makes the replay-success contract independent of USD runtime APIs.
HAT_LOCAL_BOUNDS_MIN = (-0.1191925481, -0.1096527055, -0.0736193731)
HAT_LOCAL_BOUNDS_MAX = (0.1191925481, 0.1096527055, 0.0736193731)
HANGER_TAG_LOCAL_BOUNDS_MIN = (0.0555652654, 0.0551344643, 0.9495333338)
HANGER_TAG_LOCAL_BOUNDS_MAX = (0.0565652654, 0.0561344643, 0.9505333338)


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.maximum(norm, 1.0e-12)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    matrix = np.empty((*quat.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def evaluate_hat_hanger_replay_success(payload: dict[str, np.ndarray], cfg: dict) -> dict:
    """Evaluate whether the hanger tag is inside the hat at replay completion."""

    object_name = str(cfg.get("object_asset_name", "object"))
    hanger_name = str(cfg.get("hanger_asset_name", "hanger"))
    required = {
        "hat_pose": f"{object_name}_root_pose",
        "hanger_pose": f"{hanger_name}_root_pose",
    }
    missing = [
        dataset_name
        for dataset_name in required.values()
        if dataset_name not in payload or len(payload[dataset_name]) == 0
    ]
    if missing:
        return {
            "configured": True,
            "ok": False,
            "reasons": ["missing_hanger_state" if f"{hanger_name}_root_pose" in missing else "missing_rigid_state"],
            "missing": missing,
        }

    hat_pose = np.asarray(payload[required["hat_pose"]], dtype=np.float64)
    hanger_pose = np.asarray(payload[required["hanger_pose"]], dtype=np.float64)
    frame_count = min(len(hat_pose), len(hanger_pose))
    if frame_count < 1:
        return {
            "configured": True,
            "ok": False,
            "reasons": ["missing_final_frame"],
            "frames_available": int(frame_count),
        }

    hat_pose = hat_pose[frame_count - 1 : frame_count]
    hanger_pose = hanger_pose[frame_count - 1 : frame_count]

    tag_min = np.asarray(cfg["hanger_tag_local_bounds_min"], dtype=np.float64)
    tag_max = np.asarray(cfg["hanger_tag_local_bounds_max"], dtype=np.float64)
    tag_corners = np.asarray(
        [
            (x, y, z)
            for x in (tag_min[0], tag_max[0])
            for y in (tag_min[1], tag_max[1])
            for z in (tag_min[2], tag_max[2])
        ],
        dtype=np.float64,
    )
    hat_min = np.asarray(cfg["hat_local_bounds_min"], dtype=np.float64)
    hat_max = np.asarray(cfg["hat_local_bounds_max"], dtype=np.float64)
    tolerance = float(cfg.get("containment_tolerance_m", 1.0e-6))

    hanger_rotation = _quat_wxyz_to_matrix(hanger_pose[:, 3:7])
    hat_rotation = _quat_wxyz_to_matrix(hat_pose[:, 3:7])
    tag_corners_world = (
        np.einsum("tij,kj->tki", hanger_rotation, tag_corners)
        + hanger_pose[:, None, :3]
    )
    tag_corners_hat = np.einsum(
        "tji,tkj->tki",
        hat_rotation,
        tag_corners_world - hat_pose[:, None, :3],
    )
    contained_by_frame = np.all(
        (tag_corners_hat >= hat_min[None, None, :] - tolerance)
        & (tag_corners_hat <= hat_max[None, None, :] + tolerance),
        axis=(1, 2),
    )

    checks = {"tag_contained_final_frame": bool(contained_by_frame[0])}
    result = {
        "configured": True,
        "ok": bool(all(checks.values())),
        "reasons": [name for name, passed in checks.items() if not passed],
        "checks": checks,
        "evaluated_frame": int(frame_count - 1),
        "limits": {
            "containment_tolerance_m": tolerance,
        },
        "final_hat_position": hat_pose[-1, :3].astype(float).tolist(),
        "final_hanger_position": hanger_pose[-1, :3].astype(float).tolist(),
        "final_tag_corners_hat_local": tag_corners_hat[-1].astype(float).tolist(),
    }
    return result


HAT_HANGER_REPLAY_SUCCESS_CFG = {
    "evaluator": evaluate_hat_hanger_replay_success,
    "object_asset_name": "object",
    "hanger_asset_name": "hanger",
    "hat_local_bounds_min": HAT_LOCAL_BOUNDS_MIN,
    "hat_local_bounds_max": HAT_LOCAL_BOUNDS_MAX,
    "hanger_tag_local_bounds_min": HANGER_TAG_LOCAL_BOUNDS_MIN,
    "hanger_tag_local_bounds_max": HANGER_TAG_LOCAL_BOUNDS_MAX,
    "containment_tolerance_m": 1.0e-6,
}


def make_pick_up_hat_robot_cfg():
    robot_cfg = make_robot_cfg()
    robot_cfg.spawn.usd_path = str(ROBOT_USD_PATH)
    robot_cfg.init_state.pos = ROBOT_INIT_POS
    robot_cfg.init_state.rot = ROBOT_INIT_ROT
    return robot_cfg


def reset_hat_and_placement(
    env,
    env_ids,
    object_pose_range: dict[str, tuple[float, float]],
    placement_pose_range: dict[str, tuple[float, float]],
    hanger_pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    placement_cfg: SceneEntityCfg = SceneEntityCfg("placement"),
    hanger_cfg: SceneEntityCfg = SceneEntityCfg("hanger"),
    tag_success: bool = False,
    now_attempts: int = 0,
) -> None:
    """Reset the hat and both placement assets around their configured poses."""

    del tag_success, now_attempts
    base_mdp.reset_root_state_uniform(
        env,
        env_ids,
        pose_range=object_pose_range,
        velocity_range=velocity_range,
        asset_cfg=object_cfg,
    )
    base_mdp.reset_root_state_uniform(
        env,
        env_ids,
        pose_range=placement_pose_range,
        velocity_range=velocity_range,
        asset_cfg=placement_cfg,
    )
    base_mdp.reset_root_state_uniform(
        env,
        env_ids,
        pose_range=hanger_pose_range,
        velocity_range=velocity_range,
        asset_cfg=hanger_cfg,
    )


@configclass
class FinalTask2PickUpHatSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(func=spawn_invisible_ground_plane, visible=False),
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
    hanger = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/hanger",
        init_state=RigidObjectCfg.InitialStateCfg(pos=HANGER_INIT_POS, rot=HANGER_INIT_ROT),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(HANGER_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            semantic_tags=[("class", "target_placement")],
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    robot = make_pick_up_hat_robot_cfg()


@configclass
class FinalTask2PickUpHatEventsCfg:
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
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_object_and_placement = EventTerm(
        func=reset_hat_and_placement,
        mode="reset",
        params={
            "object_pose_range": OBJECT_RESET_POSE_RANGE,
            "placement_pose_range": PLACEMENT_RESET_POSE_RANGE,
            "hanger_pose_range": HANGER_RESET_POSE_RANGE,
            "velocity_range": ZERO_VELOCITY_RANGE,
            "object_cfg": SceneEntityCfg("object"),
            "placement_cfg": SceneEntityCfg("placement"),
            "hanger_cfg": SceneEntityCfg("hanger"),
        },
    )


@configclass
class FinalTask2PickUpHatEnvCfg(FinalWalkPickPlaceEnvCfg):
    scene: FinalTask2PickUpHatSceneCfg = FinalTask2PickUpHatSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: FinalWalkPickPlaceObservationsCfg = FinalWalkPickPlaceObservationsCfg()
    actions: FinalWalkPickPlaceActionsCfg = FinalWalkPickPlaceActionsCfg()
    terminations: FinalWalkPickPlaceTerminationsCfg = FinalWalkPickPlaceTerminationsCfg()
    events: FinalTask2PickUpHatEventsCfg = FinalTask2PickUpHatEventsCfg()
    replay_success: dict = TASK2_CABINET_REPLAY_SUCCESS_CFG
    sim_evaluation: dict = TASK2_CABINET_SIM_EVALUATION_CFG

    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = (1.5, -1.8, 1.4)
        self.viewer.lookat = (-0.2, 0.2, 0.65)
