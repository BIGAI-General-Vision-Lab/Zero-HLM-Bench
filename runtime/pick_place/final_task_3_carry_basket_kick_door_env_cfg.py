from __future__ import annotations

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse

import envs.mdp as base_mdp

from .final_task_walk_pick_place_env_cfg import (
    FinalWalkPickPlaceActionsCfg,
    FinalWalkPickPlaceEnvCfg,
    FinalWalkPickPlaceObservationsCfg,
    FinalWalkPickPlaceTerminationsCfg,
    make_robot_cfg,
    spawn_invisible_ground_plane,
)
from .task3_evaluation_contract import (
    TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG,
    TASK3_CARRY_BASKET_SIM_EVALUATION_CFG,
)


FINAL_TASK_3_CARRY_BASKET_KICK_DOOR_TASK_ID = "Isaac-Final-Task3-CarryBasketKickDoor-Abs-v0"

TASK3_DEFAULT_BUNDLE_ROOT = Path(__file__).resolve().parents[2] / "assets/task3"
ASSETS_USD_ROOT = Path(
    os.environ.get("TASK3_EVAL_ASSETS_ROOT", str(TASK3_DEFAULT_BUNDLE_ROOT / "usd"))
).expanduser()
ROBOT_USD_PATH = ASSETS_USD_ROOT / "robots" / "final_robot_usd_align_sonic_portable.usd"
SCENE_USD_PATH = Path(os.environ.get(
    "TASK3_SCENE_USD_PATH",
    str(TASK3_DEFAULT_BUNDLE_ROOT / "scene.usdc"),
)).expanduser()
OBJECT_USD_PATH = ASSETS_USD_ROOT / "assets" / "B1" / "task3_lanzi_Model_1_scaled_centered.usd"
PLACEMENT_USD_PATH = ASSETS_USD_ROOT / "assets" / "B1" / "dengzi_Model_2_scaled_centered.usd"

ROBOT_INIT_POS = (0.38661, 1.70424, 0.798)
ROBOT_INIT_ROT = (0.70710678, 0.0, 0.0, 0.70710678)  # yaw +90 deg, quaternion wxyz
OBJECT_INIT_POS = (0.41693, 2.11062, 0.7893)
OBJECT_INIT_ROT = (0.78913643, -0.03094714, -0.01438752, -0.61326909)
PLACEMENT_INIT_POS = (-0.89944, 1.4914, 0.0)
PLACEMENT_INIT_ROT = (0.04311372, 0.0, 0.0, 0.99907017)  # yaw +175.058 deg
SCENE_ROOT_BODY_INIT_POS = (0.15325, 0.10805, 1.50718)
SCENE_ROOT_BODY_INIT_ROT = (0.70710678, 0.0, 0.0, -0.70710678)

# Physics-only support matching the scanned stool seat.  The scanned mesh is
# visually correct but its irregular triangles turn the roughly 0.45 m SourceC
# drop into a high-variance bounce.  Author the helper as a child collider of
# the stool rigid body so it follows every runtime stool placement atomically.
TASK3_STOOL_CATCH_SURFACE_SIZE = (0.48, 0.44, 0.02)
TASK3_STOOL_CATCH_SURFACE_CENTER_Z_M = 0.435


def spawn_task3_stool_with_catch_surface(prim_path, cfg, translation=None, orientation=None):
    """Spawn the authored stool with an invisible, zero-restitution seat collider."""

    stool_prim = sim_utils.spawn_from_usd(
        prim_path, cfg, translation=translation, orientation=orientation
    )
    from pxr import UsdPhysics

    stage = stool_prim.GetStage()
    concrete_prim_path = stool_prim.GetPath().pathString
    rigid_prim = stage.GetPrimAtPath(f"{concrete_prim_path}/mesh")
    if not rigid_prim.IsValid() or not rigid_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Task3 stool rigid body missing at {concrete_prim_path}/mesh")
    catch_cfg = sim_utils.CuboidCfg(
        size=TASK3_STOOL_CATCH_SURFACE_SIZE,
        visible=False,
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.005,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="min",
            static_friction=1.5,
            dynamic_friction=1.2,
            restitution=0.0,
        ),
    )
    catch_cfg.func(
        f"{concrete_prim_path}/mesh/task3_catch_surface",
        catch_cfg,
        translation=(0.0, 0.0, TASK3_STOOL_CATCH_SURFACE_CENTER_Z_M),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    return stool_prim

# Reset offsets are relative to OBJECT_INIT_POS. User-provided centimeter
# ranges are represented in IsaacLab's meter units.
OBJECT_RESET_POSE_RANGE = {
    "x": (-0.10, 0.05),
    "y": (0.0, 0.10),
}
ZERO_VELOCITY_RANGE: dict[str, tuple[float, float]] = {}
DOOR_CLOSED_DEG = 62.0
DOOR_CLOSED_RAD = math.radians(DOOR_CLOSED_DEG)
DOOR_CLOSE_DURATION_S = 0.30
DOOR_CLOSE_STEP_S = 0.02
DOOR_CLOSE_STEPS = round(DOOR_CLOSE_DURATION_S / DOOR_CLOSE_STEP_S)
# Retained for the authored Action Graph metadata. Runtime generation uses the
# deterministic per-step hinge-state trajectory below because the USD drive is
# capped to 15 deg/s even when its target velocity is much higher.
DOOR_CLOSE_SPEED_DEG_S = DOOR_CLOSED_DEG / DOOR_CLOSE_DURATION_S
# Bounds of the link_1 visual geometry in the link frame. The trigger uses an
# expanded OBB so the door can react without physically colliding with robot.
DOOR_TRIGGER_BOUNDS_MIN = (-0.0166663, -1.4878581, -0.3703893)
DOOR_TRIGGER_BOUNDS_MAX = (0.2405307, -0.7924428, 0.1987664)
DOOR_TRIGGER_MARGIN_M = 0.08


def task3_replay_success_quality(payload: dict[str, np.ndarray], cfg: dict) -> dict:
    """Evaluate an isolated Task3 source replay without Task1 placement rules."""

    mode = str(cfg.get("mode", ""))
    if mode == "lift":
        pose = np.asarray(payload.get("object_root_pose", ()), dtype=np.float32)
        velocity = np.asarray(payload.get("object_root_velocity", ()), dtype=np.float32)
        if pose.ndim != 2 or pose.shape[1] < 3 or velocity.ndim != 2 or velocity.shape[1] < 3:
            return {"configured": True, "mode": mode, "ok": False, "reasons": ["missing_object_state"]}
        tail_frames = min(int(cfg.get("tail_frames", 15)), len(pose), len(velocity))
        min_lift_m = float(cfg.get("min_lift_m", 0.10))
        max_tail_z_range_m = float(cfg.get("max_tail_z_range_m", 0.03))
        max_tail_speed_m_s = float(cfg.get("max_tail_speed_m_s", 0.20))
        lift = pose[:, 2] - float(pose[0, 2])
        tail_lift_min = float(np.min(lift[-tail_frames:]))
        tail_z_range = float(np.ptp(pose[-tail_frames:, 2]))
        tail_speed_max = float(np.max(np.linalg.norm(velocity[-tail_frames:, :3], axis=1)))
        checks = {
            "lift_held": tail_lift_min >= min_lift_m,
            "height_stable": tail_z_range <= max_tail_z_range_m,
            "speed_stable": tail_speed_max <= max_tail_speed_m_s,
        }
        return {
            "configured": True,
            "mode": mode,
            "ok": bool(all(checks.values())),
            "reasons": [name for name, passed in checks.items() if not passed],
            "checks": checks,
            "initial_object_z": float(pose[0, 2]),
            "max_object_lift_m": float(np.max(lift)),
            "tail_lift_min_m": tail_lift_min,
            "tail_z_range_m": tail_z_range,
            "tail_speed_max_m_s": tail_speed_max,
            "limits": {
                "min_lift_m": min_lift_m,
                "tail_frames": tail_frames,
                "max_tail_z_range_m": max_tail_z_range_m,
                "max_tail_speed_m_s": max_tail_speed_m_s,
            },
        }

    if mode == "door_close":
        position = np.asarray(payload.get("door_joint_position", ()), dtype=np.float32).reshape(-1)
        velocity = np.asarray(payload.get("door_joint_velocity", ()), dtype=np.float32).reshape(-1)
        if position.size == 0:
            return {"configured": True, "mode": mode, "ok": False, "reasons": ["missing_door_joint_state"]}
        min_closed_rad = math.radians(float(cfg.get("min_closed_deg", 50.0)))
        min_trigger_delta_rad = math.radians(float(cfg.get("min_trigger_delta_deg", 5.0)))
        dwell_frames = min(int(cfg.get("dwell_frames", 10)), len(position))
        reached = position >= min_closed_rad
        close_delta = float(np.max(position) - position[0])
        checks = {
            "close_triggered": close_delta >= min_trigger_delta_rad,
            "closed_angle_reached": bool(np.any(reached)),
            "closed_dwell": bool(np.all(reached[-dwell_frames:])),
        }
        return {
            "configured": True,
            "mode": mode,
            # Task3's semantic success is the SourceB-triggered close event.
            # Reaching the visual closed angle is diagnostic because the
            # remainder of the close is a deterministic simulated animation.
            "ok": bool(checks["close_triggered"]),
            "reasons": [] if checks["close_triggered"] else ["close_triggered"],
            "checks": checks,
            "initial_door_deg": math.degrees(float(position[0])),
            "close_delta_deg": math.degrees(close_delta),
            "max_door_deg": math.degrees(float(np.max(position))),
            "final_door_deg": math.degrees(float(position[-1])),
            "final_door_velocity_deg_s": math.degrees(float(velocity[-1])) if velocity.size else None,
            "limits": {
                "min_trigger_delta_deg": math.degrees(min_trigger_delta_rad),
                "min_closed_deg": math.degrees(min_closed_rad),
                "dwell_frames": dwell_frames,
            },
        }

    return {"configured": True, "mode": mode, "ok": False, "reasons": ["unknown_task3_success_mode"]}


def spawn_task3_scene_at_authored_pose(prim_path, cfg, translation=None, orientation=None):
    """Keep authored transforms and bind the embedded contact graph to this env instance."""

    del translation, orientation
    scene_prim = sim_utils.spawn_from_usd(
        prim_path,
        cfg,
        translation=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )

    # Absolute relationship targets authored in a referenced USD are not a
    # reliable contract after IsaacLab namespaces/clones the asset. Bind both
    # Action Graph nodes to the concrete path used by this environment.
    from pxr import Sdf, Usd, UsdPhysics

    stage = scene_prim.GetStage()
    concrete_prim_path = scene_prim.GetPath().pathString
    contact_node = stage.GetPrimAtPath(f"{concrete_prim_path}/AutoOpenGraph/OnDoorContact")
    speed_node = stage.GetPrimAtPath(f"{concrete_prim_path}/AutoOpenGraph/OpenSpeed")
    velocity_node = stage.GetPrimAtPath(f"{concrete_prim_path}/AutoOpenGraph/WriteTargetVelocity")
    if not contact_node.IsValid() or not speed_node.IsValid() or not velocity_node.IsValid():
        raise RuntimeError(f"Task3 door Action Graph is missing below {concrete_prim_path}")
    contact_node.GetRelationship("inputs:targetBodies").SetTargets(
        [Sdf.Path(f"{concrete_prim_path}/link_1")]
    )
    velocity_node.GetRelationship("inputs:prim").SetTargets(
        [Sdf.Path(f"{concrete_prim_path}/joints/joint_1")]
    )
    speed_node.GetAttribute("inputs:value").Set(DOOR_CLOSE_SPEED_DEG_S)

    # Contact is detected geometrically below. Disable every collision shape
    # under the moving door so closing it never pushes or destabilizes robot.
    door_link_prim = stage.GetPrimAtPath(f"{concrete_prim_path}/link_1")
    if not door_link_prim.IsValid():
        raise RuntimeError(f"Task3 door link is missing below {concrete_prim_path}")
    for prim in Usd.PrimRange(door_link_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False).Set(False)
    return scene_prim


def reset_task3_door_auto_close(env, env_ids, tag_success: bool = False, now_attempts: int = 0):
    """Reset the contact latch and stop the door drive."""

    del tag_success, now_attempts
    door = env.scene["scene"]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(door.num_instances, device=door.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=door.device, dtype=torch.long)
    if not hasattr(env, "_task3_door_contact_latched"):
        env._task3_door_contact_latched = torch.zeros(
            door.num_instances, device=door.device, dtype=torch.bool
        )
    env._task3_door_contact_latched[env_ids] = False
    if not hasattr(env, "_task3_door_close_step"):
        env._task3_door_close_step = torch.full(
            (door.num_instances,), -1, device=door.device, dtype=torch.long
        )
    if not hasattr(env, "_task3_door_close_start_position"):
        env._task3_door_close_start_position = torch.zeros(
            door.num_instances, device=door.device, dtype=door.data.joint_pos.dtype
        )
    env._task3_door_close_step[env_ids] = -1
    env._task3_door_close_start_position[env_ids] = 0.0
    if hasattr(env, "_task3_door_was_touching"):
        env._task3_door_was_touching[env_ids] = False

    velocity_targets = torch.zeros(
        (len(env_ids), door.num_joints), device=door.device, dtype=door.data.joint_vel.dtype
    )
    door.root_physx_view.set_dof_velocity_targets(velocity_targets, env_ids)


def close_task3_door_on_geometric_overlap(
    env,
    env_ids,
    bounds_min: tuple[float, float, float] = DOOR_TRIGGER_BOUNDS_MIN,
    bounds_max: tuple[float, float, float] = DOOR_TRIGGER_BOUNDS_MAX,
    margin_m: float = DOOR_TRIGGER_MARGIN_M,
    closed_position_rad: float = DOOR_CLOSED_RAD,
    close_steps: int = DOOR_CLOSE_STEPS,
    close_duration_s: float = DOOR_CLOSE_DURATION_S,
):
    """Latch the door close command when a robot link enters its visual OBB."""

    del env_ids
    door = env.scene["scene"]
    robot = env.scene["robot"]
    door_body_ids, _ = door.find_bodies("link_1")
    if len(door_body_ids) != 1:
        raise RuntimeError(f"Expected one Task3 door link_1 body, found {door_body_ids}")

    door_pos_w = door.data.body_pos_w[:, door_body_ids[0]]
    door_quat_w = door.data.body_quat_w[:, door_body_ids[0]]
    robot_pos_w = robot.data.body_pos_w
    relative_pos_w = robot_pos_w - door_pos_w.unsqueeze(1)
    num_envs, num_robot_bodies, _ = relative_pos_w.shape
    relative_pos_door = quat_rotate_inverse(
        door_quat_w.unsqueeze(1).expand(-1, num_robot_bodies, -1).reshape(-1, 4),
        relative_pos_w.reshape(-1, 3),
    ).reshape(num_envs, num_robot_bodies, 3)
    lower = torch.as_tensor(bounds_min, device=door.device, dtype=relative_pos_door.dtype) - margin_m
    upper = torch.as_tensor(bounds_max, device=door.device, dtype=relative_pos_door.dtype) + margin_m
    raw_touching = torch.all((relative_pos_door >= lower) & (relative_pos_door <= upper), dim=-1).any(dim=-1)
    if not hasattr(env, "_task3_door_was_touching"):
        env._task3_door_was_touching = torch.zeros_like(raw_touching)
    require_rising_edge = getattr(env, "_task3_door_trigger_require_rising_edge", None)
    if require_rising_edge is None:
        touching = raw_touching
    else:
        require_rising_edge = require_rising_edge.to(
            device=raw_touching.device, dtype=torch.bool
        )
        touching = torch.where(
            require_rising_edge,
            raw_touching & ~env._task3_door_was_touching,
            raw_touching,
        )
    env._task3_door_was_touching[:] = raw_touching
    # Task3 generation may explicitly arm the geometric trigger only for the
    # SourceB semantic window.  This prevents SourceA/WalkA proximity from
    # closing the door while preserving the original always-armed behavior for
    # every caller that does not opt into the per-environment gate.
    trigger_enabled = getattr(env, "_task3_door_trigger_enabled", None)
    if trigger_enabled is not None:
        touching &= trigger_enabled.to(device=touching.device, dtype=torch.bool)
    if not hasattr(env, "_task3_door_contact_latched"):
        env._task3_door_contact_latched = torch.zeros_like(touching)
    newly_triggered = touching & ~env._task3_door_contact_latched
    env._task3_door_contact_latched |= touching
    if torch.any(newly_triggered):
        triggered_ids = torch.nonzero(newly_triggered).flatten().detach().cpu().tolist()
        print(
            "[TASK3_DOOR] geometric overlap detected: "
            f"env_ids={triggered_ids} margin_m={margin_m:.3f} "
            f"closing_to={math.degrees(closed_position_rad):.0f}deg "
            f"duration={close_duration_s:.2f}s",
            flush=True,
        )
    joint_ids, _ = door.find_joints("joint_1")
    joint_id = joint_ids[0]
    if not hasattr(env, "_task3_door_close_step"):
        env._task3_door_close_step = torch.full(
            (door.num_instances,), -1, device=door.device, dtype=torch.long
        )
    if not hasattr(env, "_task3_door_close_start_position"):
        env._task3_door_close_start_position = torch.zeros(
            door.num_instances, device=door.device, dtype=door.data.joint_pos.dtype
        )
    current_position = door.data.joint_pos[:, joint_id]
    env._task3_door_close_start_position[newly_triggered] = current_position[newly_triggered]
    env._task3_door_close_step[newly_triggered] = 0

    # A restored checkpoint may already have a latched, fully closed door but
    # no transient ramp counter. Initialize that state without reopening it.
    restored = env._task3_door_contact_latched & (env._task3_door_close_step < 0)
    if torch.any(restored):
        env._task3_door_close_start_position[restored] = current_position[restored]
        already_closed = restored & (current_position >= closed_position_rad - math.radians(0.1))
        env._task3_door_close_step[already_closed] = int(close_steps)
        env._task3_door_close_step[restored & ~already_closed] = 0

    latched = env._task3_door_contact_latched
    if torch.any(latched):
        step = torch.clamp(env._task3_door_close_step, 0, int(close_steps)).to(
            dtype=door.data.joint_pos.dtype
        )
        u = step / float(close_steps)
        blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        blend_rate = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / float(
            close_duration_s
        )
        start = env._task3_door_close_start_position
        delta = float(closed_position_rad) - start
        position = door.data.joint_pos.clone()
        velocity = torch.zeros_like(door.data.joint_vel)
        position[latched, joint_id] = start[latched] + delta[latched] * blend[latched]
        velocity[latched, joint_id] = delta[latched] * blend_rate[latched]
        door.write_joint_state_to_sim(position, velocity)
        env._task3_door_close_step[latched] = torch.clamp(
            env._task3_door_close_step[latched] + 1, max=int(close_steps)
        )

    # Keep the unused USD drive neutral so it cannot fight the deterministic
    # state trajectory.
    velocity_targets = torch.zeros(
        (door.num_instances, door.num_joints), device=door.device, dtype=door.data.joint_vel.dtype
    )
    door.root_physx_view.set_dof_velocity_targets(velocity_targets, door._ALL_INDICES)

# Captured from demo_0 frame 0 in pick_up_hat/teleop0731_2/wbc_collect.hdf5.
# Keep Task3's configured root pose while reproducing the complete settled
# robot joint state, including its natural arms-down and open-hand posture.
TASK3_STANDING_JOINT_POS = {
    "left_hip_pitch_joint": -0.067650437,
    "right_hip_pitch_joint": -0.014477526,
    "waist_yaw_joint": 0.064558089,
    "left_hip_roll_joint": 0.034458514,
    "right_hip_roll_joint": -0.104016848,
    "waist_roll_joint": -0.030864615,
    "left_hip_yaw_joint": 0.083896466,
    "right_hip_yaw_joint": -0.069409050,
    "waist_pitch_joint": 0.010036993,
    "left_knee_joint": 0.202888578,
    "right_knee_joint": 0.194438323,
    "left_shoulder_pitch_joint": 0.023068236,
    "right_shoulder_pitch_joint": -0.008304247,
    "xl330_joint": 0.0,
    "left_ankle_pitch_joint": -0.130019248,
    "right_ankle_pitch_joint": -0.171314180,
    "left_shoulder_roll_joint": 0.248610616,
    "right_shoulder_roll_joint": -0.273516744,
    "d455_joint": 0.0,
    "left_ankle_roll_joint": -0.024685144,
    "right_ankle_roll_joint": 0.076958515,
    "left_shoulder_yaw_joint": -0.297382236,
    "right_shoulder_yaw_joint": 0.398750454,
    "left_elbow_joint": 1.261237383,
    "right_elbow_joint": 1.269955039,
    "left_wrist_roll_joint": 0.033396158,
    "right_wrist_roll_joint": 0.161586136,
    "left_wrist_pitch_joint": -0.016504092,
    "right_wrist_pitch_joint": -0.016800383,
    "left_wrist_yaw_joint": 0.053682372,
    "right_wrist_yaw_joint": 0.044152264,
    "left_index_1_joint": 0.071914867,
    "left_little_1_joint": 0.071911193,
    "left_middle_1_joint": 0.071916603,
    "left_ring_1_joint": 0.071914293,
    "left_thumb_1_joint": 0.058295421,
    "right_index_1_joint": 0.071900032,
    "right_little_1_joint": 0.071900487,
    "right_middle_1_joint": 0.071898781,
    "right_ring_1_joint": 0.071899615,
    "right_thumb_1_joint": 0.058268145,
    "left_index_2_joint": 0.087073535,
    "left_little_2_joint": 0.090019532,
    "left_middle_2_joint": 0.079382822,
    "left_ring_2_joint": 0.081486814,
    "left_thumb_2_joint": 0.029589474,
    "right_index_2_joint": 0.083781153,
    "right_little_2_joint": 0.085271485,
    "right_middle_2_joint": 0.075816758,
    "right_ring_2_joint": 0.078053825,
    "right_thumb_2_joint": 0.029586880,
    "left_thumb_3_joint": 0.023887983,
    "right_thumb_3_joint": 0.023888800,
    "left_thumb_4_joint": 0.022552168,
    "right_thumb_4_joint": 0.022561252,
}


def make_task3_robot_cfg():
    robot_cfg = make_robot_cfg()
    robot_cfg.spawn.usd_path = str(ROBOT_USD_PATH)
    robot_cfg.init_state.pos = ROBOT_INIT_POS
    robot_cfg.init_state.rot = ROBOT_INIT_ROT
    robot_cfg.init_state.joint_pos.update(TASK3_STANDING_JOINT_POS)
    return robot_cfg


@configclass
class FinalTask3CarryBasketKickDoorSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.015)),
        spawn=sim_utils.GroundPlaneCfg(func=spawn_invisible_ground_plane, visible=False),
    )
    scene = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/b1_0730_door",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=SCENE_ROOT_BODY_INIT_POS,
            rot=SCENE_ROOT_BODY_INIT_ROT,
            joint_pos={"joint_1": 0.0},
            joint_vel={"joint_1": 0.0},
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SCENE_USD_PATH),
            func=spawn_task3_scene_at_authored_pose,
            activate_contact_sensors=False,
        ),
        # The embedded Action Graph writes the USD drive target velocity when
        # link_1 is contacted. Do not add an IsaacLab actuator here: its
        # per-step zero target would overwrite that one-shot graph command.
        actuators={},
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_INIT_POS, rot=OBJECT_INIT_ROT),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(OBJECT_USD_PATH),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            semantic_tags=[("class", "target_object")],
        ),
    )
    placement = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/placement",
        init_state=RigidObjectCfg.InitialStateCfg(pos=PLACEMENT_INIT_POS, rot=PLACEMENT_INIT_ROT),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PLACEMENT_USD_PATH),
            semantic_tags=[("class", "target_placement")],
            func=spawn_task3_stool_with_catch_surface,
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    robot = make_task3_robot_cfg()


@configclass
class FinalTask3CarryBasketKickDoorEventsCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_door_auto_close = EventTerm(func=reset_task3_door_auto_close, mode="reset")
    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": OBJECT_RESET_POSE_RANGE,
            "velocity_range": ZERO_VELOCITY_RANGE,
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    close_door_on_geometric_overlap = EventTerm(
        func=close_task3_door_on_geometric_overlap,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=True,
    )


@configclass
class FinalTask3CarryBasketKickDoorEnvCfg(FinalWalkPickPlaceEnvCfg):
    scene: FinalTask3CarryBasketKickDoorSceneCfg = FinalTask3CarryBasketKickDoorSceneCfg(
        num_envs=1,
        env_spacing=10.0,
        replicate_physics=True,
    )
    observations: FinalWalkPickPlaceObservationsCfg = FinalWalkPickPlaceObservationsCfg()
    actions: FinalWalkPickPlaceActionsCfg = FinalWalkPickPlaceActionsCfg()
    terminations: FinalWalkPickPlaceTerminationsCfg = FinalWalkPickPlaceTerminationsCfg()
    events: FinalTask3CarryBasketKickDoorEventsCfg = FinalTask3CarryBasketKickDoorEventsCfg()
    replay_success: dict = TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG
    sim_evaluation: dict = TASK3_CARRY_BASKET_SIM_EVALUATION_CFG

    # wbc_collect_any2_isaaclab_policy.py records this articulation body with
    # rigid-object-compatible HDF5 keys for later replay and task evaluation.
    recorded_scene_bodies: dict = {
        "door_link": {"asset_name": "scene", "body_name": "link_1"},
    }

    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = (-1.2, -0.6, 1.6)
        self.viewer.lookat = (-1.2, 1.7, 0.8)
