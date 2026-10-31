"""Pure initialization and success contract for Task3 carry-basket evaluation."""

from __future__ import annotations

import math

import numpy as np


TASK3_CARRY_BASKET_PROMPT = "carry the basket through the door and place it down"

# These ranges document the successful 20260817 generation campaign.  Runtime
# manifests use correlated rows from the immutable 386-scenario bank instead
# of independently sampling the marginal ranges below.
TASK3_ROBOT_INIT_CFG = {
    "x_range": (0.2627, 0.5472),
    "y_range": (1.5932, 1.7507),
    "z": 0.798,
    "yaw_center_deg": 90.0,
    "yaw_jitter_deg": (-4.0, 4.0),
}
TASK3_OBJECT_RELATIVE_POS_CFG = {
    "x_range": (-0.1041, 0.0762),
    "y_range": (0.4361, 0.5377),
    "z_range": (-0.0196, -0.0169),
}
TASK3_WALK2_ENDPOINT_CFG = {
    "center_xy": (-1.46653, 1.483),
    "full_size_xy": (0.51644, 0.3966),
    "yaw_range_rad": (-math.pi, math.radians(-165.0)),
}
TASK3_PLACEMENT_EMPIRICAL_CFG = {
    "source": "assets/evaluation/task3_training_scenarios_386_full_init.jsonl",
    "x_range": (-2.93739510, -1.40592766),
    "y_range": (1.02979279, 1.77414322),
    "z": 0.0,
    "correlated_with_robot_object_scene": True,
}
TASK3_SCENE_X_RANGE_M = (-0.46659661, 0.00591469)


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1.0e-12)
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


def evaluate_task3_carry_basket_success(payload: dict[str, np.ndarray], cfg: dict) -> dict:
    """Require a triggered door and a released basket settled on the stool."""

    required = (
        "object_root_pose",
        "object_root_velocity",
        "placement_root_pose",
        "door_joint_position",
        "door_contact_latched",
    )
    missing = [name for name in required if name not in payload or len(payload[name]) == 0]
    if missing:
        return {"ok": False, "checks": {}, "reasons": ["missing_task3_state"], "missing": missing}

    basket_pose = np.asarray(payload["object_root_pose"], dtype=np.float64)
    basket_velocity = np.asarray(payload["object_root_velocity"], dtype=np.float64)
    stool_pose = np.asarray(payload["placement_root_pose"], dtype=np.float64)
    hinge = np.asarray(payload["door_joint_position"], dtype=np.float64).reshape(-1)
    latched = np.asarray(payload["door_contact_latched"], dtype=bool).reshape(-1)
    frame_count = min(len(basket_pose), len(basket_velocity), len(stool_pose), len(hinge), len(latched))
    requested_stable_frames = int(cfg.get("stable_frames", 10))
    stable_frames = min(requested_stable_frames, frame_count)
    tail = slice(frame_count - stable_frames, frame_count)

    stool_rotation = _quat_wxyz_to_matrix(stool_pose[frame_count - 1, 3:7])
    basket_local = stool_rotation.T @ (
        basket_pose[frame_count - 1, :3] - stool_pose[frame_count - 1, :3]
    )
    half_xy = np.asarray(cfg["stool_seat_half_xy_m"], dtype=np.float64)
    inside = bool(np.all(np.abs(basket_local[:2]) <= half_xy))
    basket_rotation = _quat_wxyz_to_matrix(basket_pose[frame_count - 1, 3:7])
    local_min = np.asarray(cfg["basket_local_min_m"], dtype=np.float64)
    local_max = np.asarray(cfg["basket_local_max_m"], dtype=np.float64)
    corners = np.asarray(
        [[x, y, z] for x in (local_min[0], local_max[0])
         for y in (local_min[1], local_max[1])
         for z in (local_min[2], local_max[2])],
        dtype=np.float64,
    )
    basket_bottom = float(
        basket_pose[frame_count - 1, 2] + np.min((basket_rotation @ corners.T).T[:, 2])
    )
    seat_z = float(stool_pose[frame_count - 1, 2] + float(cfg["stool_seat_z_m"]))
    visual_bottom_supported = abs(basket_bottom - seat_z) <= float(cfg["seat_height_tolerance_m"])
    collision_hull_supported = bool(
        seat_z + 0.02 <= basket_pose[frame_count - 1, 2] <= seat_z + 0.35
    )
    supported = bool(visual_bottom_supported or collision_hull_supported)
    linear_speed = np.linalg.norm(basket_velocity[tail, :3], axis=1)
    angular_speed = np.linalg.norm(basket_velocity[tail, 3:6], axis=1)
    z_range = float(np.ptp(basket_pose[tail, 2])) if stable_frames else math.inf
    hinge_delta = float(np.max(hinge[:frame_count]) - hinge[0])
    triggered = bool(np.any(latched[:frame_count])) or hinge_delta >= math.radians(
        float(cfg["minimum_door_trigger_delta_deg"])
    )
    max_object_lift = float(np.max(basket_pose[:frame_count, 2]) - basket_pose[0, 2])

    checks = {
        "basket_center_inside_stool_seat": inside,
        "basket_supported_by_stool": supported,
        "linear_speed_settled": bool(linear_speed[-1] <= float(cfg["maximum_linear_speed_mps"])),
        "angular_speed_settled": bool(angular_speed[-1] <= float(cfg["maximum_angular_speed_radps"])),
        "height_settled": z_range <= float(cfg["maximum_tail_z_range_m"]),
        "stable_window_complete": frame_count >= requested_stable_frames,
        "door_triggered": triggered,
    }
    return {
        "ok": bool(all(checks.values())),
        "checks": checks,
        "reasons": [name for name, passed in checks.items() if not passed],
        "evaluated_frame": int(frame_count - 1),
        "stable_frames": int(stable_frames),
        "basket_local_xy_m": basket_local[:2].astype(float).tolist(),
        "basket_bottom_minus_seat_m": float(basket_bottom - seat_z),
        "basket_visual_bottom_supported": bool(visual_bottom_supported),
        "basket_collision_hull_supported": bool(collision_hull_supported),
        "basket_final_linear_speed_mps": float(linear_speed[-1]),
        "basket_final_angular_speed_radps": float(angular_speed[-1]),
        "basket_tail_z_range_m": z_range,
        "door_trigger_delta_deg": math.degrees(hinge_delta),
        "max_object_lift": max_object_lift,
        "door_final_deg": math.degrees(float(hinge[frame_count - 1])),
        "door_closed_angle_reached": bool(
            np.any(hinge[:frame_count] >= math.radians(float(cfg["diagnostic_closed_door_deg"])))
        ),
    }


TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG = {
    "evaluator": evaluate_task3_carry_basket_success,
    "required_assets": ("object", "placement", "scene"),
    "stool_seat_half_xy_m": (0.24, 0.22),
    "basket_local_min_m": (-0.15721323, -0.17953092, 0.0),
    "basket_local_max_m": (0.15721323, 0.17953092, 0.11761208),
    "stool_seat_z_m": 0.435,
    "seat_height_tolerance_m": 0.04,
    "maximum_linear_speed_mps": 0.05,
    "maximum_angular_speed_radps": 0.50,
    "maximum_tail_z_range_m": 0.01,
    "minimum_door_trigger_delta_deg": 5.0,
    "diagnostic_closed_door_deg": 50.0,
    "stable_frames": 10,
}


TASK3_CARRY_BASKET_SIM_EVALUATION_CFG = {
    "task_description": TASK3_CARRY_BASKET_PROMPT,
    "tracked_assets": ("object", "placement", "scene"),
    "success_state_assets": ("object", "placement"),
    "reset_event_name": "reset_object",
    "robot_init": TASK3_ROBOT_INIT_CFG,
    # Task3's adapter owns correlated robot/object initialization and scene
    # articulation reset; do not inject incompatible parameters into the
    # collection-time reset_root_state_uniform event.
    "reset_pose_ranges": {},
    "object_relative_pos": TASK3_OBJECT_RELATIVE_POS_CFG,
    "walk2_endpoint": TASK3_WALK2_ENDPOINT_CFG,
    "placement": TASK3_PLACEMENT_EMPIRICAL_CFG,
    "scene_x_range_m": TASK3_SCENE_X_RANGE_M,
}
