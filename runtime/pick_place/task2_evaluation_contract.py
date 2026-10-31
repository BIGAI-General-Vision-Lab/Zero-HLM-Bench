"""Pure success and initialization contract for Task2 cabinet evaluation."""

from __future__ import annotations

import numpy as np


TASK2_CABINET_PROMPT = "pick up the hat and place it on the cabinet"
TASK2_CABINET_FIXED_POS = (0.85, 1.32666, 0.0)
TASK2_CABINET_FIXED_ROT_WXYZ = (0.9515756268, 0.0, 0.0, -0.3074147468)

# Robust center of the 827-episode training distribution, with a deliberately
# small evaluation rectangle and fixed heading.
TASK2_ROBOT_INIT_CFG = {
    "x_range": (-0.462561, -0.362561),
    "y_range": (-0.368796, -0.268796),
    "z": 0.781229,
    "yaw_center_deg": 90.0,
    "yaw_jitter_deg": (0.0, 0.0),
}


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


def evaluate_task2_cabinet_success(payload: dict[str, np.ndarray], cfg: dict) -> dict:
    """Require the released hat to remain settled in the cabinet target area."""

    required = (
        "object_root_pose",
        "object_root_velocity",
        "hanger_root_pose",
    )
    missing = [name for name in required if name not in payload or len(payload[name]) == 0]
    if missing:
        return {"ok": False, "checks": {}, "reasons": ["missing_rigid_state"], "missing": missing}

    hat_pose = np.asarray(payload["object_root_pose"], dtype=np.float64)
    hat_velocity = np.asarray(payload["object_root_velocity"], dtype=np.float64)
    cabinet_pose = np.asarray(payload["hanger_root_pose"], dtype=np.float64)
    frame_count = min(len(hat_pose), len(hat_velocity), len(cabinet_pose))
    stable_frames = min(int(cfg.get("stable_frames", 10)), frame_count)
    tail = slice(frame_count - stable_frames, frame_count)

    cabinet_rotation = _quat_wxyz_to_matrix(cabinet_pose[tail, 3:7])
    hat_local = np.einsum(
        "tji,tj->ti",
        cabinet_rotation,
        hat_pose[tail, :3] - cabinet_pose[tail, :3],
    )
    lower = np.asarray(cfg["target_local_xy_lower_m"], dtype=np.float64)
    upper = np.asarray(cfg["target_local_xy_upper_m"], dtype=np.float64)
    inside = np.all((hat_local[:, :2] >= lower) & (hat_local[:, :2] <= upper), axis=1)
    above = hat_pose[tail, 2] >= float(cfg["minimum_hat_z_m"])
    linear_speed = np.linalg.norm(hat_velocity[tail, :3], axis=1)
    angular_speed = np.linalg.norm(hat_velocity[tail, 3:6], axis=1)
    linear_settled = linear_speed <= float(cfg["maximum_linear_speed_mps"])
    angular_settled = angular_speed <= float(cfg["maximum_angular_speed_radps"])
    max_hat_lift = float(np.max(hat_pose[:frame_count, 2] - hat_pose[0, 2]))

    checks = {
        "inside_left_top_empty_region_xy": bool(np.all(inside)),
        "above_minimum_z": bool(np.all(above)),
        "linear_speed_settled": bool(np.all(linear_settled)),
        "angular_speed_settled": bool(np.all(angular_settled)),
        "stable_window_complete": frame_count >= int(cfg.get("stable_frames", 10)),
    }
    return {
        "ok": bool(all(checks.values())),
        "checks": checks,
        "reasons": [name for name, passed in checks.items() if not passed],
        "evaluated_frame": int(frame_count - 1),
        "stable_frames": int(stable_frames),
        "hat_local_xy_m": hat_local[-1, :2].astype(float).tolist(),
        "hat_world_z_m": float(hat_pose[frame_count - 1, 2]),
        "max_object_lift": max_hat_lift,
        "hat_final_linear_speed_mps": float(linear_speed[-1]),
        "hat_final_angular_speed_radps": float(angular_speed[-1]),
    }


TASK2_CABINET_REPLAY_SUCCESS_CFG = {
    "evaluator": evaluate_task2_cabinet_success,
    "required_assets": ("object", "hanger"),
    "target_local_xy_lower_m": (-0.20, -0.22),
    "target_local_xy_upper_m": (0.0, 0.02),
    "minimum_hat_z_m": 0.5,
    "maximum_linear_speed_mps": 0.02,
    "maximum_angular_speed_radps": 0.25,
    "stable_frames": 10,
}


TASK2_CABINET_SIM_EVALUATION_CFG = {
    "task_description": TASK2_CABINET_PROMPT,
    "tracked_assets": ("object", "placement", "hanger"),
    "success_state_assets": ("object", "hanger"),
    "reset_event_name": "reset_object_and_placement",
    "robot_init": TASK2_ROBOT_INIT_CFG,
    "reset_pose_ranges": {
        "object_pose_range": {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.02),
            "yaw": (-0.3490658503988659, 0.3490658503988659),
        },
        "placement_pose_range": {"yaw": (0.0, 0.0)},
        "hanger_pose_range": {},
    },
}
