"""Reference and actuator-target stabilization for crouched SONIC teleoperation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


QPOS_WAIST_ROLL_INDEX = 20
QPOS_WAIST_PITCH_INDEX = 21


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-12:
        raise ValueError("Cannot normalize a zero quaternion")
    quat = quat / norm
    return -quat if quat[0] < 0.0 else quat


def _quat_to_rpy_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = _normalize_quat_wxyz(quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return _normalize_quat_wxyz(
        np.asarray(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )
    )


def stabilize_reference_qpos(
    qpos: np.ndarray,
    *,
    root_tilt_limit_rad: float,
    waist_limit_rad: float,
) -> np.ndarray:
    """Limit pelvis tilt and transfer the removed tilt to the waist joints.

    The transfer preserves the first-order torso roll/pitch represented by
    ``root tilt + waist angle`` while avoiding the large pelvis tilt that makes
    SONIC drive its waist target into the soft joint limit during crouching.
    """

    source = np.asarray(qpos)
    if source.shape[-1] != 36:
        raise ValueError(f"Expected qpos shape [...,36], got {source.shape}")
    if root_tilt_limit_rad <= 0.0 or waist_limit_rad <= 0.0:
        raise ValueError("Reference stabilization limits must be positive")

    flat = source.astype(np.float64, copy=True).reshape(-1, 36)
    for row in flat:
        roll, pitch, yaw = _quat_to_rpy_wxyz(row[3:7])
        limited_roll = float(np.clip(roll, -root_tilt_limit_rad, root_tilt_limit_rad))
        limited_pitch = float(np.clip(pitch, -root_tilt_limit_rad, root_tilt_limit_rad))
        row[3:7] = _rpy_to_quat_wxyz(limited_roll, limited_pitch, yaw)
        row[QPOS_WAIST_ROLL_INDEX] = np.clip(
            row[QPOS_WAIST_ROLL_INDEX] + (roll - limited_roll),
            -waist_limit_rad,
            waist_limit_rad,
        )
        row[QPOS_WAIST_PITCH_INDEX] = np.clip(
            row[QPOS_WAIST_PITCH_INDEX] + (pitch - limited_pitch),
            -waist_limit_rad,
            waist_limit_rad,
        )
    return flat.reshape(source.shape).astype(source.dtype if np.issubdtype(source.dtype, np.floating) else np.float32)


@dataclass
class WaistTargetStabilizer:
    """Stateful low-pass, slew-rate, and range limiter for SONIC waist targets."""

    joint_names: Sequence[str]
    control_hz: float
    pitch_limit_rad: float
    roll_limit_rad: float
    filter_alpha: float
    rate_limit_rad_s: float

    def __post_init__(self) -> None:
        if self.control_hz <= 0.0:
            raise ValueError("control_hz must be positive")
        if not 0.0 < self.filter_alpha <= 1.0:
            raise ValueError("filter_alpha must be in (0, 1]")
        if self.pitch_limit_rad <= 0.0 or self.roll_limit_rad <= 0.0 or self.rate_limit_rad_s <= 0.0:
            raise ValueError("Waist target limits must be positive")
        names = list(self.joint_names)
        self._pitch_index = names.index("waist_pitch_joint")
        self._roll_index = names.index("waist_roll_joint")
        self._previous: np.ndarray | None = None

    def reset(self, initial_target: np.ndarray | None = None) -> None:
        self._previous = None if initial_target is None else np.asarray(initial_target, dtype=np.float32).copy()

    def apply(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32).copy()
        if target.shape != (len(self.joint_names),):
            raise ValueError(f"Expected target shape {(len(self.joint_names),)}, got {target.shape}")

        target[self._pitch_index] = np.clip(target[self._pitch_index], -self.pitch_limit_rad, self.pitch_limit_rad)
        target[self._roll_index] = np.clip(target[self._roll_index], -self.roll_limit_rad, self.roll_limit_rad)
        if self._previous is None:
            self._previous = target.copy()
            return target

        max_delta = self.rate_limit_rad_s / self.control_hz
        for index in (self._pitch_index, self._roll_index):
            filtered = self._previous[index] + self.filter_alpha * (target[index] - self._previous[index])
            target[index] = self._previous[index] + np.clip(
                filtered - self._previous[index],
                -max_delta,
                max_delta,
            )
        self._previous = target.copy()
        return target
