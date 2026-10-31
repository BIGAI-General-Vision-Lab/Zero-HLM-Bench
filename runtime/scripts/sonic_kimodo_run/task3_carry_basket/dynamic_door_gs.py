#!/usr/bin/env python3
"""Coordinate contracts for composing Task3's dynamic door into its scene GS."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# Both B1_art PLYs share this authored GS-local -> Isaac-world alignment.
DEFAULT_GS_TO_WORLD_TRANSLATION = (0.15325, 0.10805, 1.50718)
DEFAULT_GS_TO_WORLD_YAW_DEG = -90.0
# The ``*_task3_world.ply`` assets bake this second rotation around the
# authored scene root with the canonical 2DGS transform utility.  Baking is
# required because that utility rotates directional SH coefficients in
# addition to Gaussian centers and quaternions.
DEFAULT_GS_WORLD_YAW_CORRECTION_DEG = -90.0

_SH_C1 = 0.4886025119029199
_SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
          -1.0925484305920792, 0.5462742152960396)
_SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
          0.3731763325901154, -0.4570457994644658, 1.445305721320277,
          -0.5900435899266435)


def _real_sh_bands(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the degree 1..3 real-SH bases used by Gaussian splatting."""

    values = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    x, y, z = values.T
    degree1 = np.stack((-_SH_C1 * y, _SH_C1 * z, -_SH_C1 * x), axis=1)
    degree2 = np.stack(
        (
            _SH_C2[0] * x * y,
            _SH_C2[1] * y * z,
            _SH_C2[2] * (2.0 * z * z - x * x - y * y),
            _SH_C2[3] * x * z,
            _SH_C2[4] * (x * x - y * y),
        ),
        axis=1,
    )
    degree3 = np.stack(
        (
            _SH_C3[0] * y * (3.0 * x * x - y * y),
            _SH_C3[1] * x * y * z,
            _SH_C3[2] * y * (4.0 * z * z - x * x - y * y),
            _SH_C3[3] * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y),
            _SH_C3[4] * x * (4.0 * z * z - x * x - y * y),
            _SH_C3[5] * z * (x * x - y * y),
            _SH_C3[6] * x * (x * x - 3.0 * y * y),
        ),
        axis=1,
    )
    return degree1, degree2, degree3


def _fibonacci_sphere(count: int = 32) -> np.ndarray:
    indices = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = math.pi * (3.0 - math.sqrt(5.0)) * indices
    return np.stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z), axis=1)


_SH_SAMPLE_DIRECTIONS = _fibonacci_sphere()
_SH_SAMPLE_BANDS = _real_sh_bands(_SH_SAMPLE_DIRECTIONS)
_SH_SAMPLE_PSEUDOINVERSES = tuple(np.linalg.pinv(band) for band in _SH_SAMPLE_BANDS)


def sh_rotation_blocks(rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dependency-free degree 1..3 coefficient rotations.

    This solves the real-SH basis identity ``Y(d) D = Y(R^-1 d)``. It matches
    the e3nn Wigner-D convention used by transform_2dgs_and_mesh_to_usd.py,
    while keeping the remote renderer independent of e3nn.
    """

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    rotated_directions = _SH_SAMPLE_DIRECTIONS @ matrix
    rotated_bands = _real_sh_bands(rotated_directions)
    return tuple(
        inverse @ rotated
        for inverse, rotated in zip(_SH_SAMPLE_PSEUDOINVERSES, rotated_bands)
    )


def pose_wxyz_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert ``[x, y, z, qw, qx, qy, qz]`` to a homogeneous matrix."""

    value = np.asarray(pose, dtype=np.float64).reshape(7)
    quat = value[3:]
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        raise ValueError("pose contains a near-zero quaternion")
    w, x, y, z = quat / norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = value[:3]
    return result


def se2_matrix(translation_xy: tuple[float, float], delta_yaw: float) -> np.ndarray:
    c = math.cos(float(delta_yaw))
    s = math.sin(float(delta_yaw))
    result = np.eye(4, dtype=np.float64)
    result[:2, :2] = ((c, -s), (s, c))
    result[:2, 3] = np.asarray(translation_xy, dtype=np.float64)
    return result


def authored_gs_to_world_matrix(
    translation: tuple[float, float, float] = DEFAULT_GS_TO_WORLD_TRANSLATION,
    yaw_degrees: float = DEFAULT_GS_TO_WORLD_YAW_DEG,
) -> np.ndarray:
    angle = math.radians(float(yaw_degrees))
    result = se2_matrix((float(translation[0]), float(translation[1])), angle)
    result[2, 3] = float(translation[2])
    return result


def yaw_about_world_point_matrix(
    yaw_degrees: float,
    pivot_xyz: tuple[float, float, float] = DEFAULT_GS_TO_WORLD_TRANSLATION,
) -> np.ndarray:
    """Rotate an already world-aligned GS asset around its authored root point."""

    angle = math.radians(float(yaw_degrees))
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.eye(4, dtype=np.float64)
    rotation[:2, :2] = ((c, -s), (s, c))
    pivot = np.asarray(pivot_xyz, dtype=np.float64).reshape(3)
    to_pivot = np.eye(4, dtype=np.float64)
    to_pivot[:3, 3] = pivot
    from_pivot = np.eye(4, dtype=np.float64)
    from_pivot[:3, 3] = -pivot
    return to_pivot @ rotation @ from_pivot


def load_videoartgs_hinge_axis(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the single dynamic hinge axis exported by VideoArtGS."""

    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    hinges = [entry for entry in entries if entry.get("joint") == "hinge"]
    if len(hinges) != 1:
        raise ValueError(f"expected exactly one hinge in {path}, found {len(hinges)}")
    axis = hinges[0]["jointData"]["axis"]
    origin = np.asarray(axis["origin"], dtype=np.float64).reshape(3)
    direction = np.asarray(axis["direction"], dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise ValueError(f"hinge direction in {path} is near zero")
    return origin, direction / norm


def transform_axis(
    origin: np.ndarray,
    direction: np.ndarray,
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform an axis point and direction through one rigid transform."""

    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    point = matrix[:3, :3] @ np.asarray(origin, dtype=np.float64).reshape(3) + matrix[:3, 3]
    vector = matrix[:3, :3] @ np.asarray(direction, dtype=np.float64).reshape(3)
    vector /= max(float(np.linalg.norm(vector)), 1.0e-12)
    return point, vector


def axis_angle_world_matrix(
    origin: np.ndarray,
    direction: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Return a world transform rotating by ``angle`` about a spatial axis."""

    axis = np.asarray(direction, dtype=np.float64).reshape(3)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    x, y, z = axis
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    rotation = np.eye(3) + math.sin(float(angle)) * skew + (1.0 - math.cos(float(angle))) * (skew @ skew)
    point = np.asarray(origin, dtype=np.float64).reshape(3)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = point - rotation @ point
    return result


def door_hinge_relative_world_matrix(
    reference_hinge: float,
    frame_hinge: float,
    axis_origin_world: np.ndarray,
    axis_direction_world: np.ndarray,
) -> np.ndarray:
    """Drive the door around VideoArtGS's own axis using recorded hinge angles."""

    return axis_angle_world_matrix(
        axis_origin_world,
        axis_direction_world,
        float(frame_hinge) - float(reference_hinge),
    )


@dataclass
class DoorHingeTransformCache:
    """Cache the native-axis transform while the recorded hinge angle is unchanged."""

    reference_hinge: float
    axis_origin_world: np.ndarray
    axis_direction_world: np.ndarray
    angle_tolerance: float = 1.0e-8
    _last_hinge: float | None = field(default=None, init=False, repr=False)
    _last_transform: np.ndarray | None = field(default=None, init=False, repr=False)

    def transform(self, frame_hinge: float) -> tuple[np.ndarray, bool]:
        """Return ``(transform, changed)`` for one recorded hinge angle."""

        hinge = float(frame_hinge)
        if self._last_hinge is not None and abs(hinge - self._last_hinge) <= self.angle_tolerance:
            assert self._last_transform is not None
            return self._last_transform, False
        self._last_hinge = hinge
        self._last_transform = door_hinge_relative_world_matrix(
            self.reference_hinge,
            hinge,
            self.axis_origin_world,
            self.axis_direction_world,
        )
        return self._last_transform, True


def _decode_json(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def scene_transform_from_demo(group: Any) -> np.ndarray:
    """Read per-demo scene SE(2), supporting replay and final-HDF5 schemas."""

    if "scene/se2_delta_yaw" in group and "scene/se2_translation_xy" in group:
        yaw = float(np.asarray(group["scene/se2_delta_yaw"][()]).reshape(-1)[0])
        xy = np.asarray(group["scene/se2_translation_xy"][()], dtype=np.float64).reshape(2)
        return se2_matrix((float(xy[0]), float(xy[1])), yaw)

    for attribute in ("candidate_contract_json", "success_metadata_json"):
        contract = _decode_json(group.attrs.get(attribute))
        scene = contract.get("scene_transform")
        if not isinstance(scene, dict):
            continue
        xy = np.asarray(scene.get("translation_xy", (0.0, 0.0)), dtype=np.float64).reshape(2)
        yaw = float(scene.get("delta_yaw", 0.0))
        return se2_matrix((float(xy[0]), float(xy[1])), yaw)
    return np.eye(4, dtype=np.float64)


def demo_gs_to_world_matrix(group: Any) -> np.ndarray:
    """Return the common transform used by the static scene and reference door GS."""

    return scene_transform_from_demo(group) @ authored_gs_to_world_matrix()


def door_relative_world_matrix(reference_pose: np.ndarray, frame_pose: np.ndarray) -> np.ndarray:
    """Return motion from a door GS reference pose to one recorded link pose.

    ``obj_1_gs.ply`` is captured in one fixed hinge configuration.  The caller
    must pass the HDF5 link or mesh pose for that same configuration as
    ``reference_pose``.  Task3's B1_art ``obj_1_gs.ply`` uses the initial
    zero-degree door pose.
    """

    reference = pose_wxyz_to_matrix(reference_pose)
    current = pose_wxyz_to_matrix(frame_pose)
    return current @ np.linalg.inv(reference)


def door_gs_to_world_matrix(
    reference_pose: np.ndarray,
    frame_pose: np.ndarray,
    base_gs_to_world: np.ndarray,
) -> np.ndarray:
    """Map original ``obj_1_gs`` coordinates to the current Isaac-world pose."""

    return door_relative_world_matrix(reference_pose, frame_pose) @ np.asarray(
        base_gs_to_world, dtype=np.float64
    ).reshape(4, 4)


def select_door_reference_frame(hinge_position: np.ndarray, mode: str = "first") -> int:
    """Choose the recorded link pose matching the door PLY's reference state."""

    hinge = np.asarray(hinge_position, dtype=np.float64).reshape(-1)
    if hinge.size == 0:
        raise ValueError("hinge trajectory is empty")
    if mode == "first":
        return 0
    if mode == "max-hinge":
        return int(np.argmax(hinge))
    raise ValueError(f"unsupported door reference mode: {mode!r}")


def first_existing_dataset(group: Any, paths: tuple[str, ...]) -> tuple[np.ndarray, str]:
    """Load the first available HDF5 dataset from an ordered alias list."""

    for path in paths:
        if path in group:
            return np.asarray(group[path]), path
    raise KeyError(f"none of the required datasets exists: {paths}")


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def select_diagnostic_frames(hinge_position: np.ndarray, contact_latched: np.ndarray | None) -> list[int]:
    """Select closed, trigger, intermediate, and terminal frames deterministically."""

    hinge = np.asarray(hinge_position, dtype=np.float64).reshape(-1)
    if hinge.size == 0:
        raise ValueError("hinge trajectory is empty")
    selected = {0, int(hinge.size - 1), int(np.argmax(np.abs(hinge - hinge[0])))}
    span = float(np.max(np.abs(hinge - hinge[0])))
    if span > 1.0e-8:
        delta = np.abs(hinge - hinge[0])
        for fraction in (0.1, 0.5, 0.9):
            selected.add(int(np.argmin(np.abs(delta - fraction * span))))
    if contact_latched is not None:
        latched = np.flatnonzero(np.asarray(contact_latched).reshape(-1).astype(bool))
        if latched.size:
            selected.add(int(latched[0]))
            selected.add(max(0, int(latched[0]) - 1))
    return sorted(selected)
