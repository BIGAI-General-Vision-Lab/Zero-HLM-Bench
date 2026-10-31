#!/usr/bin/env python3
"""Render Task3's static scene and native-axis dynamic door as one 2DGS background.

This stage intentionally renders no Isaac or robot foreground.  Door motion is
driven only by the recorded hinge angle; recorded door/link translations are
not used.  The resulting RGB videos are inputs to the later foreground
compositor.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dynamic_door_gs import (  # noqa: E402
    DEFAULT_GS_WORLD_YAW_CORRECTION_DEG,
    DoorHingeTransformCache,
    authored_gs_to_world_matrix,
    first_existing_dataset,
    load_videoartgs_hinge_axis,
    scene_transform_from_demo,
    sh_rotation_blocks,
    transform_axis,
    yaw_about_world_point_matrix,
)


DEFAULT_GS_DIR = PROJECT_ROOT / "assets" / "GS_Assets" / "B1_art"
DEFAULT_REAL2SIM = PROJECT_ROOT / "submodule" / "2D_GS" / "Real2Sim"
HINGE_PATHS = (
    "states/door/hinge_position",
    "states/articulation/scene/joint_position",
)
CAMERA_PATHS = (
    "camera/pos_w",
    "camera/quat_w_world",
    "camera/intrinsic_matrices",
)


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = left.unbind(dim=-1)
    w2, x2, y2, z2 = right.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert one proper 3x3 rotation matrix to a normalized wxyz quaternion."""

    m = matrix.detach().cpu().numpy().astype(np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                0.25 * scale,
                (m[2, 1] - m[1, 2]) / scale,
                (m[0, 2] - m[2, 0]) / scale,
                (m[1, 0] - m[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            scale = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (m[2, 1] - m[1, 2]) / scale,
                    0.25 * scale,
                    (m[0, 1] + m[1, 0]) / scale,
                    (m[0, 2] + m[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (m[0, 2] - m[2, 0]) / scale,
                    (m[0, 1] + m[1, 0]) / scale,
                    0.25 * scale,
                    (m[1, 2] + m[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quat = np.asarray(
                [
                    (m[1, 0] - m[0, 1]) / scale,
                    (m[0, 2] + m[2, 0]) / scale,
                    (m[1, 2] + m[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quat /= max(float(np.linalg.norm(quat)), 1.0e-12)
    return torch.as_tensor(quat, device=matrix.device, dtype=matrix.dtype)


class CombinedDynamicDoorModel:
    """Combine scene and door GS once, then update only the door suffix."""

    ATTRIBUTES = ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")

    def __init__(self, scene_renderer: Any, door_renderer: Any):
        self.scene_count = int(scene_renderer.gs._xyz.shape[0])
        model = scene_renderer.backend["GaussianModel"](3)
        for name in self.ATTRIBUTES:
            setattr(
                model,
                name,
                torch.cat((getattr(scene_renderer.gs, name), getattr(door_renderer.gs, name)), dim=0),
            )
        model.active_sh_degree = min(
            scene_renderer.gs.active_sh_degree,
            door_renderer.gs.active_sh_degree,
        )
        scene_renderer.gs = model
        self.model = model
        self.door_xyz0 = door_renderer.gs._xyz.detach().clone()
        self.door_rotation0 = torch.nn.functional.normalize(
            door_renderer.gs._rotation.detach().clone(), dim=-1
        )
        self.door_features_rest0 = door_renderer.gs._features_rest.detach().clone()

    @torch.no_grad()
    def set_relative_world_transform(self, transform: np.ndarray) -> None:
        matrix = torch.as_tensor(
            transform, device=self.door_xyz0.device, dtype=self.door_xyz0.dtype
        )
        rotation = matrix[:3, :3]
        self.model._xyz[self.scene_count :] = self.door_xyz0 @ rotation.T + matrix[:3, 3]
        delta_quat = matrix_to_quaternion(rotation)
        self.model._rotation[self.scene_count :] = quaternion_multiply(
            delta_quat.expand_as(self.door_rotation0), self.door_rotation0
        )
        band_start = 0
        for block in sh_rotation_blocks(np.asarray(transform, dtype=np.float64)[:3, :3]):
            band_stop = band_start + block.shape[0]
            block_tensor = torch.as_tensor(
                block, device=self.door_features_rest0.device, dtype=self.door_features_rest0.dtype
            )
            self.model._features_rest[self.scene_count :, band_start:band_stop, :] = torch.einsum(
                "ij,njc->nic",
                block_tensor,
                self.door_features_rest0[:, band_start:band_stop, :],
            )
            band_start = band_stop


