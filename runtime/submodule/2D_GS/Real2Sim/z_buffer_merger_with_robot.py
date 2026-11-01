"""Gaussian background compositing for simulated camera observations."""

from __future__ import annotations

import sys
import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BUNDLED_REAL2SIM = PROJECT_ROOT / "utils" / "real2sim"
if str(BUNDLED_REAL2SIM) not in sys.path:
    sys.path.insert(0, str(BUNDLED_REAL2SIM))

import z_buffer_merger as _base
from z_buffer_merger import *  # noqa: F401,F403


def camera_intrinsics_from_matrix(matrix, width: int, height: int) -> dict:
    k = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    return {
        "width": int(width),
        "height": int(height),
        "K": torch.from_numpy(k.copy()).float(),
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }


def z_buffer_merge_force_foreground_over_bg(
    qian_rgb,
    qian_depth,
    bg_rgb,
    bg_depth,
    force_mask,
):
    """Force Isaac mesh foreground over the Gaussian scene."""
    qian_depth = qian_depth.squeeze()
    qian_rgb_bgra = _base.cv2.cvtColor(qian_rgb, _base.cv2.COLOR_RGBA2BGRA)
    final_rgb = bg_rgb.copy()
    final_depth = bg_depth.copy()

    force_mask = force_mask & (qian_depth > 0) & (qian_depth < np.iinfo(np.uint16).max)
    final_rgb[force_mask] = qian_rgb_bgra[force_mask, :3]
    final_depth[force_mask] = qian_depth[force_mask]

    return final_rgb, final_depth


def _sqrt_positive_part(value: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(value)
    mask = value > 0
    result[mask] = torch.sqrt(value[mask])
    return result


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(9), dim=-1)
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ]
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[0] ** 2, m21 - m12, m02 - m20, m10 - m01]),
            torch.stack([m21 - m12, q_abs[1] ** 2, m10 + m01, m02 + m20]),
            torch.stack([m02 - m20, m10 + m01, q_abs[2] ** 2, m12 + m21]),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[3] ** 2]),
        ]
    )
    candidates = quat_by_rijk / (2.0 * q_abs[:, None].clamp_min(0.1))
    return candidates[q_abs.argmax()]


def _quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


class GSBackgroundRenderer(_base.GSBackgroundRenderer):
    def __init__(
        self,
        model_path,
        intrinsics,
        sh_params=None,
        T=None,
        inference_only=False,
        backend_override=None,
    ):
        self.model_path = model_path
        self.backend = backend_override or _base.load_gs_backend(model_path)
        self.backend_name = self.backend["name"]
        self.render_fn = self.backend["render_fn"]
        self.pipe = self.backend["pipe"]
        if self.backend_name == "2dgs" and not hasattr(self.pipe, "depth_ratio"):
            self.pipe.depth_ratio = 0.0
        self.gs = self.backend["GaussianModel"](3)
        self.gs.load_ply(model_path)
        self.intrinsics = intrinsics
        self.inference_only = bool(inference_only)
        self._supports_inference_only = (
            "inference_only" in inspect.signature(self.render_fn).parameters
        )
        if self.inference_only:
            for attr_name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
                getattr(self.gs, attr_name).requires_grad_(False)
        self.fake_plane_depth = _base.FakePlaneDepthRenderer.from_model_path(model_path)

        if T is not None:
            transform = torch.as_tensor(T, device=self.gs._xyz.device, dtype=self.gs._xyz.dtype)
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            self.gs._xyz = self.gs._xyz @ rotation.T + translation
            rotation_quat = _matrix_to_quaternion(rotation)
            self.gs._rotation = _quaternion_multiply(
                rotation_quat.expand_as(self.gs._rotation),
                F.normalize(self.gs._rotation, dim=-1),
            )

        if sh_params:
            self.adjust_sh_parameters(**sh_params)

    @torch.no_grad()
    def render(self, viewpoint_camera, depth_threshold=0.1):
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        if self.backend_name == "2dgs" and not hasattr(self.pipe, "depth_ratio"):
            self.pipe.depth_ratio = 0.0
        if self.backend_name == "2dgs" and self.inference_only and self._supports_inference_only:
            package = self.render_fn(
                viewpoint_camera,
                self.gs,
                self.pipe,
                background,
                inference_only=True,
            )
        else:
            package = self.render_fn(viewpoint_camera, self.gs, self.pipe, background)
        if self.backend_name == "2dgs":
            depth_m = package["surf_depth"]
            valid_mask = package["rend_alpha"] > depth_threshold
        else:
            depth_m = package["depth"]
            valid_mask = (
                torch.isfinite(depth_m)
                & (depth_m > viewpoint_camera.znear)
                & (depth_m < viewpoint_camera.zfar)
            )
        depth_m = torch.nan_to_num(
            depth_m,
            nan=viewpoint_camera.zfar,
            posinf=viewpoint_camera.zfar,
            neginf=viewpoint_camera.zfar,
        ).clone()
        depth_m[~valid_mask] = viewpoint_camera.zfar
        package["depth_m"] = depth_m
        package["valid_mask"] = valid_mask.float()
        return package
