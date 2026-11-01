from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R


_GS_IMPORT_PREFIXES = (
    "scene",
    "gaussian_renderer",
    "utils",
    "arguments",
    "diff_gaussian_rasterization",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def _clear_gs_modules() -> None:
    for module_name in list(sys.modules.keys()):
        if module_name.startswith(_GS_IMPORT_PREFIXES):
            sys.modules.pop(module_name, None)


class _MiniCam:
    """Minimal camera container compatible with 3DGS renderer."""

    def __init__(
        self,
        width: int,
        height: int,
        focal_x: float,
        focal_y: float,
        pose: torch.Tensor,
        get_projection_matrix,
    ):
        self.image_width = int(width)
        self.image_height = int(height)
        self.focal_x = torch.tensor([float(focal_x)], dtype=torch.float32)
        self.focal_y = torch.tensor([float(focal_y)], dtype=torch.float32)
        self.FoVx = 2 * torch.atan(self.image_width / (2 * self.focal_x))
        self.FoVy = 2 * torch.atan(self.image_height / (2 * self.focal_y))
        self.znear = 0.01
        self.zfar = 100.0
        self.world_view_transform = pose.inverse().transpose(0, 1).cuda()
        self.projection_matrix = get_projection_matrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
        ).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.camera_to_world = pose.detach().cpu().numpy().astype(np.float32)


def _parse_pose_file(
    translation: np.ndarray,
    quat_xyzw: np.ndarray,
    ros_pose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Match z_buffer_merger.parse_pose_file behavior exactly."""
    translation = np.asarray(translation, dtype=np.float32)
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)

    if ros_pose:
        return translation, quat_xyzw

    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    original_rot = R.from_quat(quat_wxyz)

    rotation_y = R.from_euler("y", 90, degrees=True)
    rotation_x = R.from_euler("x", -90, degrees=True)

    transformed_rot = original_rot.inv() * rotation_y * rotation_x
    transformed_rot = transformed_rot.inv()

    transformed_quat_wxyz = transformed_rot.as_quat()
    transformed_quat_xyzw = np.array(
        [
            transformed_quat_wxyz[1],
            transformed_quat_wxyz[2],
            transformed_quat_wxyz[3],
            transformed_quat_wxyz[0],
        ],
        dtype=np.float32,
    )

    return translation, transformed_quat_xyzw


def _pose_to_matrix(translation: np.ndarray, rotation_xyzw: np.ndarray) -> torch.Tensor:
    """Match z_buffer_merger.pose_to_matrix behavior."""
    tx, ty, tz = translation
    qx, qy, qz, qw = rotation_xyzw

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    pose = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), tx],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), ty],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return torch.from_numpy(pose).float()


class FakePlaneDepthRenderer:
    """Rasterize a sibling ``*_fake_plane.obj`` as analytic background depth."""

    def __init__(self, obj_path: str | Path, vertices: np.ndarray, faces: np.ndarray):
        self.obj_path = Path(obj_path)
        self.vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        self.faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        print(
            f"[INFO] Loaded fake-plane depth mesh {self.obj_path} "
            f"({len(self.vertices)} vertices, {len(self.faces)} triangles)"
        )

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> "FakePlaneDepthRenderer | None":
        model_path = Path(model_path)
        obj_path = model_path.with_name(f"{model_path.stem}_fake_plane.obj")
        if not obj_path.is_file():
            return None
        vertices, faces = cls._load_obj(obj_path)
        if len(vertices) == 0 or len(faces) == 0:
            print(f"[WARN] Ignoring empty fake-plane OBJ: {obj_path}")
            return None
        return cls(obj_path, vertices, faces)

    @staticmethod
    def _load_obj(obj_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        vertices: list[list[float]] = []
        faces: list[list[int]] = []
        with Path(obj_path).open("r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    face: list[int] = []
                    for token in line.split()[1:]:
                        idx_text = token.split("/", 1)[0]
                        if not idx_text:
                            continue
                        idx = int(idx_text)
                        face.append(len(vertices) + idx if idx < 0 else idx - 1)
                    for i in range(1, len(face) - 1):
                        faces.append([face[0], face[i], face[i + 1]])
        return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)

    def render_depth_m(
        self,
        camera: _MiniCam,
        *,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> np.ndarray:
        camera_to_world = np.asarray(camera.camera_to_world, dtype=np.float32)
        world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)

        verts_h = np.concatenate(
            [self.vertices, np.ones((self.vertices.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        verts_cam = (verts_h @ world_to_camera.T)[:, :3]
        z = verts_cam[:, 2]
        valid_vertex = np.isfinite(z) & (z > camera.znear) & (z < camera.zfar)

        projected = np.empty((verts_cam.shape[0], 3), dtype=np.float32)
        projected[:, 0] = (verts_cam[:, 0] * fx / np.maximum(z, 1e-6)) + cx
        projected[:, 1] = (verts_cam[:, 1] * fy / np.maximum(z, 1e-6)) + cy
        projected[:, 2] = z

        depth = np.full((height, width), np.inf, dtype=np.float32)
        for tri in self.faces:
            if not np.any(valid_vertex[tri]):
                continue
            p0, p1, p2 = projected[tri]
            if np.any(~np.isfinite([p0, p1, p2])):
                continue
            if p0[2] <= camera.znear or p1[2] <= camera.znear or p2[2] <= camera.znear:
                continue

            min_x = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
            max_x = min(width - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
            min_y = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
            max_y = min(height - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))
            if min_x > max_x or min_y > max_y:
                continue

            xs = np.arange(min_x, max_x + 1, dtype=np.float32)
            ys = np.arange(min_y, max_y + 1, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)
            px = xx + 0.5
            py = yy + 0.5

            denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
            if abs(float(denom)) < 1e-8:
                continue
            w0 = ((p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])) / denom
            w1 = ((p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])) / denom
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
            if not np.any(inside):
                continue

            tri_depth = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
            valid = inside & np.isfinite(tri_depth) & (tri_depth > camera.znear) & (tri_depth < camera.zfar)
            if not np.any(valid):
                continue
            patch = depth[min_y : max_y + 1, min_x : max_x + 1]
            patch[valid] = np.minimum(patch[valid], tri_depth[valid].astype(np.float32))

        return depth


class ThreeDGSRenderer:
    """3DGS adapter following z_buffer_merger camera and z-buffer conventions."""

    def __init__(
        self,
        gs_ply: str,
        intrinsics_matrix: np.ndarray,
        foreground_semantic_rgba: np.ndarray,
        foreground_depth_scale_semantic_rgba: np.ndarray | None = None,
        foreground_depth_scale: float = 0.5,
        foreground_depth_margin_mm: float = 5.0,
        gs_root: str = str(PROJECT_ROOT / "submodule" / "3DGS"),
        init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        init_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        render_scale: float = 1.0,
        max_points: int | None = None,
        ros_pose: bool = False,
        use_fake_plane_depth: bool = True,
    ):
        self.gs_ply = str(gs_ply)
        load_ply = self._prepare_subsampled_ply(self.gs_ply, max_points=max_points)
        self.gs_root = Path(gs_root)
        if not Path(self.gs_ply).is_file():
            raise FileNotFoundError(f"3DGS PLY not found: {self.gs_ply}")
        if not self.gs_root.is_dir():
            raise FileNotFoundError(f"3DGS root not found: {self.gs_root}")

        self._render_fn, self._gaussian_model_cls, self._get_projection_matrix = self._import_3dgs()
        self._pipe = SimpleNamespace(
            compute_cov3D_python=False,
            convert_SHs_python=False,
            debug=False,
            antialiasing=False,
        )
        self._bg_color = torch.tensor([0.0, 0.0, 0.0], device="cuda", dtype=torch.float32)

        self._gs_model = self._gaussian_model_cls(3)
        self._gs_model.load_ply(load_ply)
        self._apply_initial_transform(init_pos, init_wxyz)
        self._subsample_gaussians(max_points=max_points)

        intrinsics_matrix = np.asarray(intrinsics_matrix, dtype=np.float32)
        if intrinsics_matrix.shape != (4, 4):
            raise ValueError(f"Expected intrinsics_matrix shape (4,4), got {intrinsics_matrix.shape}")

        self._src_fx = float(intrinsics_matrix[0, 0])
        self._src_fy = float(intrinsics_matrix[1, 1])
        self._src_width = int(intrinsics_matrix[3, 1])
        self._src_height = int(intrinsics_matrix[3, 0])
        if self._src_width <= 0 or self._src_height <= 0:
            raise ValueError(
                f"Invalid image size from intrinsics_matrix row[3]: height={self._src_height}, width={self._src_width}"
            )

        self._render_scale = float(render_scale)
        if self._render_scale <= 0.0 or self._render_scale > 1.0:
            raise ValueError(f"render_scale must be in (0,1], got {self._render_scale}")

        self._width = max(64, int(round(self._src_width * self._render_scale)))
        self._height = max(64, int(round(self._src_height * self._render_scale)))
        self._fx = self._src_fx * (self._width / self._src_width)
        self._fy = self._src_fy * (self._height / self._src_height)

        self._foreground_semantic_rgba = np.asarray(foreground_semantic_rgba, dtype=np.uint8)
        if foreground_depth_scale_semantic_rgba is None:
            foreground_depth_scale_semantic_rgba = self._foreground_semantic_rgba
        self._foreground_depth_scale_semantic_rgba = np.asarray(
            foreground_depth_scale_semantic_rgba,
            dtype=np.uint8,
        )
        self._foreground_depth_scale = float(foreground_depth_scale)
        if not np.isfinite(self._foreground_depth_scale) or self._foreground_depth_scale <= 0.0:
            raise ValueError(f"foreground_depth_scale must be positive and finite, got {self._foreground_depth_scale}")
        self._foreground_depth_margin_mm = float(foreground_depth_margin_mm)
        if not np.isfinite(self._foreground_depth_margin_mm) or self._foreground_depth_margin_mm < 0.0:
            raise ValueError(
                f"foreground_depth_margin_mm must be finite and non-negative, got {self._foreground_depth_margin_mm}"
            )
        self._ros_pose = bool(ros_pose)
        self._fake_plane_depth = FakePlaneDepthRenderer.from_model_path(self.gs_ply) if use_fake_plane_depth else None

    @staticmethod
    def _prepare_subsampled_ply(gs_ply: str, max_points: int | None) -> str:
        if max_points is None or max_points < 0:
            return gs_ply
        if max_points == 0:
            raise ValueError("max_points must be positive, None, or negative for no subsampling.")

        source = Path(gs_ply)
        cache_dir = Path("/tmp") / "humanoidvla_3dgs_subsample"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{source.stem}_{int(max_points)}.ply"
        if cache_path.is_file() and cache_path.stat().st_mtime >= source.stat().st_mtime:
            print(f"[INFO] Using cached subsampled 3DGS PLY: {cache_path}")
            return str(cache_path)

        ply = PlyData.read(str(source))
        vertex = ply["vertex"].data
        total = int(len(vertex))
        if total <= max_points:
            return gs_ply
        idx = np.linspace(0, total - 1, num=int(max_points), dtype=np.int64)
        subsampled = vertex[idx].copy()
        PlyData([PlyElement.describe(subsampled, "vertex")], text=ply.text).write(str(cache_path))
        print(f"[INFO] Wrote subsampled 3DGS PLY: {cache_path} ({len(subsampled)}/{total} points)")
        return str(cache_path)

    def _import_3dgs(self):
        _clear_gs_modules()

        gs_paths = [
            self.gs_root,
            self.gs_root / "submodules" / "simple-knn",
            self.gs_root / "submodules" / "diff-gaussian-rasterization",
        ]
        for path in reversed(gs_paths):
            path_str = str(path)
            if path_str in sys.path:
                sys.path.remove(path_str)
            if path.is_dir():
                sys.path.insert(0, path_str)

        try:
            gaussian_model_module = importlib.import_module("scene.gaussian_model")
            gaussian_renderer_module = importlib.import_module("gaussian_renderer")
            graphics_utils_module = importlib.import_module("utils.graphics_utils")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Failed to import 3DGS runtime dependencies. "
                "Please install local 3DGS backend in env_isaaclab. "
                f"Original error: {exc}"
            ) from exc

        return (
            gaussian_renderer_module.render,
            gaussian_model_module.GaussianModel,
            graphics_utils_module.getProjectionMatrix,
        )

    @staticmethod
    def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
        m = matrix
        t = torch.trace(m)
        if t > 0.0:
            s = torch.sqrt(t + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        q = torch.tensor([w, x, y, z], dtype=torch.float32, device=matrix.device)
        return q / torch.linalg.norm(q)

    @staticmethod
    def _quaternion_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        aw, ax, ay, az = torch.unbind(a, -1)
        bw, bx, by, bz = torch.unbind(b, -1)
        ow = aw * bw - ax * bx - ay * by - az * bz
        ox = aw * bx + ax * bw + ay * bz - az * by
        oy = aw * by - ax * bz + ay * bw + az * bx
        oz = aw * bz + ax * by - ay * bx + az * bw
        return torch.stack((ow, ox, oy, oz), -1)

    @staticmethod
    def _create_transform_matrix(
        pos: tuple[float, float, float],
        rot_wxyz: tuple[float, float, float, float],
    ) -> np.ndarray:
        rotation = R.from_quat([rot_wxyz[1], rot_wxyz[2], rot_wxyz[3], rot_wxyz[0]]).as_matrix()
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(pos, dtype=np.float32)
        return transform

    def _apply_initial_transform(
        self,
        pos: tuple[float, float, float],
        rot_wxyz: tuple[float, float, float, float],
    ) -> None:
        transform = torch.from_numpy(self._create_transform_matrix(pos, rot_wxyz)).to(self._gs_model._xyz.device).float()

        xyz = self._gs_model._xyz
        self._set_gs_tensor("_xyz", xyz @ transform[:3, :3].T + transform[:3, 3])

        rotation_delta = self._matrix_to_quaternion(transform[:3, :3])
        self._set_gs_tensor(
            "_rotation",
            self._quaternion_mul(
                rotation_delta,
                torch.nn.functional.normalize(self._gs_model._rotation, dim=-1),
            ),
        )

    def _set_gs_tensor(self, name: str, value: torch.Tensor) -> None:
        old = getattr(self._gs_model, name)
        if isinstance(old, torch.nn.Parameter):
            value = torch.nn.Parameter(value, requires_grad=old.requires_grad)
        setattr(self._gs_model, name, value)

    def _subsample_gaussians(self, max_points: int | None) -> None:
        if max_points is None or max_points < 0:
            return
        if max_points == 0:
            raise ValueError("max_points must be positive, None, or negative for no subsampling.")

        xyz = getattr(self._gs_model, "_xyz", None)
        if not isinstance(xyz, torch.Tensor):
            return
        total = int(xyz.shape[0])
        if total <= max_points:
            return

        device = xyz.device
        idx = torch.linspace(0, total - 1, steps=max_points, device=device).long()
        for attr in (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
        ):
            tensor = getattr(self._gs_model, attr, None)
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1 and tensor.shape[0] == total:
                setattr(self._gs_model, attr, tensor.index_select(0, idx).contiguous())

    @staticmethod
    def _quat_wxyz_to_xyzw(quat_w_world: np.ndarray) -> np.ndarray:
        quat_w_world = np.asarray(quat_w_world, dtype=np.float32).reshape(-1)
        if quat_w_world.shape[0] != 4:
            raise ValueError(f"Expected quat_w_world with shape (4,), got {quat_w_world.shape}")
        return np.array([quat_w_world[1], quat_w_world[2], quat_w_world[3], quat_w_world[0]], dtype=np.float32)

    def _build_camera(self, pos_w: np.ndarray, quat_w_world: np.ndarray) -> _MiniCam:
        quat_xyzw = self._quat_wxyz_to_xyzw(quat_w_world)
        translation, rotation_xyzw = _parse_pose_file(pos_w, quat_xyzw, ros_pose=self._ros_pose)
        pose_matrix = _pose_to_matrix(translation, rotation_xyzw)

        return _MiniCam(
            width=self._width,
            height=self._height,
            focal_x=self._fx,
            focal_y=self._fy,
            pose=pose_matrix,
            get_projection_matrix=self._get_projection_matrix,
        )

    @torch.no_grad()
    def _render_bg_rgb_depth_mm(
        self,
        camera: _MiniCam,
    ) -> tuple[np.ndarray, np.ndarray]:
        render_pkg = self._render_fn(camera, self._gs_model, self._pipe, self._bg_color)

        bg_rgb = (
            render_pkg["render"].clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
        ).astype(np.uint8)

        depth_m = render_pkg.get("depth", render_pkg.get("render_depth"))
        if depth_m is None:
            raise KeyError(f"3DGS render package has no depth output; keys={list(render_pkg.keys())}")
        if depth_m.ndim == 3:
            depth_m = depth_m.squeeze(0)
        depth_m = depth_m.detach().float()

        valid_mask = torch.isfinite(depth_m) & (depth_m > camera.znear) & (depth_m < camera.zfar)
        depth_m = torch.nan_to_num(
            depth_m,
            nan=camera.zfar,
            posinf=camera.zfar,
            neginf=camera.zfar,
        )
        depth_m = depth_m.clone()
        depth_m[~valid_mask] = camera.zfar

        if self._fake_plane_depth is not None:
            plane_depth_np = self._fake_plane_depth.render_depth_m(
                camera,
                width=self._width,
                height=self._height,
                fx=self._fx,
                fy=self._fy,
                cx=self._width / 2.0,
                cy=self._height / 2.0,
            )
            plane_depth = torch.from_numpy(plane_depth_np).to(device=depth_m.device, dtype=depth_m.dtype)
            plane_valid = torch.isfinite(plane_depth) & (plane_depth > camera.znear) & (plane_depth < camera.zfar)
            depth_m = torch.where(plane_valid, plane_depth, depth_m)
            valid_mask = valid_mask | plane_valid

        depth_mm = np.clip(
            depth_m.cpu().numpy() * 1000.0,
            0.0,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        depth_mm[~valid_mask.cpu().numpy()] = np.iinfo(np.uint16).max

        return bg_rgb, depth_mm

    @staticmethod
    def _semantic_mask(
        semantic_img: np.ndarray,
        foreground_semantic_rgba: np.ndarray,
    ) -> np.ndarray:
        if foreground_semantic_rgba.size == 0:
            return np.zeros(semantic_img.shape[:2], dtype=bool)

        if semantic_img.ndim != 3 or semantic_img.shape[2] < 3:
            return np.zeros(semantic_img.shape[:2], dtype=bool)

        semantic_rgb = semantic_img[:, :, :3]
        masks = []
        for color in foreground_semantic_rgba:
            color = np.asarray(color, dtype=np.uint8).reshape(-1)
            color_rgb = color[:3]
            masks.append(np.all(semantic_rgb == color_rgb, axis=-1))
        if not masks:
            return np.zeros(semantic_rgb.shape[:2], dtype=bool)
        return np.any(np.stack(masks, axis=-1), axis=-1)

    @staticmethod
    def _z_buffer_merge(
        fg_rgba: np.ndarray,
        fg_depth_mm: np.ndarray,
        bg_bgr: np.ndarray,
        bg_depth_mm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match z_buffer_merger.z_buffer_merge behavior."""
        alpha = fg_rgba[:, :, 3] / 255.0
        fg_mask = alpha > 0

        fg_depth_mm = np.asarray(fg_depth_mm).squeeze()
        fg_bgra = cv2.cvtColor(fg_rgba, cv2.COLOR_RGBA2BGRA)

        final_bgr = bg_bgr.copy()
        final_depth_mm = bg_depth_mm.copy()

        adjusted_fg_depth = fg_depth_mm.copy()
        adjusted_fg_depth[~fg_mask] = np.iinfo(np.uint16).max

        depth_mask = adjusted_fg_depth < bg_depth_mm
        final_bgr[depth_mask] = fg_bgra[depth_mask, :3]
        final_depth_mm[depth_mask] = fg_depth_mm[depth_mask]

        return final_bgr, final_depth_mm

    def render_gs_only(
        self,
        pos_: np.ndarray,
        quat_w_world: np.ndarray,
        output_shape_hw: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        camera = self._build_camera(pos_, quat_w_world)
        bg_rgb, bg_depth_mm = self._render_bg_rgb_depth_mm(camera)

        if output_shape_hw is not None:
            target_h, target_w = int(output_shape_hw[0]), int(output_shape_hw[1])
            if target_h > 0 and target_w > 0 and (bg_rgb.shape[0] != target_h or bg_rgb.shape[1] != target_w):
                bg_rgb = cv2.resize(bg_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                bg_depth_mm = cv2.resize(bg_depth_mm, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                bg_depth_mm = bg_depth_mm.astype(np.uint16, copy=False)

        bg_depth_m = bg_depth_mm.astype(np.float32) / 1000.0
        return bg_rgb, bg_depth_m

    def render_with_gs_bg(
        self,
        pos_: np.ndarray,
        quat_w_world: np.ndarray,
        isaac_render_rgb: np.ndarray,
        isaac_render_depth: np.ndarray,
        isaac_render_semantic_images: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rgb = np.asarray(isaac_render_rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        fg_rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        fg_rgba[:, :, :3] = rgb

        fg_mask = self._semantic_mask(
            semantic_img=np.asarray(isaac_render_semantic_images),
            foreground_semantic_rgba=self._foreground_semantic_rgba,
        )
        fg_rgba[:, :, 3] = np.where(fg_mask, 255, 0).astype(np.uint8)

        fg_depth_raw = np.asarray(isaac_render_depth, dtype=np.float32)
        if fg_depth_raw.ndim == 3 and fg_depth_raw.shape[-1] == 1:
            fg_depth_raw = fg_depth_raw[:, :, 0]
        fg_depth_m = np.nan_to_num(
            fg_depth_raw,
            nan=100.0,
            posinf=100.0,
            neginf=0.0,
        )
        fg_depth_mm = np.clip(
            fg_depth_m * 1000.0,
            0.0,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        bg_rgb, bg_depth_m = self.render_gs_only(
            pos_=pos_,
            quat_w_world=quat_w_world,
            output_shape_hw=(rgb.shape[0], rgb.shape[1]),
        )
        bg_bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)
        bg_depth_mm = np.clip(
            bg_depth_m * 1000.0,
            0.0,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)

        depth_scale_mask = self._semantic_mask(
            semantic_img=np.asarray(isaac_render_semantic_images),
            foreground_semantic_rgba=self._foreground_depth_scale_semantic_rgba,
        )
        if np.any(depth_scale_mask):
            scaled_fg_depth_mm = fg_depth_mm.astype(np.float32, copy=True)
            scaled_fg_depth_mm[depth_scale_mask] *= self._foreground_depth_scale
            bg_depth_mm_float = bg_depth_mm.astype(np.float32, copy=False)
            scaled_fg_depth_mm[depth_scale_mask] = np.minimum(
                scaled_fg_depth_mm[depth_scale_mask],
                np.maximum(bg_depth_mm_float[depth_scale_mask] - self._foreground_depth_margin_mm, 0.0),
            )
            fg_depth_mm = np.clip(
                scaled_fg_depth_mm,
                0.0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)

        merged_bgr, merged_depth_mm = self._z_buffer_merge(
            fg_rgba=fg_rgba,
            fg_depth_mm=fg_depth_mm,
            bg_bgr=bg_bgr,
            bg_depth_mm=bg_depth_mm,
        )

        merged_rgb = cv2.cvtColor(merged_bgr, cv2.COLOR_BGR2RGB)
        merged_depth_m = merged_depth_mm.astype(np.float32) / 1000.0
        return merged_rgb, merged_depth_m
