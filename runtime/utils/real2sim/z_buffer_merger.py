import importlib
import json
import math
import multiprocessing
import os
import sys
from pathlib import Path
from contextlib import closing
from functools import partial
from types import SimpleNamespace

import cv2
import h5py
import matplotlib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

_REAL2SIM_UTILS_DIR = Path(__file__).resolve().parent
if str(_REAL2SIM_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_REAL2SIM_UTILS_DIR))
from my_utils import *


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TWO_D_GS_ROOT = str(
    Path(
        os.environ.get(
            "TWO_D_GS_ROOT",
            PROJECT_ROOT / "submodule" / "2D_GS" / "2d-gaussian-splatting",
        )
    )
)
THREE_D_GS_ROOT = str(Path(os.environ.get("THREE_D_GS_ROOT", PROJECT_ROOT / "submodule" / "3DGS")))
GS_IMPORT_PREFIXES = (
    "scene",
    "gaussian_renderer",
    "utils",
    "arguments",
    "diff_surfel_rasterization",
    "diff_gaussian_rasterization",
)
_BACKEND_CACHE = {}


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def inspect_ply_scale_dims(model_path):
    scale_fields = []
    with open(model_path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"Unsupported model format for {model_path}: expected PLY.")

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while parsing PLY header: {model_path}")
            text = line.decode("ascii", errors="replace").strip()
            if text == "end_header":
                break
            if text.startswith("property ") and " scale_" in text:
                scale_fields.append(text.rsplit(" ", 1)[-1])

    return len(scale_fields), scale_fields


def detect_gs_backend(model_path):
    scale_dims, scale_fields = inspect_ply_scale_dims(model_path)
    if scale_dims == 2:
        return "2dgs"
    if scale_dims == 3:
        return "3dgs"
    raise ValueError(
        f"Unable to infer GS backend from {model_path}. "
        f"Expected 2 or 3 scale fields, got {scale_dims}: {scale_fields}"
    )


def _clear_gs_modules():
    for module_name in list(sys.modules.keys()):
        if module_name.startswith(GS_IMPORT_PREFIXES):
            del sys.modules[module_name]


def load_gs_backend(model_path):
    backend_name = detect_gs_backend(model_path)
    if backend_name in _BACKEND_CACHE:
        return _BACKEND_CACHE[backend_name]

    _clear_gs_modules()

    backend_root = TWO_D_GS_ROOT if backend_name == "2dgs" else THREE_D_GS_ROOT
    if backend_root in sys.path:
        sys.path.remove(backend_root)
    sys.path.insert(0, backend_root)

    try:
        gaussian_model_module = importlib.import_module("scene.gaussian_model")
        gaussian_renderer_module = importlib.import_module("gaussian_renderer")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Failed to import the {backend_name} backend for {model_path}: {exc}. "
            "If this is a 3DGS model, install the local 3DGS rasterizer into env_isaaclab."
        ) from exc

    if backend_name == "2dgs":
        pipe = SimpleNamespace(
            compute_cov3D_python=False,
            convert_SHs_python=False,
            depth_ratio=0.0,
        )
    else:
        pipe = SimpleNamespace(
            compute_cov3D_python=False,
            convert_SHs_python=False,
            debug=False,
            antialiasing=False,
        )

    backend = {
        "name": backend_name,
        "GaussianModel": gaussian_model_module.GaussianModel,
        "render_fn": gaussian_renderer_module.render,
        "pipe": pipe,
    }
    _BACKEND_CACHE[backend_name] = backend
    return backend


def process_episode_batch(args, episode_batch, progress_queue=None):

    input_hdf5, output_video_path_root, model_path, intrinsics_path, sh_params, T, print_zbuffer_stats = args
    results = []

    try:

        renderer = GSBackgroundRenderer(model_path, intrinsics_path, sh_params, T)


        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(input_hdf5)

        dir_name = os.path.dirname(input_hdf5)

        for episode_name in episode_batch:
            try:

                episode_output_paths = {
                    'rgb_video': os.path.join(output_video_path_root, "gs_rgb_videos", f"{episode_name}.mp4"),
                    'depth_h5': os.path.join(output_video_path_root, "gs_depth_h5", f"{episode_name}.h5"),
                    'depth_vis': os.path.join(output_video_path_root, "gs_depth_vis", f"{episode_name}.mp4")
                }

                print(f"Main Python: Saved paths for {episode_name}: {episode_output_paths}")


                os.makedirs(os.path.dirname(episode_output_paths['rgb_video']), exist_ok=True)
                os.makedirs(os.path.dirname(episode_output_paths['depth_h5']), exist_ok=True)
                os.makedirs(os.path.dirname(episode_output_paths['depth_vis']), exist_ok=True)


                episode = dataset_file_handler.load_episode(episode_name, "cpu")

                process_episode(
                    episode=episode,
                    episode_name=episode_name,
                    save_root=os.path.dirname(input_hdf5),
                    renderer=renderer,
                    output_video_path=episode_output_paths,
                    input_dir=dir_name,
                    print_zbuffer_stats=print_zbuffer_stats,
                )

                results.append(f"Success: {episode_name}")
                if progress_queue:
                    progress_queue.put(1)  

            except Exception as e:
                results.append(f"Failed {episode_name}: {str(e)}")
                if progress_queue:
                    progress_queue.put(1)

    except Exception as e:
        results.append(f"Process crash: {str(e)}")
    finally:

        if 'dataset_file_handler' in locals():
            dataset_file_handler.close()
        return results

class GSBackgroundRenderer:
    def __init__(self, model_path, intrinsics_path, sh_params=None, T=None):
        self.model_path = model_path
        self.backend = load_gs_backend(model_path)
        self.backend_name = self.backend["name"]
        self.render_fn = self.backend["render_fn"]
        self.pipe = self.backend["pipe"]

        print(f"Main Python: Detected {self.backend_name} backend for {model_path}")

        self.gs = self.backend["GaussianModel"](3)
        self.gs.load_ply(model_path)

        if T is not None:
            self.gs.transform_gs(T, save_path=None)

        if sh_params:
            self.adjust_sh_parameters(**sh_params)

        self.intrinsics = self.load_camera_intrinsics(intrinsics_path)
        self.fake_plane_depth = FakePlaneDepthRenderer.from_model_path(model_path)

    def load_camera_intrinsics(self, json_path):

        with open(json_path, 'r') as f:
            data = json.load(f)

        width = data['width']
        height = data['height']
        k = data['k']

        fx = k[0]  # fx
        fy = k[4]  # fy
        cx = k[2]  # cx
        cy = k[5]  # cy

        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]).float()

        return {
            'width': width,
            'height': height,
            'K': K,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }

    def adjust_sh_parameters(self, dc_offset=None, rest_offset=None):


        if dc_offset is None and rest_offset is None:
            return

        sh = self.gs.get_features

        if dc_offset is not None:
            sh[:, 0, 0] += dc_offset[0]
            sh[:, 0, 1] += dc_offset[1]
            sh[:, 0, 2] += dc_offset[2]

        if rest_offset is not None:
            sh[:, 1:, 0] += rest_offset[0]
            sh[:, 1:, 1] += rest_offset[1]
            sh[:, 1:, 2] += rest_offset[2]

        self.gs._features_dc = sh[:, :1, :].contiguous()
        self.gs._features_rest = sh[:, 1:, :].contiguous()

    @torch.no_grad()
    def render(self, viewpoint_camera, depth_threshold=0.1):
        bg_color = torch.tensor([0.0, 0.0, 0.0], device="cuda")
        render_pkg = self.render_fn(viewpoint_camera, self.gs, self.pipe, bg_color)

        if self.backend_name == "2dgs":
            depth_m = render_pkg["surf_depth"]
            valid_mask = render_pkg["rend_alpha"] > depth_threshold
        else:
            depth_m = render_pkg["depth"]
            valid_mask = torch.isfinite(depth_m) & (depth_m > viewpoint_camera.znear) & (depth_m < viewpoint_camera.zfar)

        depth_m = torch.nan_to_num(
            depth_m,
            nan=viewpoint_camera.zfar,
            posinf=viewpoint_camera.zfar,
            neginf=viewpoint_camera.zfar,
        )
        depth_m = depth_m.clone()
        depth_m[~valid_mask] = viewpoint_camera.zfar

        if self.fake_plane_depth is not None:
            plane_depth_np = self.fake_plane_depth.render_depth_m(
                viewpoint_camera=viewpoint_camera,
                intrinsics=self.intrinsics,
                znear=float(viewpoint_camera.znear),
                zfar=float(viewpoint_camera.zfar),
            )
            if plane_depth_np is not None:
                plane_depth = torch.from_numpy(plane_depth_np).to(device=depth_m.device, dtype=depth_m.dtype)
                while plane_depth.ndim < depth_m.ndim:
                    plane_depth = plane_depth.unsqueeze(0)
                plane_valid = torch.isfinite(plane_depth) & (plane_depth > viewpoint_camera.znear) & (
                    plane_depth < viewpoint_camera.zfar
                )
                depth_m = torch.where(plane_valid, plane_depth, depth_m)
                valid_mask = valid_mask | plane_valid

        render_pkg["depth_m"] = depth_m
        render_pkg["valid_mask"] = valid_mask.float()
        return render_pkg


class FakePlaneDepthRenderer:
    """Rasterize a sibling ``*_fake_plane.obj`` as analytic background depth."""

    def __init__(self, obj_path, vertices, faces):
        self.obj_path = Path(obj_path)
        self.vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        self.faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        print(
            f"Main Python: Loaded fake-plane depth mesh {self.obj_path} "
            f"({len(self.vertices)} vertices, {len(self.faces)} triangles)"
        )

    @classmethod
    def from_model_path(cls, model_path):
        model_path = Path(model_path)
        obj_path = model_path.with_name(f"{model_path.stem}_fake_plane.obj")
        if not obj_path.is_file():
            return None
        vertices, faces = cls._load_obj(obj_path)
        if len(vertices) == 0 or len(faces) == 0:
            print(f"Main Python: Ignoring empty fake-plane OBJ: {obj_path}")
            return None
        return cls(obj_path, vertices, faces)

    @staticmethod
    def _load_obj(obj_path):
        vertices = []
        faces = []
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
                    face = []
                    for token in line.split()[1:]:
                        if not token:
                            continue
                        idx_text = token.split("/", 1)[0]
                        if not idx_text:
                            continue
                        idx = int(idx_text)
                        if idx < 0:
                            idx = len(vertices) + idx
                        else:
                            idx -= 1
                        face.append(idx)
                    for i in range(1, len(face) - 1):
                        faces.append([face[0], face[i], face[i + 1]])
        return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)

    def render_depth_m(self, viewpoint_camera, intrinsics, znear=0.01, zfar=100.0):
        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])

        if hasattr(viewpoint_camera, "camera_to_world"):
            camera_to_world = np.asarray(viewpoint_camera.camera_to_world, dtype=np.float32)
        else:
            camera_to_world = (
                viewpoint_camera.world_view_transform.detach().cpu().numpy().T
            )
            camera_to_world = np.linalg.inv(camera_to_world).astype(np.float32)
        world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)

        verts_h = np.concatenate(
            [self.vertices, np.ones((self.vertices.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        verts_cam = (verts_h @ world_to_camera.T)[:, :3]
        z = verts_cam[:, 2]
        valid_vertex = np.isfinite(z) & (z > znear) & (z < zfar)

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
            if p0[2] <= znear or p1[2] <= znear or p2[2] <= znear:
                # Fake planes are expected to be in front of the camera and not clipped.

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
            valid = inside & np.isfinite(tri_depth) & (tri_depth > znear) & (tri_depth < zfar)
            if not np.any(valid):
                continue
            patch = depth[min_y : max_y + 1, min_x : max_x + 1]
            patch[valid] = np.minimum(patch[valid], tri_depth[valid].astype(np.float32))

        return depth

class MiniCam:
    def __init__(self, width, height, focal_x, focal_y, pose):
        self.image_width = width
        self.image_height = height
        self.focal_x = torch.tensor([focal_x])
        self.focal_y = torch.tensor([focal_y])
        self.FoVx = 2 * torch.atan(self.image_width / (2 * self.focal_x))
        self.FoVy = 2 * torch.atan(self.image_height / (2 * self.focal_y))
        self.znear = 0.01
        self.zfar = 100.0
        self.world_view_transform = pose.inverse().transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy
        ).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.camera_to_world = pose.detach().cpu().numpy().astype(np.float32)

def parse_pose_file(translation, quat_xyzw, ros_pose=False):

    if ros_pose:

        return translation, quat_xyzw
    else:

        
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]


        original_rot = R.from_quat(quat_wxyz)


        rotation_y = R.from_euler('y', 90, degrees=True)
        rotation_x = R.from_euler('x', -90, degrees=True)


        transformed_rot = original_rot.inv() * rotation_y * rotation_x
        transformed_rot = transformed_rot.inv()  


        transformed_quat_wxyz = transformed_rot.as_quat()
        transformed_quat_xyzw = [
            transformed_quat_wxyz[1],
            transformed_quat_wxyz[2],
            transformed_quat_wxyz[3],
            transformed_quat_wxyz[0]
        ]

        return translation, transformed_quat_xyzw

def pose_to_matrix(translation, rotation):

    tx, ty, tz = translation
    qx, qy, qz, qw = rotation


    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    rot_matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0],
        [0, 0, 0, 1]
    ])


    rot_matrix[:3, 3] = [tx, ty, tz]
    return torch.from_numpy(rot_matrix).float()

def z_buffer_merge(qian_rgb, qian_depth, bg_rgb, bg_depth):


    alpha = qian_rgb[:, :, 3] / 255.0
    qian_mask = alpha > 0
    qian_depth = qian_depth.squeeze()
    qian_rgb = cv2.cvtColor(qian_rgb, cv2.COLOR_RGBA2BGRA)  


    final_rgb = bg_rgb.copy()
    final_depth = bg_depth.copy()


    adjusted_qian_depth = qian_depth.copy()
    adjusted_qian_depth[~qian_mask] = np.iinfo(np.uint16).max


    depth_mask = adjusted_qian_depth < bg_depth
    final_rgb[depth_mask] = qian_rgb[depth_mask, :3]  
    final_depth[depth_mask] = qian_depth[depth_mask]


    return final_rgb, final_depth


def vis_depth(depth):

    if isinstance(depth, np.ndarray) and depth.dtype == np.uint16:
        depth_m = depth / 1000.0  
    else:
        depth_m = depth.copy()

    depth_max = 10  
    depth_m[depth_m > depth_max] = depth_max  
    depth_norm = (depth_m / depth_max * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)


def process_episode(
    episode,
    episode_name,
    save_root,  
    renderer,
    output_video_path,  
    ros_pose=False,
    input_dir=None,  
    print_zbuffer_stats=False,
):

    # Extract metadata
    pos_w = episode._data["images"]["info"]["pos_w"]
    quat_w_world = episode._data["images"]["info"]["quat_w_world"]


    if "rgb" not in episode._data["images"]:
        if input_dir is None:
            raise ValueError("Input directory must be provided if RGB images are not in episode data.")
        rgb_images = read_rgb_video(Path(os.path.join(input_dir, "rgb", f"{episode_name}.mp4")))
        # print(f"Main Python: Read RGB images from {input_dir}/rgb/{episode_name}.mp4")
        # print(f"Main Python: RGB images shape: {rgb_images.shape}")
    else:
        rgb_images = episode._data["images"]["rgb"]

    if "depth" not in episode._data["images"]:
        if input_dir is None:
            raise ValueError("Input directory must be provided if depth images are not in episode data.")
        depth_images = read_depth_h5(Path(os.path.join(input_dir, "depth", f"{episode_name}.h5")))
        # print(f"Main Python: Read depth images from {input_dir}/depth/{episode_name}.h5")
        # print(f"Main Python: Depth images shape: {depth_images.shape}")
    else:
        depth_images = episode._data["images"]["depth"]

    if "semantic" not in episode._data["images"]:
        if input_dir is None:
            raise ValueError("Input directory must be provided if semantic images are not in episode data.")
        semantic_images = read_semantic_video(Path(os.path.join(input_dir, "semantic", f"{episode_name}.avi")))
        # print(f"Main Python: Read semantic images from {input_dir}/semantic/{episode_name}.avi")
        # print(f"Main Python: Semantic images shape: {semantic_images.shape}")
    else:
        semantic_images = episode._data["images"]["semantic"]


    # Check dimensions
    seq_length = len(rgb_images)
    assert len(pos_w) == seq_length, "pos_w length mismatch"
    assert len(quat_w_world) == seq_length, "quat_w_world length mismatch"
    assert len(depth_images) == seq_length, "depth_images length mismatch"
    assert len(semantic_images) == seq_length, "semantic_images length mismatch"

    # Create output directories
    # os.makedirs(output_video_path, exist_ok=True)

    # Define colors for semantic segmentation
    background_color = np.array([12, 12, 12, 255])
    unlabeled_color = np.array([128, 128, 128, 255])
    others_color = np.array([0, 0, 0, 255])
    robot = np.array([255, 36, 66, 255])
    target_placement = np.array([255, 184, 48, 255])
    target_object = np.array([55, 255, 139, 255])

    # Initialize video writers and HDF5 file
    frame_size = (rgb_images[0].shape[1], rgb_images[0].shape[0])
    # depth_video_path = os.path.join(output_video_path, f"{episode_name}_depth.mp4")
    rgb_video_path = output_video_path['rgb_video']
    depth_h5_path = output_video_path['depth_h5']
    depth_colormap_video_path = output_video_path['depth_vis']

    fps, w, h = 30, frame_size[0], frame_size[1]
    mp4 = cv2.VideoWriter_fourcc(*'mp4v')


    rgb_writer = cv2.VideoWriter(rgb_video_path, mp4, fps, (w, h), isColor=True)
    # depth_writer = cv2.VideoWriter(depth_video_path, mp4, fps, (w, h), isColor=False)
    depth_colormap_writer = cv2.VideoWriter(depth_colormap_video_path, mp4, fps, (w, h), isColor=True)


    depth_h5_file = h5py.File(depth_h5_path, 'w')


    try:
        for i in range(seq_length):

            # 1. Process foreground RGB and create assets RGBA

            rgb_img = rgb_images[i]
            semantic_img = semantic_images[i]

            # Create foreground mask for robot, target_placement, and target_object
            foreground_mask = np.any(
                np.stack([
                    np.all(semantic_img == robot, axis=-1),
                    np.all(semantic_img == target_placement, axis=-1),
                    np.all(semantic_img == target_object, axis=-1)
                ], axis=-1), axis=-1
            )


            # Create RGBA image for foreground
            rgba_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 4), dtype=np.uint8)
            rgba_img[:, :, :3] = rgb_img  # RGB channels
            rgba_img[:, :, 3] = np.where(foreground_mask, 255, 0)  # Alpha channel

            # 2. Process foreground depth (convert to mm and uint16)
            qian_depth = depth_images[i] * 1000  # Convert to mm
            qian_depth = np.clip(qian_depth, 0, 65535)  
            qian_depth = qian_depth.astype(np.uint16)

            # 3. Render background using GS model
            translation = pos_w[i].tolist()
            rotation = [
                quat_w_world[i][1],  # x
                quat_w_world[i][2],  # y
                quat_w_world[i][3],  # z
                quat_w_world[i][0]   # w
            ]

            translation, rotation = parse_pose_file(translation, rotation, ros_pose)
            pose_matrix = pose_to_matrix(translation, rotation)
            cam = MiniCam(
                width=renderer.intrinsics['width'],
                height=renderer.intrinsics['height'],
                focal_x=renderer.intrinsics['fx'],
                focal_y=renderer.intrinsics['fy'],
                pose=pose_matrix
            )

            render_pkg = renderer.render(cam)

            # Get background RGB and depth
            bg_rgb = (render_pkg['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            bg_rgb = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)

            bg_depth_m = render_pkg['depth_m'].squeeze().cpu().numpy()
            valid_mask = render_pkg['valid_mask'].squeeze().cpu().numpy().astype(bool)
            bg_depth_mm = np.clip(
                bg_depth_m * 1000.0,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
            bg_depth_mm[~valid_mask] = np.iinfo(np.uint16).max

            if print_zbuffer_stats:
                fg_total = int(foreground_mask.sum())
                if fg_total > 0:
                    fg_front = int((foreground_mask & (qian_depth < bg_depth_mm)).sum())
                    fg_ratio = fg_front / fg_total
                else:
                    fg_front = 0
                    fg_ratio = float('nan')
                print(
                    f"[{episode_name}] frame {i:05d}: "
                    f"fg_front={fg_front}/{fg_total} "
                    f"({fg_ratio:.4f})"
                )

            # 4. Merge foreground and background
            final_rgb, final_depth = z_buffer_merge(
                rgba_img, qian_depth,
                bg_rgb, bg_depth_mm
            )

            # 5. Write to video and HDF5

            rgb_writer.write(final_rgb)


            depth_map_name = f"frame_{i:05d}"
            depth_encoded = cv2.imencode('.png', final_depth)[1]  
            depth_h5_file.create_dataset(depth_map_name, data=np.array(depth_encoded))


            depth_vis = cv2.convertScaleAbs(final_depth, alpha=0.03)  
            # depth_writer.write(depth_vis)


            finally_depth_colormap = vis_depth(final_depth)
            depth_colormap_writer.write(finally_depth_colormap)

    finally:

        rgb_writer.release()
        # depth_writer.release()
        depth_colormap_writer.release()
        depth_h5_file.close()

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process Gaussian Splatting data")
    default_gs = PROJECT_ROOT / "assets" / "GS_Assets" / "WBC_Floor2_5_26.ply"
    default_intrinsics = PROJECT_ROOT / "assets" / "calibration" / "111_final_ex_0511.json"

    parser.add_argument('--input_hdf5',
                        default="",
                        help="Path to input HDF5 file, need with videos now")

    parser.add_argument('--episodes_to_process',
                        nargs='+',
                        type=int,
                        default=None,
                        help="List of episode indices to process (default: 0 1)")

    parser.add_argument('--model_path',
                        default=str(default_gs),
                        help="Path to Gaussian Splatting model file")

    parser.add_argument('--intrinsics_path',
                        default=str(default_intrinsics),
                        help="Path to camera intrinsics JSON file")

    parser.add_argument('--output_video_path_root',
                        default="/tmp/real2sim_render_test",
                        help="Root directory for output videos")

    parser.add_argument('--num_processes',
                        type=int,
                        default=8,
                        help="Number of worker processes")

    parser.add_argument('--scene_rot_z_deg',
                        type=float,
                        default=0.0,
                        help="Rotate the GS scene around +Z before rendering")

    parser.add_argument('--print_zbuffer_stats',
                        action='store_true',
                        help="Print per-frame foreground depth front-of-background ratio")

    return parser.parse_args()

def main():
    args = parse_args()

    input_hdf5 = args.input_hdf5
    episodes_to_process = args.episodes_to_process
    model_path = args.model_path
    intrinsics_path = args.intrinsics_path
    output_video_path_root = args.output_video_path_root
    print_zbuffer_stats = args.print_zbuffer_stats
    num_processes = args.num_processes
    scene_rot_z_deg = args.scene_rot_z_deg

    theta = np.deg2rad(scene_rot_z_deg)
    c, s = np.cos(theta), np.sin(theta)
    T = np.array([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    print(f"Main Python: Processing episodes {episodes_to_process} from {input_hdf5}")
    print(f"Main Python: Scene rotation around Z = {scene_rot_z_deg} deg")


    sh_params = {
        'dc_offset': (0.0, 0.0, 0.0),
        'rest_offset': (0.0, 0.0, 0.0)
    }


    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.empty_cache()


    os.makedirs(os.path.join(output_video_path_root, "gs_rgb_videos"), exist_ok=True)
    os.makedirs(os.path.join(output_video_path_root, "gs_depth_h5"), exist_ok=True)
    os.makedirs(os.path.join(output_video_path_root, "gs_depth_vis"), exist_ok=True)


    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(input_hdf5)
    all_episode_names = list(dataset_file_handler.get_episode_names())  
    dataset_file_handler.close()


    if episodes_to_process is not None:
        # episode_names = [all_episode_names[i] for i in episodes_to_process]
        episode_names = [f"demo_{str(i)}" for i in episodes_to_process]

        episode_names = [name for name in episode_names if name in all_episode_names]
        if not episode_names:
            print("No valid episodes found to process. Please check the episode indices.")
            return
        print(f"Processing specified episodes: {episode_names}")
    else:
        episode_names = all_episode_names

    # print(f"Found {len(episode_names)} episodes to process")


    if num_processes > len(episode_names):
        num_processes = len(episode_names)
    batch_size = max(1, len(episode_names) // num_processes)
    episode_batches = [episode_names[i:i + batch_size]
                      for i in range(0, len(episode_names), batch_size)]

    # print(f"Using {num_processes} processes with {len(episode_batches)} batches (~{batch_size} episodes/batch)")


    shared_args = (input_hdf5, output_video_path_root, model_path, intrinsics_path, sh_params, T, print_zbuffer_stats)


    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()


    pbar = tqdm(total=len(episode_names), desc="Processing episodes", unit="episode")


    def update_progress():
        while True:
            _ = progress_queue.get()
            if _ is None:  
                break
            pbar.update(1)

    progress_thread = threading.Thread(target=update_progress)
    progress_thread.start()


    if num_processes == 1:

        try:
            batch_results = [process_episode_batch(shared_args, batch, progress_queue=progress_queue) for batch in episode_batches]
            success_count = sum("Success" in res for batch in batch_results for res in batch)
            print(f"\nProcessed {success_count}/{len(episode_names)} episodes successfully")
            failures = [
                res for batch in batch_results for res in batch
                if "Failed" in res or "Process crash" in res
            ]
            if failures:
                print("\nFailed episodes:")
                for fail in failures[:10]:
                    print(f"  {fail}")
        except Exception as e:
            print(f"\nFatal error in main process: {str(e)}")
        finally:
            progress_queue.put(None)
            progress_thread.join()
            pbar.close()
    else:
        ctx = multiprocessing.get_context('spawn')
        with closing(ctx.Pool(processes=num_processes)) as pool:
            try:

                process_func = partial(process_episode_batch, shared_args, progress_queue=progress_queue)
                batch_results = list(pool.imap(process_func, episode_batches))


                success_count = sum("Success" in res for batch in batch_results for res in batch)
                print(f"\nProcessed {success_count}/{len(episode_names)} episodes successfully")


                failures = [
                    res for batch in batch_results for res in batch
                    if "Failed" in res or "Process crash" in res
                ]
                if failures:
                    print("\nFailed episodes:")
                    for fail in failures[:10]:  
                        print(f"  {fail}")

            except Exception as e:
                print(f"\nFatal error in main process: {str(e)}")
            finally:

                progress_queue.put(None)
                progress_thread.join()
                pbar.close()

    print("\nProcessing complete!")
    print(f"Output saved to: {output_video_path_root}")

if __name__ == "__main__":
    import threading
    multiprocessing.freeze_support()
    main()


