#!/usr/bin/env python3

#
# Run command:
#   python \
#     scripts/imitation_learning/robomimic/wbc_play_groot_sonic.py \
#     --enable_cameras \
#     --task Isaac-Final-Walk-PickPlace-Render-Abs-v0 \
#     --checkpoint_dir "${CHECKPOINT_DIR}" \
#     --gr00t_host 127.0.0.1 --gr00t_port 5555 \
#     --task_description "walk to the sink" \
#     --num_rollouts 10 --horizon 50 --chunk_horizon 40 --max_control_steps -1 \
#     --init_target_xy 0.3239 1.35047 \
#     --init_x_range -0.83371 1.65581 \
#     --init_y_range -0.65 0.35 \
#     --warmup_steps 0 \
#     --output_dir /tmp/wbc_play_groot_sonic_task1_walk_to_sink_test1_rollout10_h50 \
#     --log_every 30 --stop_on_fall --require_gs --show_obs
from __future__ import annotations

import argparse
import contextlib
import ctypes
import faulthandler
import importlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import zmq

_conda_lib_path = os.path.join(
    os.environ.get("CONDA_PREFIX", os.path.dirname(os.path.dirname(sys.executable))),
    f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/cmeel.prefix/lib/libassimp.so.5",
)
if os.path.exists(_conda_lib_path):
    try:
        ctypes.CDLL(_conda_lib_path, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        _conda_lib_dir = os.path.dirname(_conda_lib_path)
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if _conda_lib_dir not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = (
                f"{_conda_lib_dir}:{current_ld_path}" if current_ld_path else _conda_lib_dir
            )

from isaaclab.app import AppLauncher

faulthandler.enable(all_threads=True)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOMIMIC_DIR = Path(__file__).resolve().parent
SONIC_RUN_DIR = REPO_ROOT / "scripts" / "sonic_kimodo_run"
DEFAULT_TASK = "Isaac-Final-Walk-PickPlace-Render-Abs-v0"
DEFAULT_MODELS_DIR = REPO_ROOT.parent / "assets/shared/sonic"
DEFAULT_B1_GS_PLY = REPO_ROOT.parent / "assets/task1/assets/GS_Assets" / "B1_0716_aligned_0711.ply"
DEFAULT_HEADCAM_CALIB = REPO_ROOT.parent / "assets/task1/assets/calibration" / "111_final_ex_0511.json"
DEFAULT_TWO_D_GS_REAL2SIM_ROOT = Path(
    os.environ.get("TWO_D_GS_REAL2SIM_ROOT", REPO_ROOT / "submodule" / "2D_GS" / "Real2Sim")
)
DEFAULT_GS_ROOT = Path(os.environ.get("THREE_D_GS_ROOT", REPO_ROOT / "submodule" / "3DGS"))
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "wbc_play_groot_sonic"
DEFAULT_ROBOT_GS_ROOT = REPO_ROOT / "assets" / "robot_gs"
DEFAULT_ROBOT_GS_PARTS_DIR = DEFAULT_ROBOT_GS_ROOT / "part_ply_joints"
DEFAULT_ROBOT_GS_REFERENCE_PLY = DEFAULT_ROBOT_GS_ROOT / "full_body_recon.ply"
DEFAULT_ROBOT_LINK_JSON = DEFAULT_ROBOT_GS_ROOT / "urdf_pcd" / "g1_arm_link_T_base.json"
DEFAULT_ROBOT_URDF = REPO_ROOT / "submodule" / "GMR" / "assets" / "unitree_g1" / "g1_29dof_inspire.urdf"

FOREGROUND_SEMANTIC_RGB = np.array(
    [
        [255, 36, 66],
        [255, 184, 48],
        [55, 255, 139],
    ],
    dtype=np.int16,
)
OBJECT_PLACEMENT_SEMANTIC_RGB = FOREGROUND_SEMANTIC_RGB[1:].copy()
TRAINING_RGB_SCALE = np.array([0.902, 0.824, 0.831], dtype=np.float32)
TRAINING_RGB_OFFSET = np.array([11.6, 15.9, 9.0], dtype=np.float32)


def apply_training_rgb_color_match(rgb: np.ndarray) -> np.ndarray:
    """Apply the fixed RGB mapping used by the Task1 LeRobot training videos."""

    image = np.asarray(rgb, dtype=np.float32)
    return np.clip(image * TRAINING_RGB_SCALE + TRAINING_RGB_OFFSET, 0.0, 255.0).astype(np.uint8)

LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
WAIST_JOINT_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
HEAD_JOINT_NAMES = ["d455_joint", "xl330_joint"]
LEFT_LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]
RIGHT_LEG_JOINT_NAMES = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
LEFT_HAND_6D_TO_ACTIVE = [
    "left_little_1_joint",
    "left_ring_1_joint",
    "left_middle_1_joint",
    "left_index_1_joint",
    "left_thumb_2_joint",
    "left_thumb_1_joint",
]
RIGHT_HAND_6D_TO_ACTIVE = [
    "right_little_1_joint",
    "right_ring_1_joint",
    "right_middle_1_joint",
    "right_index_1_joint",
    "right_thumb_2_joint",
    "right_thumb_1_joint",
]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "envs") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "envs"))
if str(SONIC_RUN_DIR) not in sys.path:
    sys.path.insert(0, str(SONIC_RUN_DIR))


@dataclass
class ModalityConfig:
    delta_indices: list[int]
    modality_keys: list[str]
    sin_cos_embedding_keys: list[str] | None = None
    mean_std_embedding_keys: list[str] | None = None
    action_configs: list[Any] | None = None


@dataclass
class SonicActionChunk:
    token: np.ndarray
    left_hand_gripper: np.ndarray
    right_hand_gripper: np.ndarray

    @property
    def horizon(self) -> int:
        return int(self.token.shape[1])


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes, raw=False)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        nd_marker = obj.get("nd", obj.get(b"nd"))
        if nd_marker is True:
            dtype_value = obj.get("type", obj.get(b"type"))
            shape_value = obj.get("shape", obj.get(b"shape"))
            data_value = obj.get("data", obj.get(b"data"))
            if dtype_value is None or shape_value is None or data_value is None:
                raise ValueError(f"Malformed ndarray payload keys={sorted(repr(k) for k in obj.keys())}")
            dtype = np.dtype(dtype_value)
            if dtype.kind == "O":
                raise ValueError("Refusing to decode object-dtype ndarray payload")
            return np.frombuffer(data_value, dtype=dtype).reshape(tuple(shape_value)).copy()
        if "__ModalityConfig_class__" in obj or "__ModalityConfig__" in obj:
            return ModalityConfig(**obj["as_json"])
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, ModalityConfig):
            return {
                "__ModalityConfig__": True,
                "as_json": {
                    "delta_indices": list(obj.delta_indices),
                    "modality_keys": list(obj.modality_keys),
                },
            }
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind == "O":
                raise TypeError(f"Refusing to encode object-dtype ndarray with shape={obj.shape}")
            array = np.ascontiguousarray(obj)
            return {
                b"nd": True,
                b"type": array.dtype.str,
                b"kind": b"",
                b"shape": tuple(array.shape),
                b"data": array.tobytes(),
            }
        return obj


class RobotInferenceClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 30000):
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.context = zmq.Context()
        self.socket = self._connect_socket()
        self._endpoint_counts: dict[str, int] = {}

    def _connect_socket(self):
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.connect(f"tcp://{self.host}:{self.port}")
        return socket

    def _reconnect_socket(self) -> None:
        self.socket.close(linger=0)
        self.socket = self._connect_socket()

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        started = time.perf_counter()
        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            response = MsgSerializer.from_bytes(self.socket.recv())
        except zmq.Again as exc:
            # A timed-out REQ socket cannot send another request until its
            # missing reply is consumed. Reconnect so one overloaded request
            # does not turn every later episode into EFSM errors.
            self._reconnect_socket()
            raise TimeoutError(
                f"GR00T {endpoint} timed out after {self.timeout_ms} ms "
                f"(target=tcp://{self.host}:{self.port})"
            ) from exc
        except zmq.ZMQError:
            self._reconnect_socket()
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        count = self._endpoint_counts.get(endpoint, 0) + 1
        self._endpoint_counts[endpoint] = count
        if endpoint != "get_action" or count <= 3 or count % 20 == 0 or elapsed_ms > 1000.0:
            print(f"[GR00T] endpoint={endpoint} call={count} elapsed_ms={elapsed_ms:.1f}", flush=True)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def ping(self) -> dict[str, Any]:
        return self.call_endpoint("ping", requires_input=False)

    def reset(self) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": None})

    def get_modality_config(self) -> dict[str, Any]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint("get_action", {"observation": observation, "options": options})
        if not isinstance(response, (tuple, list)) or len(response) != 2:
            raise RuntimeError(f"Unexpected GR00T get_action response: {type(response)}")
        return dict(response[0]), dict(response[1])


class HeadCameraCalibration:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        with self.path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        intr = payload["camera_intrinsics"]
        self.width = int(intr["width"])
        self.height = int(intr["height"])
        self.k = np.asarray(intr["K"], dtype=np.float64).reshape(3, 3)
        self.relative_position_root_xyz = np.asarray(payload["relative_position_root_xyz"], dtype=np.float64).reshape(3)
        self.relative_quat_root_wxyz = quat_normalize_np(
            np.asarray(payload["relative_quat_root_wxyz"], dtype=np.float64).reshape(4)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GR00T N1.7 token actions through SONIC decoder-only policy in IsaacLab."
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--embodiment", type=str, default="new_embodiment")
    parser.add_argument("--gr00t_host", type=str, default="127.0.0.1")
    parser.add_argument("--gr00t_port", type=int, default=5555)
    parser.add_argument("--gr00t_timeout_ms", type=int, default=30000)
    parser.add_argument("--task_description", type=str, default="walk forward naturally")
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=50, help="Number of GR00T chunk queries per rollout.")
    parser.add_argument(
        "--chunk_horizon",
        "--gr00t_action_horizon",
        dest="chunk_horizon",
        type=int,
        default=40,
        help="Maximum token frames to consume from each returned GR00T chunk. Negative consumes the full chunk.",
    )
    parser.add_argument(
        "--max_control_steps",
        type=int,
        default=-1,
        help="Optional exact rollout length in consumed 50 Hz control steps. Negative disables this cap.",
    )
    parser.add_argument("--control_hz", type=float, default=50.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--fall_height", type=float, default=0.35)
    parser.add_argument("--stop_on_fall", action="store_true")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--enable_head_hand", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--head_lock_rad",
        type=float,
        default=-0.15,
        help="Replay-compatible target for d455_joint and xl330_joint.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models_dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--sonic_cpu_onnx", action="store_true")
    parser.add_argument("--sonic_actuator_profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sonic_waist_stiffness_scale", type=float, default=None)
    parser.add_argument("--clip_body_targets_to_limits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reset_to_sonic_default",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the SONIC runner by resetting the 29 SONIC body joints to the official G1 default pose.",
    )
    parser.add_argument("--random_init_pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init_target_xy", type=float, nargs=2, default=(0.3239, 1.35047))
    parser.add_argument("--init_x_range", type=float, nargs=2, default=(-0.83371, 1.65581))
    parser.add_argument("--init_y_range", type=float, nargs=2, default=(-1.811475, 1.118935))
    parser.add_argument("--init_yaw_jitter_deg", type=float, nargs=2, default=(-15.0, 15.0))
    parser.add_argument("--init_robot_z", type=float, default=0.8)
    parser.add_argument("--init_min_start_distance_m", type=float, default=0.05)
    parser.add_argument("--headcam_calib", type=Path, default=DEFAULT_HEADCAM_CALIB)
    parser.add_argument(
        "--headcam_extrinsic_mode",
        choices=("root_calib_attached_local", "task_cfg"),
        default="root_calib_attached_local",
        help=(
            "root_calib_attached_local matches replay: keep the camera under the robot camera link and author one "
            "calibrated local xform after reset. task_cfg leaves the task camera extrinsic unchanged."
        ),
    )
    parser.add_argument("--enable_gs_rollout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gs_scene_path", type=Path, default=DEFAULT_B1_GS_PLY)
    parser.add_argument("--gs_backend", choices=("auto", "2dgs", "3dgs"), default="auto")
    parser.add_argument("--gs_root", type=Path, default=DEFAULT_GS_ROOT)
    parser.add_argument("--two_d_gs_real2sim_root", type=Path, default=DEFAULT_TWO_D_GS_REAL2SIM_ROOT)
    parser.add_argument("--robot_gs_parts_dir", type=Path, default=DEFAULT_ROBOT_GS_PARTS_DIR)
    parser.add_argument("--robot_gs_reference_ply", type=Path, default=DEFAULT_ROBOT_GS_REFERENCE_PLY)
    parser.add_argument("--robot_link_json", type=Path, default=DEFAULT_ROBOT_LINK_JSON)
    parser.add_argument("--robot_urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--gs_semantic_tolerance", type=int, default=12)
    parser.add_argument(
        "--gs_apply_training_color_match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the fixed RGB mapping used by the color-matched Task1 training videos.",
    )
    parser.add_argument("--require_gs", action="store_true")
    parser.add_argument("--save_obs_video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show_obs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show_obs_window", type=str, default="groot_sonic_obs")
    parser.add_argument("--show_obs_wait_ms", type=int, default=1)
    parser.add_argument(
        "--camera_warmup_renders",
        type=int,
        default=3,
        help="Render this many frames after each rollout reset before the first GR00T observation.",
    )
    parser.add_argument("--save_metrics_npz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--schema_only", action="store_true")
    parser.add_argument("--server_smoke_only", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if (args.enable_gs_rollout or "Render" in str(args.task)) and not getattr(args, "enable_cameras", False):
        args.enable_cameras = True
        print("[INFO] auto-enabled --enable_cameras for render/GS rollout.", flush=True)
    return args


def modality_delta_indices(config: ModalityConfig | dict[str, Any]) -> list[int]:
    if isinstance(config, dict):
        if "as_json" in config and isinstance(config["as_json"], dict):
            config = config["as_json"]
        return list(config.get("delta_indices", []))
    return list(config.delta_indices)


def modality_keys(config: ModalityConfig | dict[str, Any]) -> list[str]:
    if isinstance(config, dict):
        if "as_json" in config and isinstance(config["as_json"], dict):
            config = config["as_json"]
        return list(config.get("modality_keys", []))
    return list(config.modality_keys)


def load_processor_modality_configs(checkpoint_dir: Path, embodiment: str) -> dict[str, Any]:
    config_path = checkpoint_dir.expanduser().resolve() / "processor_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    modality_configs = payload["processor_kwargs"]["modality_configs"][embodiment]
    return modality_configs


def summarize_modality_config(configs: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "delta_indices": modality_delta_indices(cfg),
            "modality_keys": modality_keys(cfg),
        }
        for name, cfg in configs.items()
    }


def quat_normalize_np(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError(f"Invalid near-zero quaternion: {quat}")
    quat = quat / norm
    if quat[0] < 0.0:
        quat = -quat
    return quat


def quat_multiply_np(left_wxyz: np.ndarray, right_wxyz: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = quat_normalize_np(left_wxyz)
    rw, rx, ry, rz = quat_normalize_np(right_wxyz)
    return quat_normalize_np(
        np.array(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float64,
        )
    )


def quat_rotate_np(quat_wxyz: np.ndarray, vector_xyz: np.ndarray) -> np.ndarray:
    quat = quat_normalize_np(quat_wxyz)
    vector = np.asarray(vector_xyz, dtype=np.float64).reshape(3)
    q_vec = quat[1:]
    q_w = quat[0]
    return (
        2.0 * float(np.dot(q_vec, vector)) * q_vec
        + (q_w * q_w - float(np.dot(q_vec, q_vec))) * vector
        + 2.0 * q_w * np.cross(q_vec, vector)
    )


def yaw_to_quat_wxyz_np(yaw_rad: float) -> np.ndarray:
    half = 0.5 * float(yaw_rad)
    return quat_normalize_np(np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64))


def sample_start_pose(args: argparse.Namespace, rollout_index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = random.Random(int(args.seed) + int(rollout_index))
    x0, x1 = sorted(float(v) for v in args.init_x_range)
    y0, y1 = sorted(float(v) for v in args.init_y_range)
    yaw0, yaw1 = sorted(float(v) for v in args.init_yaw_jitter_deg)
    target = np.asarray(args.init_target_xy, dtype=np.float64).reshape(2)
    for _ in range(10_000):
        x = rng.uniform(x0, x1)
        y = rng.uniform(y0, y1)
        if math.hypot(x - target[0], y - target[1]) < float(args.init_min_start_distance_m):
            continue
        base_yaw = math.atan2(target[1] - y, target[0] - x)
        yaw_jitter = rng.uniform(yaw0, yaw1)
        yaw = base_yaw + math.radians(yaw_jitter)
        root_pos = np.array([x, y, float(args.init_robot_z)], dtype=np.float64)
        root_quat = yaw_to_quat_wxyz_np(yaw)
        return root_pos, root_quat, {
            "target_xy": target.astype(float).tolist(),
            "start_xy": [float(x), float(y)],
            "base_yaw_rad": float(base_yaw),
            "yaw_jitter_deg": float(yaw_jitter),
            "yaw_rad": float(yaw),
        }
    raise RuntimeError("Could not sample a valid rollout start pose")


def reset_robot_root(env, robot, root_pos: np.ndarray, root_quat_wxyz: np.ndarray) -> None:
    import torch

    root_pose = robot.data.root_state_w[:, :7].clone()
    root_pose[:, :3] = torch.as_tensor(root_pos, device=root_pose.device, dtype=root_pose.dtype).view(1, 3)
    root_pose[:, 3:7] = torch.as_tensor(root_quat_wxyz, device=root_pose.device, dtype=root_pose.dtype).view(1, 4)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.root_state_w[:, 7:13]))
    robot.write_joint_state_to_sim(robot.data.joint_pos.clone(), torch.zeros_like(robot.data.joint_vel))
    env.scene.update(dt=env.physics_dt)


def reset_body_to_sonic_default(robot, body_joint_sim_ids_t, sonic_default: np.ndarray, device) -> None:
    import torch

    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    default_t = torch.as_tensor(sonic_default, device=device, dtype=joint_pos.dtype)
    joint_pos[:, body_joint_sim_ids_t] = default_t.unsqueeze(0).expand(joint_pos.shape[0], -1)
    joint_vel[:, body_joint_sim_ids_t] = 0.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)


def robot_root_pose_np(env, env_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
    robot = env.scene["robot"]
    root_state = robot.data.root_state_w.detach().cpu().numpy().reshape(-1, 13)[int(env_id)]
    return root_state[:3].astype(np.float64), quat_normalize_np(root_state[3:7].astype(np.float64))


def camera_pose_from_root(env, calib: HeadCameraCalibration, env_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
    root_pos, root_quat = robot_root_pose_np(env, env_id=env_id)
    cam_pos = root_pos + quat_rotate_np(root_quat, calib.relative_position_root_xyz)
    cam_quat = quat_multiply_np(root_quat, calib.relative_quat_root_wxyz)
    return cam_pos, cam_quat


def get_head_camera(env):
    if hasattr(env.scene, "sensors") and "head_camera" in env.scene.sensors:
        return env.scene.sensors["head_camera"]
    return env.scene["head_camera"]


def quat_conjugate_np(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = quat_normalize_np(quat_wxyz)
    return np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def camera_world_to_opengl_quat_np(quat_wxyz: np.ndarray) -> np.ndarray:
    import torch
    from isaaclab.utils.math import convert_camera_frame_orientation_convention

    quat_tensor = torch.as_tensor(quat_normalize_np(quat_wxyz), dtype=torch.float32).reshape(1, 4)
    return (
        convert_camera_frame_orientation_convention(quat_tensor, origin="world", target="opengl")
        .detach()
        .cpu()
        .numpy()
        .reshape(4)
        .astype(np.float64)
    )


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def usd_quat_to_np(quat) -> np.ndarray:
    imag = quat.GetImaginary()
    return quat_normalize_np(np.asarray([quat.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float64))


def prim_local_pose_np(prim) -> tuple[np.ndarray, np.ndarray]:
    from pxr import UsdGeom

    result = UsdGeom.Xformable(prim).GetLocalTransformation()
    transform = result[0] if isinstance(result, tuple) else result
    pos = transform.ExtractTranslation()
    quat = transform.ExtractRotationQuat()
    return np.asarray([pos[0], pos[1], pos[2]], dtype=np.float64), usd_quat_to_np(quat)


def compose_pose_np(
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    local_pos: np.ndarray,
    local_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3)
    base_quat = quat_normalize_np(base_quat)
    local_pos = np.asarray(local_pos, dtype=np.float64).reshape(3)
    local_quat = quat_normalize_np(local_quat)
    return base_pos + quat_rotate_np(base_quat, local_pos), quat_multiply_np(base_quat, local_quat)


def attached_camera_parent_pose_from_robot_data(
    env,
    camera_prim,
    env_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    robot = env.scene["robot"]
    body_names = list(getattr(robot.data, "body_names", []))
    if not body_names:
        raise RuntimeError("robot.data.body_names is unavailable")

    parent_prim = camera_prim.GetParent()
    ancestor = parent_prim
    child_chain = []
    while ancestor and ancestor.IsValid() and ancestor.GetName() not in body_names:
        child_chain.append(ancestor)
        ancestor = ancestor.GetParent()
    if not ancestor or not ancestor.IsValid():
        raise RuntimeError(f"Could not find robot body ancestor for camera parent {parent_prim.GetPath()}")

    body_name = ancestor.GetName()
    body_idx = body_names.index(body_name)
    body_pos = _to_numpy(robot.data.body_pos_w)[int(env_id), body_idx].astype(np.float64)
    body_quat = quat_normalize_np(_to_numpy(robot.data.body_quat_w)[int(env_id), body_idx].astype(np.float64))

    parent_pos = body_pos
    parent_quat = body_quat
    for child_prim in reversed(child_chain):
        local_pos, local_quat = prim_local_pose_np(child_prim)
        parent_pos, parent_quat = compose_pose_np(parent_pos, parent_quat, local_pos, local_quat)
    return parent_pos, parent_quat


def set_prim_local_pose(prim, pos_xyz: np.ndarray, quat_wxyz: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    orient_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_op is None:
            orient_op = op
    if translate_op is None:
        translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    if orient_op is None:
        orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)

    pos = np.asarray(pos_xyz, dtype=np.float64).reshape(3)
    quat = quat_normalize_np(np.asarray(quat_wxyz, dtype=np.float64).reshape(4))
    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    orient_op.Set(Gf.Quatd(float(quat[0]), Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3]))))


def apply_head_camera_local_from_root_calib(env, calib: HeadCameraCalibration, *, render: bool = False) -> None:
    camera = get_head_camera(env)
    sensor_prims = getattr(camera, "_sensor_prims", None)
    if not sensor_prims:
        raise RuntimeError("head_camera sensor does not expose _sensor_prims; cannot set attached local xform")
    count = int(env.scene.env_origins.shape[0])
    for env_id in range(count):
        if env_id >= len(sensor_prims):
            raise RuntimeError(f"head_camera has {len(sensor_prims)} sensor prims; env_id={env_id} is unavailable")
        sensor_prim = sensor_prims[env_id]
        camera_prim = sensor_prim.GetPrim() if hasattr(sensor_prim, "GetPrim") else sensor_prim
        parent_pos, parent_quat = attached_camera_parent_pose_from_robot_data(
            env,
            camera_prim,
            env_id=env_id,
        )
        desired_pos, desired_quat_world = camera_pose_from_root(env, calib, env_id=env_id)
        desired_quat_opengl = camera_world_to_opengl_quat_np(desired_quat_world)
        parent_quat_inv = quat_conjugate_np(parent_quat)
        local_pos = quat_rotate_np(parent_quat_inv, desired_pos - parent_pos)
        local_quat = quat_multiply_np(parent_quat_inv, desired_quat_opengl)
        set_prim_local_pose(camera_prim, local_pos, local_quat)
    if render:
        env.sim.render()
        env.scene.update(dt=env.physics_dt)
        with contextlib.suppress(Exception):
            camera.update(dt=0.0, force_recompute=True)


def apply_head_camera_world_from_root_calib(env, calib: HeadCameraCalibration, *, render: bool = False) -> None:
    camera = get_head_camera(env)
    count = int(env.scene.env_origins.shape[0])
    positions = np.empty((count, 3), dtype=np.float32)
    orientations = np.empty((count, 4), dtype=np.float32)
    for env_id in range(count):
        position, orientation = camera_pose_from_root(env, calib, env_id=env_id)
        positions[env_id] = position
        orientations[env_id] = orientation

    # the world-to-local conversion and camera-frame convention conversion.
    camera.set_world_poses(
        positions=positions,
        orientations=orientations,
        convention="world",
    )
    if render:
        env.sim.render()
        env.scene.update(dt=env.physics_dt)
        with contextlib.suppress(Exception):
            camera.update(dt=0.0, force_recompute=True)


def apply_head_camera_mode(env, calib: HeadCameraCalibration, extrinsic_mode: str, *, render: bool = True) -> None:
    if extrinsic_mode == "root_calib_attached_local":
        apply_head_camera_local_from_root_calib(env, calib, render=render)
        # Isaac and GS both consume the TiledCamera pose produced above.
        with contextlib.suppress(AttributeError):
            del env._humanoidvla_headcam_root_calib
    elif extrinsic_mode == "root_calib_each_frame":
        apply_head_camera_world_from_root_calib(env, calib, render=render)
    elif extrinsic_mode in {"task_cfg", "legacy_task_cfg"}:
        if render:
            env.sim.render()
            env.scene.update(dt=env.physics_dt)
    else:
        raise ValueError(f"Unsupported headcam_extrinsic_mode: {extrinsic_mode}")


def refresh_head_camera_output(env) -> None:
    camera = get_head_camera(env)
    camera.update(dt=0.0, force_recompute=True)


def validate_head_camera_output(env) -> None:
    camera = get_head_camera(env)
    outputs = camera.data.output
    required = ("rgb", "distance_to_image_plane", "semantic_segmentation")
    missing = [name for name in required if name not in outputs]
    if missing:
        raise RuntimeError(f"Head camera is missing required outputs after render: {missing}")
    invalid = []
    for name in required:
        value = outputs[name]
        if value is None or getattr(value, "numel", lambda: 0)() == 0:
            invalid.append(name)
    if invalid:
        raise RuntimeError(f"Head camera returned empty outputs after render: {invalid}")


def wait_for_head_camera_output(env, render_count: int = 1, timeout_s: float = 60.0) -> None:
    """Block without stepping physics until consecutive camera renders are valid."""

    valid_renders = 0
    failed_attempts = 0
    deadline = time.monotonic() + float(timeout_s)
    last_error: Exception | None = None
    while valid_renders < max(1, int(render_count)):
        try:
            env.sim.render()
            env.scene.update(dt=env.physics_dt)
            refresh_head_camera_output(env)
            validate_head_camera_output(env)
        except Exception as exc:
            last_error = exc
            failed_attempts += 1
            valid_renders = 0
            if failed_attempts == 1 or failed_attempts % 50 == 0:
                print(
                    f"[OBS] waiting for head-camera render; attempts={failed_attempts} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Head camera did not produce valid output within {timeout_s:.1f}s "
                    f"after {failed_attempts} attempts; last error={type(exc).__name__}: {exc}"
                ) from last_error
            time.sleep(0.01)
            continue
        valid_renders += 1


def warmup_head_camera_output(env, render_count: int) -> None:
    wait_for_head_camera_output(env, render_count=render_count)


def patch_head_camera_for_gs(env_cfg, enable_gs: bool) -> None:
    if not enable_gs:
        return
    scene = getattr(env_cfg, "scene", None)
    head_camera_cfg = getattr(scene, "head_camera", None) if scene is not None else None
    if head_camera_cfg is None:
        raise RuntimeError("GS rollout requested but env_cfg.scene.head_camera is missing.")
    required = ["rgb", "distance_to_image_plane", "semantic_segmentation"]
    data_types = list(getattr(head_camera_cfg, "data_types", []))
    for item in required:
        if item not in data_types:
            data_types.append(item)
    head_camera_cfg.data_types = data_types
    if not getattr(head_camera_cfg, "semantic_segmentation_mapping", None):
        head_camera_cfg.semantic_segmentation_mapping = {
            "class:robot": (255, 36, 66, 255),
            "class:target_placement": (255, 184, 48, 255),
            "class:target_object": (55, 255, 139, 255),
            "class:UNLABELLED": (128, 128, 128, 255),
            "class:BACKGROUND": (12, 12, 12, 255),
        }
    if hasattr(head_camera_cfg, "colorize_semantic_segmentation"):
        head_camera_cfg.colorize_semantic_segmentation = True


def patch_head_camera_calibration(env_cfg, calib: HeadCameraCalibration | None) -> None:
    if calib is None:
        return
    import isaaclab.sim as sim_utils

    scene = getattr(env_cfg, "scene", None)
    head_camera_cfg = getattr(scene, "head_camera", None) if scene is not None else None
    if head_camera_cfg is None:
        return
    head_camera_cfg.width = calib.width
    head_camera_cfg.height = calib.height
    head_camera_cfg.update_latest_camera_pose = True
    head_camera_cfg.spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=calib.k.reshape(-1).tolist(),
        width=calib.width,
        height=calib.height,
        focal_length=1.93,
        focus_distance=0.5,
        clipping_range=(0.01, 100.0),
        lock_camera=False,
    )
    print(f"[INFO] Patched head-camera intrinsics from calibration: {calib.path}", flush=True)


def detect_gs_backend_from_ply(path: Path) -> str:
    scale_fields = []
    with path.open("rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"Unsupported GS model format: {path}")
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {path}")
            text = line.decode("ascii", errors="replace").strip()
            if text == "end_header":
                break
            if text.startswith("property ") and " scale_" in text:
                scale_fields.append(text.rsplit(" ", 1)[-1])
    if len(scale_fields) == 2:
        return "2dgs"
    if len(scale_fields) == 3:
        return "3dgs"
    raise ValueError(f"Could not infer GS backend from {path}; scale fields={scale_fields}")


def semantic_foreground_mask(
    semantic_rgb: np.ndarray,
    tolerance: int,
    colors: np.ndarray = FOREGROUND_SEMANTIC_RGB,
) -> np.ndarray:
    semantic = np.asarray(semantic_rgb, dtype=np.int16)[..., :3]
    masks = [
        np.max(np.abs(semantic - color.reshape(1, 1, 3)), axis=-1) <= int(tolerance)
        for color in np.asarray(colors, dtype=np.int16)
    ]
    return np.any(np.stack(masks, axis=0), axis=0)


def camera_pose_for_2dgs(
    zbuf: Any,
    position_w: np.ndarray,
    quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    """Convert Isaac camera pose into the 2DGS adapter convention.

    The default ``legacy`` mode delegates to the bundled adapter. ``raw``
    passes through the pose, and ``corrected`` applies an explicit basis
    conversion. Keep the default for the released evaluation profiles.
    """

    mode = os.environ.get("HUMANOIDVLA_GS_POSE_MODE", "legacy").strip().lower()
    quat_xyzw = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)[[1, 2, 3, 0]]
    position = np.asarray(position_w, dtype=np.float64).reshape(3)
    if mode == "legacy":
        return zbuf.parse_pose_file(position.tolist(), quat_xyzw.tolist(), False)
    if mode == "raw":
        return position, quat_xyzw.tolist()
    if mode == "corrected":
        from scipy.spatial.transform import Rotation

        original = Rotation.from_quat(quat_xyzw)
        basis = Rotation.from_euler("y", 90.0, degrees=True) * Rotation.from_euler("x", -90.0, degrees=True)
        converted = (original.inv() * basis).inv().as_quat()
        return position, converted.tolist()
    raise ValueError(f"Unknown HUMANOIDVLA_GS_POSE_MODE={mode!r}; expected legacy, raw, or corrected")


def gs_scene_transform_from_env() -> np.ndarray:
    """Read an optional diagnostic scene SE(2) transform for 2DGS."""

    raw = os.environ.get("HUMANOIDVLA_GS_SCENE_SE2", "").strip()
    if not raw:
        return np.eye(4, dtype=np.float32)
    try:
        tx, ty, yaw = (float(value) for value in raw.split(","))
    except ValueError as exc:
        raise ValueError("HUMANOIDVLA_GS_SCENE_SE2 must be 'tx,ty,yaw_rad'") from exc
    c, s = np.cos(yaw), np.sin(yaw)
    transform = np.eye(4, dtype=np.float32)
    transform[:2, :2] = ((c, -s), (s, c))
    transform[0, 3] = tx
    transform[1, 3] = ty
    return transform


class Online2DGSRenderer:
    def __init__(
        self,
        gs_ply: Path,
        camera,
        real2sim_root: Path,
        semantic_tolerance: int,
        *,
        foreground_mode: str = "isaac_mesh",
        robot_parts_dir: Path,
        robot_reference_ply: Path,
        robot_link_json: Path,
        robot_urdf: Path,
        robot_gs_root: Path,
        apply_training_color_match: bool = False,
    ):
        real2sim_root = real2sim_root.expanduser().resolve()
        module_path = real2sim_root / "z_buffer_merger_with_robot.py"
        if not module_path.is_file():
            raise FileNotFoundError(f"2DGS z-buffer module not found: {module_path}")
        two_d_root = real2sim_root.parent / "2d-gaussian-splatting"

        def clear_and_pin_2dgs_modules() -> None:
            for module_name in list(sys.modules.keys()):
                if module_name.startswith(("scene", "gaussian_renderer", "utils", "arguments", "diff_surfel_rasterization")):
                    sys.modules.pop(module_name, None)
            root_text = str(two_d_root)
            if root_text in sys.path:
                sys.path.remove(root_text)
            sys.path.insert(0, root_text)
            for package_name in ("utils", "scene"):
                package = types.ModuleType(package_name)
                package.__path__ = [str(two_d_root / package_name)]
                package.__package__ = package_name
                sys.modules[package_name] = package

        if two_d_root.is_dir():
            clear_and_pin_2dgs_modules()
        if str(real2sim_root) not in sys.path:
            sys.path.insert(0, str(real2sim_root))
        spec = importlib.util.spec_from_file_location("humanoidvla_online_2dgs_zbuffer", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load 2DGS z-buffer module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.zbuf = module
        if two_d_root.is_dir():
            module.TWO_D_GS_ROOT = str(two_d_root)
            module._clear_gs_modules = clear_and_pin_2dgs_modules
            # The compatibility adapter re-exports functions from ``_base``;
            # those functions retain the base module's globals. Pin both
            # modules so its cache reset cannot expose the repository-level
            # ``utils`` package instead of the 2DGS implementation.
            if hasattr(module, "_base"):
                module._base.TWO_D_GS_ROOT = str(two_d_root)
                module._base._clear_gs_modules = clear_and_pin_2dgs_modules
                module._base._BACKEND_CACHE.clear()
        height, width = camera.image_shape
        intrinsic = camera.data.intrinsic_matrices.reshape(-1, 3, 3)[0].detach().cpu().numpy().astype(np.float32)
        intrinsics = module.camera_intrinsics_from_matrix(intrinsic, width, height)
        self.renderer = module.GSBackgroundRenderer(
            model_path=str(gs_ply.expanduser().resolve()),
            intrinsics=intrinsics,
            # Keep an untransformed copy of the scene model.  Evaluation can
            # then switch the calibrated Task1 scene SE(2) between episodes
            # without reloading the 2DGS PLY or rebuilding its CUDA backend.
            T=None,
        )
        self._scene_base_xyz = self.renderer.gs._xyz.detach().clone()
        self._scene_base_rotation = self.renderer.gs._rotation.detach().clone()
        self._scene_transform = np.eye(4, dtype=np.float32)
        self.set_scene_transform(gs_scene_transform_from_env())
        if foreground_mode not in {"robot_gs", "isaac_mesh"}:
            raise ValueError(f"Unsupported GS foreground mode: {foreground_mode}")
        self.foreground_mode = foreground_mode
        self.apply_training_color_match = bool(apply_training_color_match)
        self.robot_renderer = None
        self.robot_assembler = None
        if self.foreground_mode == "robot_gs":
            required_robot_assets = (robot_parts_dir, robot_reference_ply, robot_link_json, robot_urdf)
            missing = [str(path) for path in required_robot_assets if not path.expanduser().resolve().exists()]
            if missing:
                raise FileNotFoundError(f"Robot/hand GS assets are missing: {missing}")
            robot_gs_root = robot_gs_root.expanduser().resolve()
            if not robot_gs_root.is_dir():
                raise FileNotFoundError(f"3DGS backend root not found: {robot_gs_root}")
            # The scene PLY is 2DGS while the articulated robot parts are
            # standard 3DGS. Keep their import/runtime backends independent;
            # sharing the scene backend interprets the robot's third scale
            # component with the wrong rasterizer.
            module._base.THREE_D_GS_ROOT = str(robot_gs_root)
            module._base._BACKEND_CACHE.pop("3dgs", None)
            self.robot_renderer = module.GSBackgroundRenderer(
                model_path=str(robot_reference_ply.expanduser().resolve()),
                intrinsics=intrinsics,
            )
            self.robot_assembler = module.RobotArmGSAssembler(
                gaussian_model_cls=self.robot_renderer.backend["GaussianModel"],
                parts_dir=robot_parts_dir.expanduser().resolve(),
                link_json=robot_link_json.expanduser().resolve(),
                urdf=robot_urdf.expanduser().resolve(),
            )
        self.semantic_tolerance = int(semantic_tolerance)
        self._debug_dumped = False
        print(
            "[INFO] GS layer backends: "
            f"scene={self.renderer.backend_name} "
            f"robot={None if self.robot_renderer is None else self.robot_renderer.backend_name} "
            f"training_color_match={self.apply_training_color_match}",
            flush=True,
        )

    def set_scene_transform(self, transform: np.ndarray) -> None:
        """Apply an absolute world transform to the cached 2DGS scene model.

        The production Task1 replay stores a distinct static-scene SE(2) per
        episode.  ``GSBackgroundRenderer`` applies its ``T`` argument only at
        construction, which used to leave a long-lived evaluator background in
        the first episode's frame.  Rebuild the transformed Gaussian tensors
        from the cached canonical model instead of accumulating transforms or
        reloading the PLY.
        """

        matrix = np.asarray(transform, dtype=np.float32).reshape(4, 4)
        if not np.isfinite(matrix).all():
            raise ValueError("GS scene transform must be finite")
        torch = self.zbuf.torch
        transform_t = torch.as_tensor(matrix, device=self._scene_base_xyz.device, dtype=self._scene_base_xyz.dtype)
        rotation = transform_t[:3, :3]
        translation = transform_t[:3, 3]
        with torch.no_grad():
            self.renderer.gs._xyz.copy_(self._scene_base_xyz @ rotation.T + translation)
            rotation_quat = self.zbuf._matrix_to_quaternion(rotation)
            transformed_rotation = self.zbuf._quaternion_multiply(
                rotation_quat.expand_as(self._scene_base_rotation),
                self.zbuf.F.normalize(self._scene_base_rotation, dim=-1),
            )
            self.renderer.gs._rotation.copy_(transformed_rotation)
        self._scene_transform = matrix.copy()

    def set_scene_se2(self, translation_xy: np.ndarray | tuple[float, float], delta_yaw: float) -> None:
        """Set the canonical 2DGS background to a Task1 scene SE(2)."""

        translation_xy = np.asarray(translation_xy, dtype=np.float32).reshape(2)
        yaw = float(delta_yaw)
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        transform = np.eye(4, dtype=np.float32)
        transform[:2, :2] = ((c, -s), (s, c))
        transform[:2, 3] = translation_xy
        self.set_scene_transform(transform)

    def _finalize_rgb(self, merged_bgr: np.ndarray) -> np.ndarray:
        rgb = np.asarray(merged_bgr, dtype=np.uint8)[..., ::-1].copy()
        if self.apply_training_color_match:
            rgb = apply_training_rgb_color_match(rgb)
        return rgb

    def render_env(self, *, env, env_id: int, robot, sim_joint_names: list[str]) -> np.ndarray:
        root_calib = getattr(env, "_humanoidvla_headcam_root_calib", None)
        camera_pose_override = (
            camera_pose_from_root(env, root_calib, env_id=int(env_id))
            if root_calib is not None
            else None
        )
        frame, _ = gs_render_from_env(
            self,
            get_head_camera(env),
            robot=robot,
            sim_joint_names=sim_joint_names,
            env_id=env_id,
            env_origin_w=env.scene.env_origins[int(env_id)].detach().cpu().numpy(),
            camera_pose_override=camera_pose_override,
        )
        return frame

    def robot_attrs(self, *, root_state_w: np.ndarray, joint_names: list[str], joint_pos: np.ndarray):
        if self.robot_assembler is None:
            return None
        return self.robot_assembler.frame_attrs(
            np.asarray(root_state_w, dtype=np.float32)[:7],
            list(joint_names),
            np.asarray(joint_pos, dtype=np.float32),
        )

    def _render_robot_layer(self, camera, robot_attrs):
        robot_model = self.robot_renderer.build_robot_model(robot_attrs)
        black_pkg = self.robot_renderer.render(camera, gs_model=robot_model)
        black_rgb = black_pkg["render"].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        inverse_depth = black_pkg["depth_m"].detach().squeeze().cpu().numpy()
        black_valid = black_pkg["valid_mask"].detach().squeeze().cpu().numpy().astype(bool)
        del black_pkg
        white_bg = self.zbuf.torch.ones(3, dtype=self.zbuf.torch.float32, device="cuda")
        if self.robot_renderer.backend_name == "2dgs" and not hasattr(self.robot_renderer.pipe, "depth_ratio"):
            self.robot_renderer.pipe.depth_ratio = 0.0
        white_pkg = self.robot_renderer.render_fn(
            camera,
            robot_model,
            self.robot_renderer.pipe,
            white_bg,
        )
        white_rgb = white_pkg["render"].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        del white_pkg, robot_model
        alpha = np.clip(1.0 - np.mean(white_rgb - black_rgb, axis=-1), 0.0, 1.0)
        depth_m = np.full_like(inverse_depth, camera.zfar, dtype=np.float32)
        np.divide(1.0, inverse_depth, out=depth_m, where=inverse_depth > 1.0e-6)
        valid = black_valid & (alpha > 0.02)
        return black_rgb, alpha, depth_m, valid

    def render_with_gs_bg(
        self,
        *,
        pos_: np.ndarray,
        quat_w_world: np.ndarray,
        isaac_render_rgb: np.ndarray,
        isaac_render_depth: np.ndarray,
        isaac_render_semantic_images: np.ndarray,
        robot_attrs: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.foreground_mode == "robot_gs" and robot_attrs is None:
            raise RuntimeError("Online 2DGS observation requires robot/hand GS attributes")
        rgb = np.asarray(isaac_render_rgb, dtype=np.uint8)[..., :3]
        semantic = np.asarray(isaac_render_semantic_images)[..., :3]
        height, width = rgb.shape[:2]
        fg_mask = semantic_foreground_mask(
            semantic,
            self.semantic_tolerance,
            colors=(
                FOREGROUND_SEMANTIC_RGB
                if self.foreground_mode == "isaac_mesh"
                else OBJECT_PLACEMENT_SEMANTIC_RGB
            ),
        )
        debug_dir = os.environ.get("HUMANOIDVLA_GS_DEBUG_DIR")
        if debug_dir and not self._debug_dumped:
            import cv2

            out_dir = Path(debug_dir).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / "isaac_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / "isaac_semantic.png"), cv2.cvtColor(semantic, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / "foreground_mask.png"), fg_mask.astype(np.uint8) * 255)
            np.save(out_dir / "semantic_unique_rgba.npy", np.unique(np.asarray(isaac_render_semantic_images).reshape(-1, np.asarray(isaac_render_semantic_images).shape[-1]), axis=0))
            np.savez(
                out_dir / "camera_pose_input.npz",
                position_w=np.asarray(pos_, dtype=np.float64),
                quat_wxyz=np.asarray(quat_w_world, dtype=np.float64),
            )
            self._debug_dumped = True
        translation, rotation_xyzw = camera_pose_for_2dgs(self.zbuf, pos_, quat_w_world)
        cam = self.zbuf.MiniCam(
            width=self.renderer.intrinsics["width"],
            height=self.renderer.intrinsics["height"],
            focal_x=self.renderer.intrinsics["fx"],
            focal_y=self.renderer.intrinsics["fy"],
            pose=self.zbuf.pose_to_matrix(translation, rotation_xyzw),
        )
        render_pkg = self.renderer.render(cam)
        bg_rgb = (render_pkg["render"].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        background_rgb_transform = getattr(self, "background_rgb_transform", None)
        if background_rgb_transform is not None:
            bg_rgb = np.asarray(background_rgb_transform(bg_rgb), dtype=np.uint8)
        bg_bgr = bg_rgb[..., ::-1]
        bg_depth_m = render_pkg["depth_m"].detach().squeeze().cpu().numpy()
        valid_mask = render_pkg["valid_mask"].detach().squeeze().cpu().numpy().astype(bool)
        del render_pkg
        bg_depth_mm = np.clip(bg_depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        bg_depth_mm[~valid_mask] = np.iinfo(np.uint16).max
        # Compose semantic foreground without cross-renderer depth tests.
        # The observation contract is GS background -> Isaac mesh foreground.
        merged_bgr = bg_bgr.copy()
        merged_depth_mm = bg_depth_mm.copy()
        if np.any(fg_mask):
            merged_bgr[fg_mask] = rgb[..., ::-1][fg_mask]
        if self.foreground_mode == "isaac_mesh":
            return self._finalize_rgb(merged_bgr), merged_depth_mm


        robot_rgb, robot_alpha, robot_depth_m, robot_valid = self._render_robot_layer(cam, robot_attrs)
        robot_bgr = np.clip(robot_rgb * 255.0, 0, 255).astype(np.uint8)[..., ::-1]
        robot_depth_mm = np.clip(robot_depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        # Do not depth-test robot GS against the Isaac foreground.  The two
        # renderers use different depth conventions; the layer order above is
        # the authoritative visibility contract for online policy observations.
        robot_front = robot_valid & (robot_alpha > 0.02)
        if np.any(robot_front):
            alpha = robot_alpha[robot_front, None].astype(np.float32)
            merged_bgr[robot_front] = np.clip(
                robot_bgr[robot_front].astype(np.float32)
                + (1.0 - alpha) * merged_bgr[robot_front].astype(np.float32),
                0,
                255,
            ).astype(np.uint8)
            merged_depth_mm[robot_front] = robot_depth_mm[robot_front]
        return self._finalize_rgb(merged_bgr), merged_depth_mm


def init_gs_renderer(args: argparse.Namespace, env) -> Any | None:
    if not args.enable_gs_rollout:
        return None
    gs_ply = args.gs_scene_path.expanduser().resolve()
    if not gs_ply.is_file():
        if args.require_gs:
            raise FileNotFoundError(gs_ply)
        print(f"[WARN] GS scene path missing, fallback to Isaac RGB: {gs_ply}", flush=True)
        return None
    backend = detect_gs_backend_from_ply(gs_ply) if args.gs_backend == "auto" else args.gs_backend
    camera = get_head_camera(env)
    try:
        if backend == "2dgs":
            print(f"[INFO] Using online 2DGS background renderer: {gs_ply}", flush=True)
            return Online2DGSRenderer(
                gs_ply,
                camera,
                args.two_d_gs_real2sim_root,
                args.gs_semantic_tolerance,
                foreground_mode=getattr(args, "gs_foreground_mode", "robot_gs"),
                robot_parts_dir=args.robot_gs_parts_dir,
                robot_reference_ply=args.robot_gs_reference_ply,
                robot_link_json=args.robot_link_json,
                robot_urdf=args.robot_urdf,
                robot_gs_root=args.gs_root,
                apply_training_color_match=bool(getattr(args, "gs_apply_training_color_match", False)),
            )
        from gs_3d_adapter import ThreeDGSRenderer

        intrinsic = np.zeros((4, 4), dtype=np.float32)
        intrinsic[:3, :3] = camera.data.intrinsic_matrices.squeeze(0).detach().cpu().numpy().astype(np.float32)
        height, width = camera.image_shape
        intrinsic[3, 0] = float(height)
        intrinsic[3, 1] = float(width)
        foreground_rgba = np.concatenate(
            [FOREGROUND_SEMANTIC_RGB.astype(np.uint8), np.full((3, 1), 255, dtype=np.uint8)],
            axis=1,
        )
        print(f"[INFO] Using online 3DGS background renderer: {gs_ply}", flush=True)
        return ThreeDGSRenderer(
            gs_ply=str(gs_ply),
            gs_root=str(args.gs_root.expanduser().resolve()),
            init_pos=(0.0, 0.0, 0.0),
            init_wxyz=(1.0, 0.0, 0.0, 0.0),
            foreground_semantic_rgba=foreground_rgba,
            foreground_depth_scale_semantic_rgba=foreground_rgba[1:],
            foreground_depth_scale=1.0,
            intrinsics_matrix=intrinsic,
            texture_robot=False,
            use_fake_plane_depth=False,
        )
    except Exception as exc:
        if args.require_gs:
            print(f"[ERROR] Failed to initialize required {backend} renderer: {exc}", flush=True)
            raise
        print(f"[WARN] Failed to initialize {backend} renderer, fallback to Isaac RGB: {exc}", flush=True)
        return None


def gs_render_from_env(
    gs_render,
    camera,
    robot=None,
    sim_joint_names: list[str] | None = None,
    *,
    env_id: int = 0,
    env_origin_w: np.ndarray | None = None,
    camera_pose_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    outputs = camera.data.output
    required = ("rgb", "distance_to_image_plane", "semantic_segmentation")
    missing = [name for name in required if name not in outputs]
    if missing:
        raise RuntimeError(f"GS rollout requires camera outputs {required}; missing={missing}")
    camera_cached_pos_w = camera.data.pos_w[int(env_id)].detach().cpu().numpy()
    camera_cached_quat_w = camera.data.quat_w_world[int(env_id)].detach().cpu().numpy()
    if camera_pose_override is None:
        camera_pos_w = camera_cached_pos_w
        camera_quat_w = camera_cached_quat_w
    else:
        camera_pos_w, camera_quat_w = camera_pose_override
    rgb = outputs["rgb"][int(env_id)].detach().cpu().numpy()
    depth = outputs["distance_to_image_plane"][int(env_id)].detach().cpu().numpy()
    semantic = outputs["semantic_segmentation"][int(env_id)].detach().cpu().numpy()
    origin = np.zeros(3, dtype=np.float32) if env_origin_w is None else np.asarray(env_origin_w, dtype=np.float32).reshape(3)
    camera_pos_w = camera_pos_w - origin
    debug_dir = os.environ.get("HUMANOIDVLA_GS_DEBUG_DIR")
    if debug_dir:
        Path(debug_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)
        np.savez(
            Path(debug_dir).expanduser().resolve() / "camera_pose_sources.npz",
            camera_cached_position_w=np.asarray(camera_cached_pos_w, dtype=np.float64),
            camera_cached_quat_wxyz=np.asarray(camera_cached_quat_w, dtype=np.float64),
            camera_2dgs_position_w=np.asarray(camera_pos_w, dtype=np.float64),
            camera_2dgs_quat_wxyz=np.asarray(camera_quat_w, dtype=np.float64),
            env_origin_w=np.asarray(origin, dtype=np.float64),
        )
    robot_attrs = None
    if (
        robot is not None
        and sim_joint_names is not None
        and getattr(gs_render, "foreground_mode", "robot_gs") == "robot_gs"
        and hasattr(gs_render, "robot_attrs")
    ):
        root_state_w = robot.data.root_state_w[int(env_id)].detach().cpu().numpy()
        root_state_w = root_state_w.copy()
        root_state_w[:3] -= origin
        joint_pos = robot.data.joint_pos[int(env_id)].detach().cpu().numpy()
        robot_attrs = gs_render.robot_attrs(root_state_w=root_state_w, joint_names=sim_joint_names, joint_pos=joint_pos)
    return gs_render.render_with_gs_bg(
        pos_=camera_pos_w,
        quat_w_world=camera_quat_w,
        isaac_render_rgb=rgb,
        isaac_render_depth=depth,
        isaac_render_semantic_images=semantic,
        robot_attrs=robot_attrs,
    )


def capture_head_camera_rgb(
    env,
    gs_render,
    robot,
    sim_joint_names: list[str],
    *,
    env_id: int = 0,
) -> tuple[np.ndarray, str]:
    wait_for_head_camera_output(env)
    camera = get_head_camera(env)
    if gs_render is not None:
        frame, _ = gs_render_from_env(
            gs_render,
            camera,
            robot=robot,
            sim_joint_names=sim_joint_names,
            env_id=env_id,
        )
        source = "gs_zbuffer"
    else:
        frame = camera.data.output["rgb"][int(env_id)].detach().cpu().numpy()
        source = "isaac"
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    return frame, source


def repeat_to_horizon(batch_item: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(batch_item[:, None, ...], int(horizon), axis=1)


def gather_joint_values(joint_tensor, joint_names: list[str], target_names: list[str]) -> np.ndarray:
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    values = [float(joint_tensor[name_to_idx[name]].item()) if name in name_to_idx else 0.0 for name in target_names]
    return np.asarray(values, dtype=np.float32)


def resolve_joint_pos_limits(robot, sim_joint_names: list[str]) -> dict[str, tuple[float, float]]:
    limits_tensor = getattr(robot.data, "soft_joint_pos_limits", None)
    if limits_tensor is None:
        limits_tensor = getattr(robot.data, "joint_pos_limits", None)
    limits: dict[str, tuple[float, float]] = {}
    if limits_tensor is None:
        return limits
    limits_np = limits_tensor.detach().cpu().numpy()
    if limits_np.ndim == 3:
        limits_np = limits_np[0]
    for idx, name in enumerate(sim_joint_names):
        if idx >= limits_np.shape[0]:
            break
        lo, hi = float(limits_np[idx, 0]), float(limits_np[idx, 1])
        if np.isfinite(lo) and np.isfinite(hi) and abs(hi - lo) > 1e-8:
            limits[name] = (lo, hi)
    return limits


def joint_pose_to_hand_open01(value: float, joint_name: str, joint_limits: dict[str, tuple[float, float]]) -> float:
    lo_hi = joint_limits.get(joint_name)
    if lo_hi is None:
        return float(np.clip(value, 0.0, 1.0))
    lo, hi = lo_hi
    return float(np.clip((hi - float(value)) / (hi - lo), 0.0, 1.0))


def gather_hand_open01(joint_tensor, joint_names: list[str], target_names: list[str], joint_limits: dict[str, tuple[float, float]]) -> np.ndarray:
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    values = []
    for name in target_names:
        idx = name_to_idx.get(name)
        qpos = float(joint_tensor[idx].item()) if idx is not None else 0.0
        values.append(joint_pose_to_hand_open01(qpos, name, joint_limits))
    return np.asarray(values, dtype=np.float32)


def build_gr00t_state_values(
    robot,
    sim_joint_names: list[str],
    joint_limits: dict[str, tuple[float, float]],
    *,
    env_id: int = 0,
) -> dict[str, np.ndarray]:
    import torch
    from isaaclab.utils.math import quat_apply_inverse

    joint_pos = robot.data.joint_pos[int(env_id)].detach().cpu()
    body_quat = robot.data.body_quat_w[int(env_id), 0].detach().cpu()
    projected_gravity = quat_apply_inverse(
        body_quat.view(1, 4),
        torch.tensor([[0.0, 0.0, -1.0]], dtype=body_quat.dtype, device=body_quat.device),
    )[0]
    left_hand = gather_hand_open01(joint_pos, sim_joint_names, LEFT_HAND_6D_TO_ACTIVE, joint_limits)
    right_hand = gather_hand_open01(joint_pos, sim_joint_names, RIGHT_HAND_6D_TO_ACTIVE, joint_limits)
    return {
        "left_q": gather_joint_values(joint_pos, sim_joint_names, LEFT_ARM_JOINT_NAMES),
        "right_q": gather_joint_values(joint_pos, sim_joint_names, RIGHT_ARM_JOINT_NAMES),
        "left_hand_gripper": np.asarray([left_hand[2]], dtype=np.float32),
        "right_hand_gripper": np.asarray([right_hand[2]], dtype=np.float32),
        "waist": gather_joint_values(joint_pos, sim_joint_names, WAIST_JOINT_NAMES),
        "left_leg": gather_joint_values(joint_pos, sim_joint_names, LEFT_LEG_JOINT_NAMES),
        "right_leg": gather_joint_values(joint_pos, sim_joint_names, RIGHT_LEG_JOINT_NAMES),
        "projected_gravity": projected_gravity.cpu().numpy().astype(np.float32),
    }


def build_gr00t_observation(
    env,
    robot,
    sim_joint_names: list[str],
    joint_limits: dict[str, tuple[float, float]],
    modality_configs: dict[str, Any],
    task_description: str,
    gs_render,
    *,
    env_id: int = 0,
) -> tuple[dict[str, Any], str, np.ndarray]:
    frame, frame_source = capture_head_camera_rgb(
        env,
        gs_render,
        robot=robot,
        sim_joint_names=sim_joint_names,
        env_id=env_id,
    )
    state_values = build_gr00t_state_values(
        robot,
        sim_joint_names,
        joint_limits,
        env_id=env_id,
    )
    observation: dict[str, Any] = {"video": {}, "state": {}, "language": {}}
    video_horizon = len(modality_delta_indices(modality_configs["video"]))
    for key in modality_keys(modality_configs["video"]):
        if key != "ego_view":
            raise KeyError(f"Unsupported GR00T video key for this script: {key}")
        observation["video"][key] = repeat_to_horizon(frame[None, ...].astype(np.uint8), video_horizon)
    state_horizon = len(modality_delta_indices(modality_configs["state"]))
    for key in modality_keys(modality_configs["state"]):
        if key not in state_values:
            raise KeyError(f"Unsupported GR00T state key for this script: {key}")
        observation["state"][key] = repeat_to_horizon(state_values[key][None, ...].astype(np.float32), state_horizon)
    language_horizon = len(modality_delta_indices(modality_configs.get("language", {"delta_indices": [0]})))
    for key in modality_keys(modality_configs.get("language", {"modality_keys": ["annotation.human.action.task_description"]})):
        observation["language"][key] = [[task_description for _ in range(language_horizon)]]
    return observation, frame_source, frame


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        out[name] = value
        if isinstance(value, dict):
            out.update(flatten_dict(value, name))
    return out


def ensure_numpy_action(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, None, :]
    elif array.ndim == 2:
        array = array[None, :, :]
    elif array.ndim != 3:
        raise ValueError(f"Expected action array with ndim in {{1,2,3}}, got shape {array.shape}")
    return array


def _get_action_value(flat: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in flat:
            return flat[name]
    raise KeyError(f"GR00T action missing candidates={names}; available={sorted(flat.keys())}")


def _scalar_action(value: Any, horizon: int, name: str) -> np.ndarray:
    array = ensure_numpy_action(value)
    if array.shape[-1] != 1:
        array = array[..., :1]
    if array.shape[1] == 1 and horizon > 1:
        array = np.repeat(array, horizon, axis=1)
    if array.shape[1] != horizon:
        raise ValueError(f"{name} horizon {array.shape[1]} does not match token horizon {horizon}")
    return np.clip(array.astype(np.float32), 0.0, 1.0)


def parse_sonic_actions(action_dict: dict[str, Any]) -> SonicActionChunk:
    flat = flatten_dict(action_dict)
    token = ensure_numpy_action(_get_action_value(flat, "action.sonic_token", "sonic_token"))
    if token.shape[-1] != 64:
        raise ValueError(f"Expected sonic_token dim 64, got {token.shape}")
    horizon = int(token.shape[1])
    left = _scalar_action(
        _get_action_value(flat, "action.left_hand_gripper", "left_hand_gripper"),
        horizon,
        "left_hand_gripper",
    )
    right = _scalar_action(
        _get_action_value(flat, "action.right_hand_gripper", "right_hand_gripper"),
        horizon,
        "right_hand_gripper",
    )
    return SonicActionChunk(token=token.astype(np.float32), left_hand_gripper=left, right_hand_gripper=right)


def hand_open01_to_joint_target(open01: float, joint_name: str, joint_limits: dict[str, tuple[float, float]]) -> float:
    lo_hi = joint_limits.get(joint_name)
    if lo_hi is None:
        return float(np.clip(open01, 0.0, 1.0))
    lo, hi = lo_hi
    return float(lo + (1.0 - float(np.clip(open01, 0.0, 1.0))) * (hi - lo))


def scalar_hand_targets(open01: float, side: str, available_joint_names: set[str], joint_limits: dict[str, tuple[float, float]]) -> dict[str, float]:
    if side == "left":
        with_head_names = [
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
        ]
        active = LEFT_HAND_6D_TO_ACTIVE
        prefix = "left_"
    else:
        with_head_names = [
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
        ]
        active = RIGHT_HAND_6D_TO_ACTIVE
        prefix = "right_"
    if any(name in available_joint_names for name in with_head_names):
        names = with_head_names
    else:
        names = active + [
            f"{prefix}thumb_3_joint",
            f"{prefix}thumb_4_joint",
            f"{prefix}index_2_joint",
            f"{prefix}middle_2_joint",
            f"{prefix}ring_2_joint",
            f"{prefix}little_2_joint",
        ]
    return {
        name: hand_open01_to_joint_target(open01, name, joint_limits)
        for name in names
        if name in available_joint_names
    }


def hard_lock_joint_state(
    robot,
    *,
    sim_joint_names: list[str],
    joint_names: list[str],
    value: float,
) -> None:
    import torch

    joint_ids = [sim_joint_names.index(name) for name in joint_names if name in sim_joint_names]
    if not joint_ids:
        return
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    joint_pos[:, joint_ids] = float(value)
    joint_vel[:, joint_ids] = 0.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    with contextlib.suppress(Exception):
        target = torch.full(
            (joint_pos.shape[0], len(joint_ids)),
            float(value),
            device=robot.device,
            dtype=torch.float32,
        )
        robot.set_joint_position_target(target, joint_ids=joint_ids)


def apply_head_action_targets(
    env_action,
    action_name_to_idx: dict[str, int],
    *,
    enabled: bool,
    value: float,
) -> None:
    if not enabled:
        return
    for name in HEAD_JOINT_NAMES:
        idx = action_name_to_idx.get(name)
        if idx is not None:
            env_action[idx] = float(value)


class VideoRecorder:
    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = float(fps)
        self.writer = None
        self.frames = 0

    def write(self, frame_rgb: np.ndarray) -> None:
        import cv2

        frame = np.asarray(frame_rgb, dtype=np.uint8)[..., :3]
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h), isColor=True)
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self.path}")
        self.writer.write(frame[..., ::-1])
        self.frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class ObsPreview:
    def __init__(self, window_name: str, wait_ms: int):
        self.window_name = str(window_name)
        self.wait_ms = max(1, int(wait_ms))
        self.enabled = True

    def write(self, frame_rgb: np.ndarray, frame_source: str) -> None:
        if not self.enabled:
            return
        try:
            import cv2

            frame = np.asarray(frame_rgb, dtype=np.uint8)[..., :3]
            cv2.imshow(f"{self.window_name}[{frame_source}]", frame[..., ::-1])
            cv2.waitKey(self.wait_ms)
        except Exception as exc:
            print(f"[WARN] cv2.imshow failed; disabling obs preview: {exc}", flush=True)
            self.enabled = False

    def close(self) -> None:
        if not self.enabled:
            return
        with contextlib.suppress(Exception):
            import cv2

            cv2.destroyWindow(self.window_name)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_task_success(records: dict[str, list[Any]], cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Evaluate the task's object-placement gates after a complete online rollout."""

    if not isinstance(cfg, dict) or not cfg:
        return {"configured": False, "ok": True, "reasons": []}
    required = ("object_root_pose", "object_root_velocity", "placement_root_pose")
    missing = [key for key in required if not records.get(key)]
    if missing:
        return {"configured": True, "ok": False, "reasons": ["missing_rigid_state"], "missing": missing}

    object_pose = np.asarray(records["object_root_pose"], dtype=np.float32)
    object_velocity = np.asarray(records["object_root_velocity"], dtype=np.float32)
    placement_pose = np.asarray(records["placement_root_pose"], dtype=np.float32)
    final_delta = object_pose[-1, :3] - placement_pose[-1, :3]
    final_speed = float(np.linalg.norm(object_velocity[-1, :3]))
    max_lift = float(np.max(object_pose[:, 2]) - object_pose[0, 2])
    limits = {
        "min_x": float(cfg.get("min_x", -0.15)),
        "max_x": float(cfg.get("max_x", 0.15)),
        "min_y": float(cfg.get("min_y", -0.10)),
        "max_y": float(cfg.get("max_y", 0.10)),
        "max_height": float(cfg.get("max_height", 0.13)),
        "max_object_speed": float(cfg.get("max_object_speed", 0.10)),
        "min_object_lift": float(cfg.get("min_object_lift", 0.03)),
    }
    checks = {
        "x": limits["min_x"] <= float(final_delta[0]) <= limits["max_x"],
        "y": limits["min_y"] <= float(final_delta[1]) <= limits["max_y"],
        "height": float(final_delta[2]) < limits["max_height"],
        "object_speed": final_speed < limits["max_object_speed"],
        "object_lifted": max_lift >= limits["min_object_lift"],
    }
    return {
        "configured": True,
        "ok": bool(all(checks.values())),
        "reasons": [name for name, passed in checks.items() if not passed],
        "checks": checks,
        "limits": limits,
        "final_object_position": object_pose[-1, :3].astype(float).tolist(),
        "final_placement_position": placement_pose[-1, :3].astype(float).tolist(),
        "final_object_minus_placement": final_delta.astype(float).tolist(),
        "final_object_speed": final_speed,
        "max_object_lift": max_lift,
    }




