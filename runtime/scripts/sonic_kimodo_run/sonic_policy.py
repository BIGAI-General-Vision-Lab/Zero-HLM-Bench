from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


HISTORY_FRAMES = 10
MOTION_FRAMES = 10
MOTION_STEP = 5
CONTROL_HZ = 50.0
KIMODO_G1_FPS = 30.0
SONIC_NUM_JOINTS = 29
SONIC_TOKEN_DIM = 64

SONIC_OBSERVATION_DIMENSIONS = {
    "token_state": 64,
    "his_base_angular_velocity_10frame_step1": 30,
    "his_body_joint_positions_10frame_step1": 290,
    "his_body_joint_velocities_10frame_step1": 290,
    "his_last_actions_10frame_step1": 290,
    "his_gravity_dir_10frame_step1": 30,
    "encoder_mode_4": 4,
    "motion_joint_positions_10frame_step5": 290,
    "motion_joint_velocities_10frame_step5": 290,
    "motion_root_z_position_10frame_step5": 10,
    "motion_root_z_position": 1,
    "motion_anchor_orientation": 6,
    "motion_anchor_orientation_10frame_step5": 60,
    "motion_joint_positions_lowerbody_10frame_step5": 120,
    "motion_joint_velocities_lowerbody_10frame_step5": 120,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "smpl_joints_10frame_step1": 720,
    "smpl_anchor_orientation_10frame_step1": 60,
    "motion_joint_positions_wrists_10frame_step1": 60,
}

G1_MUJOCO_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# For each IsaacLab-order SONIC joint index, this is the corresponding MuJoCo qpos index.
MUJOCO_INDEX_FOR_ISAACLAB = np.asarray(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)
ISAACLAB_INDEX_FOR_MUJOCO = np.asarray(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
G1_ISAACLAB_BODY_JOINT_NAMES = tuple(G1_MUJOCO_BODY_JOINT_NAMES[i] for i in MUJOCO_INDEX_FOR_ISAACLAB.tolist())

G1_DEFAULT_ANGLES_MUJOCO = np.asarray(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)
G1_DEFAULT_ANGLES_ISAACLAB = G1_DEFAULT_ANGLES_MUJOCO[MUJOCO_INDEX_FOR_ISAACLAB]

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10.0 * 2.0 * math.pi
EFFORT_LIMIT_5020 = 25.0
EFFORT_LIMIT_7520_14 = 88.0
EFFORT_LIMIT_7520_22 = 139.0
EFFORT_LIMIT_4010 = 5.0


def _stiffness(armature: float) -> float:
    return armature * NATURAL_FREQ * NATURAL_FREQ


G1_ACTION_SCALE_MUJOCO = np.asarray(
    [
        0.25 * EFFORT_LIMIT_7520_22 / _stiffness(ARMATURE_7520_22),
        0.25 * EFFORT_LIMIT_7520_22 / _stiffness(ARMATURE_7520_22),
        0.25 * EFFORT_LIMIT_7520_14 / _stiffness(ARMATURE_7520_14),
        0.25 * EFFORT_LIMIT_7520_22 / _stiffness(ARMATURE_7520_22),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_7520_22 / _stiffness(ARMATURE_7520_22),
        0.25 * EFFORT_LIMIT_7520_22 / _stiffness(ARMATURE_7520_22),
        0.25 * EFFORT_LIMIT_7520_14 / _stiffness(ARMATURE_7520_14),
        0.25 * EFFORT_LIMIT_7520_22 / _stiffness(ARMATURE_7520_22),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_7520_14 / _stiffness(ARMATURE_7520_14),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_4010 / _stiffness(ARMATURE_4010),
        0.25 * EFFORT_LIMIT_4010 / _stiffness(ARMATURE_4010),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_5020 / _stiffness(ARMATURE_5020),
        0.25 * EFFORT_LIMIT_4010 / _stiffness(ARMATURE_4010),
        0.25 * EFFORT_LIMIT_4010 / _stiffness(ARMATURE_4010),
    ],
    dtype=np.float32,
)
G1_ACTION_SCALE_ISAACLAB = G1_ACTION_SCALE_MUJOCO[MUJOCO_INDEX_FOR_ISAACLAB]


@dataclass(frozen=True)
class ObservationEntry:
    name: str
    dimension: int
    offset: int


@dataclass(frozen=True)
class ObservationLayout:
    policy_entries: tuple[ObservationEntry, ...]
    encoder_entries: tuple[ObservationEntry, ...]
    encoder_mode_ids: dict[str, int]
    encoder_mode_required: dict[str, tuple[str, ...]]

    @property
    def policy_dimension(self) -> int:
        return self.policy_entries[-1].offset + self.policy_entries[-1].dimension if self.policy_entries else 0

    @property
    def encoder_dimension(self) -> int:
        return self.encoder_entries[-1].offset + self.encoder_entries[-1].dimension if self.encoder_entries else 0

    @property
    def encoder_by_name(self) -> dict[str, ObservationEntry]:
        return {entry.name: entry for entry in self.encoder_entries}


@dataclass
class RobotState:
    root_quat_wxyz: np.ndarray
    joint_pos_rel_isaaclab: np.ndarray
    joint_vel_isaaclab: np.ndarray
    base_ang_vel: np.ndarray
    gravity_dir: np.ndarray


@dataclass
class SonicAction:
    raw_action: np.ndarray
    q_target_isaaclab: np.ndarray
    encoder_obs: np.ndarray
    token_state: np.ndarray
    decoder_obs: np.ndarray


@dataclass
class ReferenceMotion:
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_pos_isaaclab: np.ndarray
    joint_vel_isaaclab: np.ndarray
    control_hz: float
    source_path: str | None = None
    future_mode: str = "trajectory"

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos_isaaclab.shape[0])

    def future_indices(self, frame_idx: int, num_frames: int = MOTION_FRAMES, step_size: int = MOTION_STEP) -> np.ndarray:
        if self.num_frames <= 0:
            raise ValueError("ReferenceMotion has no frames")
        if self.future_mode == "hold":
            safe_idx = min(max(int(frame_idx), 0), self.num_frames - 1)
            return np.full((num_frames,), safe_idx, dtype=np.int64)
        if self.future_mode != "trajectory":
            raise ValueError(f"Unsupported ReferenceMotion future_mode: {self.future_mode}")
        idx = int(frame_idx) + np.arange(num_frames, dtype=np.int64) * int(step_size)
        return np.minimum(idx, self.num_frames - 1)

    def future_joint_pos(self, frame_idx: int, num_frames: int = MOTION_FRAMES, step_size: int = MOTION_STEP) -> np.ndarray:
        return self.joint_pos_isaaclab[self.future_indices(frame_idx, num_frames, step_size)]

    def future_joint_vel(self, frame_idx: int, num_frames: int = MOTION_FRAMES, step_size: int = MOTION_STEP) -> np.ndarray:
        if self.future_mode == "hold":
            return np.zeros((num_frames, self.joint_vel_isaaclab.shape[1]), dtype=np.float32)
        return self.joint_vel_isaaclab[self.future_indices(frame_idx, num_frames, step_size)]

    def future_root_quat(self, frame_idx: int, num_frames: int = MOTION_FRAMES, step_size: int = MOTION_STEP) -> np.ndarray:
        return self.root_quat_wxyz[self.future_indices(frame_idx, num_frames, step_size)]

    @classmethod
    def from_qpos_csv(
        cls,
        path: str | Path,
        source_fps: float = KIMODO_G1_FPS,
        control_hz: float = CONTROL_HZ,
        future_mode: str = "trajectory",
    ) -> "ReferenceMotion":
        qpos = np.loadtxt(path, delimiter=",", dtype=np.float64)
        if qpos.ndim == 1:
            qpos = qpos[None, :]
        return cls.from_qpos(
            qpos,
            source_fps=source_fps,
            control_hz=control_hz,
            source_path=str(path),
            future_mode=future_mode,
        )

    @classmethod
    def from_sonic_reference_dir(
        cls,
        path: str | Path,
        control_hz: float = CONTROL_HZ,
    ) -> "ReferenceMotion":
        path = Path(path)
        joint_pos = _load_csv_matrix(path / "joint_pos.csv")
        joint_vel = _load_csv_matrix(path / "joint_vel.csv")
        body_pos = _load_csv_matrix(path / "body_pos.csv")
        body_quat = _load_csv_matrix(path / "body_quat.csv")
        if joint_pos.ndim != 2 or joint_pos.shape[1] != SONIC_NUM_JOINTS:
            raise ValueError(f"Expected joint_pos.csv shape [T,29], got {joint_pos.shape}")
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(f"Expected joint_vel.csv shape {joint_pos.shape}, got {joint_vel.shape}")
        if body_pos.ndim != 2 or body_pos.shape[1] % 3 != 0:
            raise ValueError(f"Expected body_pos.csv shape [T,3*N], got {body_pos.shape}")
        if body_quat.ndim != 2 or body_quat.shape[1] % 4 != 0:
            raise ValueError(f"Expected body_quat.csv shape [T,4*N], got {body_quat.shape}")
        if body_pos.shape[0] != joint_pos.shape[0] or body_quat.shape[0] != joint_pos.shape[0]:
            raise ValueError("SONIC reference CSV frame counts do not match")

        root_pos = body_pos.reshape(body_pos.shape[0], -1, 3)[:, 0]
        root_quat = body_quat.reshape(body_quat.shape[0], -1, 4)[:, 0]
        return cls(
            root_pos=root_pos.astype(np.float32),
            root_quat_wxyz=normalize_quat(root_quat).astype(np.float32),
            joint_pos_isaaclab=joint_pos.astype(np.float32),
            joint_vel_isaaclab=joint_vel.astype(np.float32),
            control_hz=float(control_hz),
            source_path=str(path),
        )

    @classmethod
    def from_qpos(
        cls,
        qpos: np.ndarray,
        source_fps: float = KIMODO_G1_FPS,
        control_hz: float = CONTROL_HZ,
        source_path: str | None = None,
        future_mode: str = "trajectory",
    ) -> "ReferenceMotion":
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != 36:
            raise ValueError(f"Expected Kimodo/G1 MuJoCo qpos shape [T,36], got {qpos.shape}")
        if qpos.shape[0] == 0:
            raise ValueError("Reference qpos is empty")
        if source_fps <= 0.0 or control_hz <= 0.0:
            raise ValueError("source_fps and control_hz must be positive")
        if future_mode not in {"trajectory", "hold"}:
            raise ValueError(f"Unsupported ReferenceMotion future_mode: {future_mode}")

        root_pos_src = qpos[:, :3]
        root_quat_src = normalize_quat(qpos[:, 3:7])
        joints_mujoco_src = qpos[:, 7:]

        if qpos.shape[0] == 1:
            root_pos = root_pos_src.astype(np.float32)
            root_quat = root_quat_src.astype(np.float32)
            joints_mujoco = joints_mujoco_src.astype(np.float32)
        else:
            src_t = np.arange(qpos.shape[0], dtype=np.float64) / float(source_fps)
            dst_count = int(round(src_t[-1] * float(control_hz))) + 1
            dst_t = np.arange(dst_count, dtype=np.float64) / float(control_hz)
            dst_t = np.minimum(dst_t, src_t[-1])

            root_pos = np.stack([np.interp(dst_t, src_t, root_pos_src[:, i]) for i in range(3)], axis=-1)
            joints_mujoco = np.stack([np.interp(dst_t, src_t, joints_mujoco_src[:, i]) for i in range(29)], axis=-1)
            root_quat = slerp_times(src_t, root_quat_src, dst_t)

        joint_pos_isaaclab = joints_mujoco[:, MUJOCO_INDEX_FOR_ISAACLAB].astype(np.float32)
        joint_vel_isaaclab = np.zeros_like(joint_pos_isaaclab, dtype=np.float32)
        if joint_pos_isaaclab.shape[0] > 1:
            joint_vel_isaaclab[:-1] = np.diff(joint_pos_isaaclab, axis=0) * float(control_hz)
            joint_vel_isaaclab[-1] = joint_vel_isaaclab[-2]

        return cls(
            root_pos=root_pos.astype(np.float32),
            root_quat_wxyz=normalize_quat(root_quat).astype(np.float32),
            joint_pos_isaaclab=joint_pos_isaaclab,
            joint_vel_isaaclab=joint_vel_isaaclab,
            control_hz=float(control_hz),
            source_path=source_path,
            future_mode=future_mode,
        )


def clamp_reference_root_pitch_min(
    reference: ReferenceMotion,
    *,
    min_pitch_rad: float,
) -> dict[str, float | int]:
    """Clamp only the reference root pitch while preserving roll, yaw, and joints.

    This intentionally changes the encoder reference only.  It does not mutate
    the source qpos/HDF5, transfer tilt into a waist joint, or alter the
    simulator's measured root orientation.
    """

    if not np.isfinite(min_pitch_rad):
        raise ValueError("min_pitch_rad must be finite")
    quat = normalize_quat(reference.root_quat_wxyz).astype(np.float64)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    limited_pitch = np.maximum(pitch, float(min_pitch_rad))

    half_roll = 0.5 * roll
    half_pitch = 0.5 * limited_pitch
    half_yaw = 0.5 * yaw
    cr, sr = np.cos(half_roll), np.sin(half_roll)
    cp, sp = np.cos(half_pitch), np.sin(half_pitch)
    cy, sy = np.cos(half_yaw), np.sin(half_yaw)
    reference.root_quat_wxyz = normalize_quat(
        np.stack(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            axis=-1,
        )
    ).astype(np.float32)
    return {
        "frames_clamped": int(np.count_nonzero(pitch < float(min_pitch_rad))),
        "min_pitch_before_deg": float(np.degrees(np.min(pitch))),
        "min_pitch_after_deg": float(np.degrees(np.min(limited_pitch))),
    }


def _load_csv_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)


def parse_observation_layout(config_path: str | Path) -> ObservationLayout:
    config = yaml.safe_load(Path(config_path).read_text())
    policy_entries: list[ObservationEntry] = []
    encoder_entries: list[ObservationEntry] = []

    offset = 0
    for observation in config["observations"]:
        if not observation.get("enabled", False):
            continue
        name = observation["name"]
        if name not in SONIC_OBSERVATION_DIMENSIONS:
            raise KeyError(f"Unsupported SONIC policy observation: {name}")
        dim = SONIC_OBSERVATION_DIMENSIONS[name]
        policy_entries.append(ObservationEntry(name=name, dimension=dim, offset=offset))
        offset += dim

    offset = 0
    for observation in config["encoder"]["encoder_observations"]:
        if not observation.get("enabled", False):
            continue
        name = observation["name"]
        if name not in SONIC_OBSERVATION_DIMENSIONS:
            raise KeyError(f"Unsupported SONIC encoder observation: {name}")
        dim = SONIC_OBSERVATION_DIMENSIONS[name]
        encoder_entries.append(ObservationEntry(name=name, dimension=dim, offset=offset))
        offset += dim

    modes = config["encoder"]["encoder_modes"]
    encoder_mode_ids = {mode["name"]: int(mode["mode_id"]) for mode in modes}
    encoder_mode_required = {mode["name"]: tuple(mode["required_observations"]) for mode in modes}
    return ObservationLayout(
        policy_entries=tuple(policy_entries),
        encoder_entries=tuple(encoder_entries),
        encoder_mode_ids=encoder_mode_ids,
        encoder_mode_required=encoder_mode_required,
    )


class SonicPolicy:
    def __init__(
        self,
        reference_motion: ReferenceMotion,
        models_dir: str | Path,
        encoder_mode_name: str = "g1",
        observation_config_path: str | Path | None = None,
        use_cuda: bool = True,
        load_encoder: bool = True,
        encoder_motion_step: int = MOTION_STEP,
    ):
        if int(encoder_motion_step) <= 0:
            raise ValueError(f"encoder_motion_step must be positive, got {encoder_motion_step}")
        self.reference_motion = reference_motion
        self.encoder_motion_step = int(encoder_motion_step)
        self.models_dir = Path(models_dir)
        self.encoder_model_path = self.models_dir / "model_encoder.onnx"
        self.decoder_model_path = self.models_dir / "model_decoder.onnx"
        config_path = Path(observation_config_path) if observation_config_path else self.models_dir / "observation_config.yaml"
        required_paths = [self.decoder_model_path, config_path]
        if load_encoder:
            required_paths.insert(0, self.encoder_model_path)
        for path in required_paths:
            if not path.is_file():
                raise FileNotFoundError(path)

        self.layout = parse_observation_layout(config_path)
        if self.layout.policy_dimension != 994:
            raise ValueError(f"Unexpected SONIC decoder observation dimension: {self.layout.policy_dimension}")
        if self.layout.encoder_dimension != 1762:
            raise ValueError(f"Unexpected SONIC encoder observation dimension: {self.layout.encoder_dimension}")
        if encoder_mode_name != "g1":
            raise ValueError("This minimal adapter only supports SONIC g1 encoder mode")
        self.encoder_mode_name = encoder_mode_name
        self.encoder_mode_id = self.layout.encoder_mode_ids[encoder_mode_name]
        self.encoder_required = self.layout.encoder_mode_required[encoder_mode_name]

        self._ort = _import_onnxruntime()
        self.encoder_session = self._make_session(self.encoder_model_path, use_cuda=use_cuda) if load_encoder else None
        self.decoder_session = self._make_session(self.decoder_model_path, use_cuda=use_cuda)
        self.frame_idx = 0

        self.base_ang_vel_hist = np.zeros((HISTORY_FRAMES, 3), dtype=np.float32)
        self.joint_pos_hist = np.zeros((HISTORY_FRAMES, SONIC_NUM_JOINTS), dtype=np.float32)
        self.joint_vel_hist = np.zeros((HISTORY_FRAMES, SONIC_NUM_JOINTS), dtype=np.float32)
        self.last_actions_hist = np.zeros((HISTORY_FRAMES, SONIC_NUM_JOINTS), dtype=np.float32)
        self.gravity_dir_hist = np.zeros((HISTORY_FRAMES, 3), dtype=np.float32)
        self.last_raw_action = np.zeros((SONIC_NUM_JOINTS,), dtype=np.float32)
        self.init_base_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.init_ref_quat = self.reference_motion.root_quat_wxyz[0].copy()

    def _make_session(self, model_path: Path, use_cuda: bool):
        options = self._ort.SessionOptions()
        options.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = self._ort.get_available_providers()
        selected = []
        if use_cuda and "CUDAExecutionProvider" in providers:
            selected.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in providers:
            selected.append("CPUExecutionProvider")
        if not selected:
            selected = providers
        return self._ort.InferenceSession(str(model_path), sess_options=options, providers=selected)

    def reset(self, state: RobotState, frame_idx: int = 0) -> None:
        state = normalize_robot_state(state)
        self.frame_idx = int(frame_idx)
        self.base_ang_vel_hist[:] = state.base_ang_vel[None, :]
        self.joint_pos_hist[:] = state.joint_pos_rel_isaaclab[None, :]
        self.joint_vel_hist[:] = state.joint_vel_isaaclab[None, :]
        self.gravity_dir_hist[:] = state.gravity_dir[None, :]
        self.last_actions_hist[:] = 0.0
        self.last_raw_action[:] = 0.0
        self.init_base_quat = state.root_quat_wxyz.astype(np.float32)
        safe_idx = min(max(self.frame_idx, 0), self.reference_motion.num_frames - 1)
        self.init_ref_quat = self.reference_motion.root_quat_wxyz[safe_idx].astype(np.float32)

    def act(
        self,
        state: RobotState,
        frame_idx: int | None = None,
        advance: bool = True,
        reference_future_mode: str | None = None,
    ) -> SonicAction:
        if self.encoder_session is None:
            raise RuntimeError("SonicPolicy was created with load_encoder=False; use act_with_token().")
        state = normalize_robot_state(state)
        if frame_idx is not None:
            self.frame_idx = int(frame_idx)
        self._push_history(state)

        encoder_obs = self.build_encoder_observation(
            self.frame_idx,
            reference_future_mode=reference_future_mode,
        )
        token_state = self._run_onnx(self.encoder_session, encoder_obs, "obs_dict")
        decoder_obs = self.build_decoder_observation(token_state)
        raw_action = self._run_onnx(self.decoder_session, decoder_obs, "obs_dict")
        raw_action = raw_action.astype(np.float32).reshape(SONIC_NUM_JOINTS)
        q_target = raw_action_to_q_target_isaaclab(raw_action)

        self.last_raw_action = raw_action.copy()
        self.last_actions_hist[0] = raw_action
        if advance:
            self.frame_idx += 1
        return SonicAction(
            raw_action=raw_action,
            q_target_isaaclab=q_target,
            encoder_obs=encoder_obs,
            token_state=token_state.astype(np.float32),
            decoder_obs=decoder_obs,
        )

    def act_with_token(
        self,
        state: RobotState,
        token_state: np.ndarray,
        frame_idx: int | None = None,
        advance: bool = True,
    ) -> SonicAction:
        """Run only SONIC decoder using a caller-supplied 64-D token."""
        state = normalize_robot_state(state)
        if frame_idx is not None:
            self.frame_idx = int(frame_idx)
        self._push_history(state)

        token_state = np.asarray(token_state, dtype=np.float32).reshape(SONIC_TOKEN_DIM)
        decoder_obs = self.build_decoder_observation(token_state)
        raw_action = self._run_onnx(self.decoder_session, decoder_obs, "obs_dict")
        raw_action = raw_action.astype(np.float32).reshape(SONIC_NUM_JOINTS)
        q_target = raw_action_to_q_target_isaaclab(raw_action)

        self.last_raw_action = raw_action.copy()
        self.last_actions_hist[0] = raw_action
        if advance:
            self.frame_idx += 1
        return SonicAction(
            raw_action=raw_action,
            q_target_isaaclab=q_target,
            encoder_obs=np.zeros((self.layout.encoder_dimension,), dtype=np.float32),
            token_state=token_state,
            decoder_obs=decoder_obs,
        )

    def build_encoder_observation(
        self,
        frame_idx: int,
        reference_future_mode: str | None = None,
    ) -> np.ndarray:
        previous_future_mode = self.reference_motion.future_mode
        if reference_future_mode is not None:
            if reference_future_mode not in {"trajectory", "hold"}:
                raise ValueError(f"Unsupported encoder reference future mode: {reference_future_mode}")
            self.reference_motion.future_mode = reference_future_mode
        try:
            obs = np.zeros((self.layout.encoder_dimension,), dtype=np.float32)
            by_name = self.layout.encoder_by_name
            for name in self.encoder_required:
                entry = by_name[name]
                values = self._encoder_term(name, frame_idx)
                if values.shape != (entry.dimension,):
                    raise ValueError(
                        f"Unexpected shape for encoder term {name}: {values.shape}, expected {(entry.dimension,)}"
                    )
                obs[entry.offset : entry.offset + entry.dimension] = values
            return obs
        finally:
            self.reference_motion.future_mode = previous_future_mode

    def build_decoder_observation(self, token_state: np.ndarray) -> np.ndarray:
        token_state = np.asarray(token_state, dtype=np.float32).reshape(SONIC_TOKEN_DIM)
        obs = np.zeros((self.layout.policy_dimension,), dtype=np.float32)
        for entry in self.layout.policy_entries:
            values = self._decoder_term(entry.name, token_state)
            if values.shape != (entry.dimension,):
                raise ValueError(f"Unexpected shape for decoder term {entry.name}: {values.shape}, expected {(entry.dimension,)}")
            obs[entry.offset : entry.offset + entry.dimension] = values
        return obs

    def _push_history(self, state: RobotState) -> None:
        self._latest_root_quat = state.root_quat_wxyz.astype(np.float32)
        self.base_ang_vel_hist[1:] = self.base_ang_vel_hist[:-1]
        self.joint_pos_hist[1:] = self.joint_pos_hist[:-1]
        self.joint_vel_hist[1:] = self.joint_vel_hist[:-1]
        self.last_actions_hist[1:] = self.last_actions_hist[:-1]
        self.gravity_dir_hist[1:] = self.gravity_dir_hist[:-1]
        self.base_ang_vel_hist[0] = state.base_ang_vel
        self.joint_pos_hist[0] = state.joint_pos_rel_isaaclab
        self.joint_vel_hist[0] = state.joint_vel_isaaclab
        self.last_actions_hist[0] = self.last_raw_action
        self.gravity_dir_hist[0] = state.gravity_dir

    def _encoder_term(self, name: str, frame_idx: int) -> np.ndarray:
        if name == "encoder_mode_4":
            out = np.zeros((4,), dtype=np.float32)
            out[0] = float(self.encoder_mode_id)
            return out
        if name == "motion_joint_positions_10frame_step5":
            return self.reference_motion.future_joint_pos(
                frame_idx, MOTION_FRAMES, self.encoder_motion_step
            ).reshape(-1).astype(np.float32)
        if name == "motion_joint_velocities_10frame_step5":
            return self.reference_motion.future_joint_vel(
                frame_idx, MOTION_FRAMES, self.encoder_motion_step
            ).reshape(-1).astype(np.float32)
        if name == "motion_anchor_orientation_10frame_step5":
            return self._motion_anchor_orientation(frame_idx, MOTION_FRAMES, self.encoder_motion_step)
        raise KeyError(f"Unsupported g1 encoder observation term: {name}")

    def _decoder_term(self, name: str, token_state: np.ndarray) -> np.ndarray:
        if name == "token_state":
            return token_state
        if name == "his_base_angular_velocity_10frame_step1":
            return self.base_ang_vel_hist[::-1].reshape(-1).astype(np.float32)
        if name == "his_body_joint_positions_10frame_step1":
            return self.joint_pos_hist[::-1].reshape(-1).astype(np.float32)
        if name == "his_body_joint_velocities_10frame_step1":
            return self.joint_vel_hist[::-1].reshape(-1).astype(np.float32)
        if name == "his_last_actions_10frame_step1":
            return self.last_actions_hist[::-1].reshape(-1).astype(np.float32)
        if name == "his_gravity_dir_10frame_step1":
            return self.gravity_dir_hist[::-1].reshape(-1).astype(np.float32)
        raise KeyError(f"Unsupported decoder observation term: {name}")

    def _motion_anchor_orientation(self, frame_idx: int, num_frames: int, step_size: int) -> np.ndarray:
        future_quat = self.reference_motion.future_root_quat(frame_idx, num_frames, step_size)
        apply_delta_heading = quat_mul(yaw_quat(self.init_base_quat), quat_inv(yaw_quat(self.init_ref_quat)))
        aligned_ref = quat_mul(np.broadcast_to(apply_delta_heading, future_quat.shape), future_quat)
        current_base = np.broadcast_to(self.base_quat_for_anchor(), future_quat.shape)
        base_to_ref = quat_mul(quat_inv(current_base), aligned_ref)
        mats = quat_to_matrix(base_to_ref)
        return mats[..., :2].reshape(-1).astype(np.float32)

    def base_quat_for_anchor(self) -> np.ndarray:
        # The newest history sample corresponds to the state used for this action.
        return getattr(self, "_latest_root_quat", self.init_base_quat)

    def _run_onnx(self, session, observation: np.ndarray, input_name: str) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        output = session.run(None, {input_name: observation})[0]
        return np.asarray(output, dtype=np.float32).reshape(-1)


def raw_action_to_q_target_isaaclab(raw_action_isaaclab: np.ndarray) -> np.ndarray:
    raw_action_isaaclab = np.asarray(raw_action_isaaclab, dtype=np.float32).reshape(SONIC_NUM_JOINTS)
    return (G1_DEFAULT_ANGLES_ISAACLAB + raw_action_isaaclab * G1_ACTION_SCALE_ISAACLAB).astype(np.float32)


def absolute_qpos_isaaclab_to_relative(qpos_isaaclab: np.ndarray) -> np.ndarray:
    qpos_isaaclab = np.asarray(qpos_isaaclab, dtype=np.float32).reshape(SONIC_NUM_JOINTS)
    return (qpos_isaaclab - G1_DEFAULT_ANGLES_ISAACLAB).astype(np.float32)


def normalize_robot_state(state: RobotState) -> RobotState:
    root_quat = normalize_quat(np.asarray(state.root_quat_wxyz, dtype=np.float32).reshape(4))
    joint_pos = np.asarray(state.joint_pos_rel_isaaclab, dtype=np.float32).reshape(SONIC_NUM_JOINTS)
    joint_vel = np.asarray(state.joint_vel_isaaclab, dtype=np.float32).reshape(SONIC_NUM_JOINTS)
    base_ang = np.asarray(state.base_ang_vel, dtype=np.float32).reshape(3)
    gravity = np.asarray(state.gravity_dir, dtype=np.float32).reshape(3)
    n = np.linalg.norm(gravity)
    if n > 1e-8:
        gravity = gravity / n
    state = RobotState(root_quat, joint_pos, joint_vel, base_ang, gravity)
    return state


def projected_gravity_from_root_quat(root_quat_wxyz: np.ndarray) -> np.ndarray:
    root_quat = normalize_quat(np.asarray(root_quat_wxyz, dtype=np.float32).reshape(4))
    return quat_rotate_inverse(root_quat, np.asarray([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    out = quat / norm
    if out.ndim == 1 and out[0] < 0.0:
        out = -out
    elif out.ndim > 1:
        out = np.where(out[..., :1] < 0.0, -out, out)
    return out


def quat_inv(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat)
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    out = np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )
    return normalize_quat(out)


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat)
    vec = np.asarray(vec, dtype=np.float64)
    mat = quat_to_matrix(quat)
    return np.einsum("...ji,...j->...i", mat, vec)


def yaw_quat(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    zeros = np.zeros_like(half)
    out = np.stack([np.cos(half), zeros, zeros, np.sin(half)], axis=-1)
    return normalize_quat(out)


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    two = 2.0
    return np.stack(
        [
            np.stack([1.0 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)], axis=-1),
            np.stack([two * (x * y + z * w), 1.0 - two * (x * x + z * z), two * (y * z - x * w)], axis=-1),
            np.stack([two * (x * z - y * w), two * (y * z + x * w), 1.0 - two * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def slerp_times(src_t: np.ndarray, src_quat: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    src_quat = normalize_quat(src_quat)
    out = np.empty((dst_t.shape[0], 4), dtype=np.float64)
    for i, t in enumerate(dst_t):
        if t <= src_t[0]:
            out[i] = src_quat[0]
            continue
        if t >= src_t[-1]:
            out[i] = src_quat[-1]
            continue
        right = int(np.searchsorted(src_t, t, side="right"))
        left = right - 1
        denom = src_t[right] - src_t[left]
        alpha = 0.0 if denom <= 0.0 else float((t - src_t[left]) / denom)
        out[i] = slerp(src_quat[left], src_quat[right], alpha)
    return normalize_quat(out)


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        return normalize_quat(q0 + alpha * (q1 - q0))
    theta_0 = math.acos(dot)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return normalize_quat((s0 * q0) + (s1 * q1))


def _import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for SonicPolicy inference; use env_isaaclab python.") from exc
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(directory="")
        except Exception:
            pass
    return ort


def make_synthetic_walk_qpos(frames: int = 90, source_fps: float = KIMODO_G1_FPS) -> np.ndarray:
    t = np.arange(frames, dtype=np.float64) / float(source_fps)
    qpos = np.zeros((frames, 36), dtype=np.float64)
    qpos[:, 0] = 0.35 * t
    qpos[:, 1] = 0.0
    qpos[:, 2] = 0.78 + 0.015 * np.sin(2.0 * math.pi * 1.0 * t)
    qpos[:, 3] = 1.0
    qpos[:, 7:] = G1_DEFAULT_ANGLES_MUJOCO[None, :]
    phase = 2.0 * math.pi * 1.2 * t
    qpos[:, 7 + 0] += 0.12 * np.sin(phase)
    qpos[:, 7 + 6] += 0.12 * np.sin(phase + math.pi)
    qpos[:, 7 + 3] += 0.18 * np.maximum(0.0, np.sin(phase + math.pi))
    qpos[:, 7 + 9] += 0.18 * np.maximum(0.0, np.sin(phase))
    qpos[:, 7 + 15] += 0.16 * np.sin(phase + math.pi)
    qpos[:, 7 + 22] += 0.16 * np.sin(phase)
    qpos[:, 7 + 18] += 0.12 * np.sin(phase + math.pi)
    qpos[:, 7 + 25] += 0.12 * np.sin(phase)
    return qpos


def _self_test(args: argparse.Namespace) -> dict:
    if args.csv:
        reference = ReferenceMotion.from_qpos_csv(args.csv, source_fps=args.source_fps, control_hz=args.control_hz)
    else:
        reference = ReferenceMotion.from_qpos(make_synthetic_walk_qpos(), source_fps=args.source_fps, control_hz=args.control_hz)

    state = RobotState(
        root_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        joint_pos_rel_isaaclab=np.zeros((SONIC_NUM_JOINTS,), dtype=np.float32),
        joint_vel_isaaclab=np.zeros((SONIC_NUM_JOINTS,), dtype=np.float32),
        base_ang_vel=np.zeros((3,), dtype=np.float32),
        gravity_dir=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
    )
    summary = {
        "reference_frames": reference.num_frames,
        "reference_source": reference.source_path,
        "isaaclab_joint_names": list(G1_ISAACLAB_BODY_JOINT_NAMES),
        "default_q_target_min": float(G1_DEFAULT_ANGLES_ISAACLAB.min()),
        "default_q_target_max": float(G1_DEFAULT_ANGLES_ISAACLAB.max()),
    }

    if args.run_onnx:
        policy = SonicPolicy(reference, args.models_dir, use_cuda=not args.cpu)
        policy.reset(state)
        action = policy.act(state, frame_idx=0, advance=False)
        summary.update(
            {
                "encoder_obs_shape": list(action.encoder_obs.shape),
                "decoder_obs_shape": list(action.decoder_obs.shape),
                "token_state_shape": list(action.token_state.shape),
                "raw_action_shape": list(action.raw_action.shape),
                "raw_action_finite": bool(np.isfinite(action.raw_action).all()),
                "q_target_min": float(action.q_target_isaaclab.min()),
                "q_target_max": float(action.q_target_isaaclab.max()),
            }
        )
    else:
        layout = parse_observation_layout(Path(args.models_dir) / "observation_config.yaml")
        summary.update({"encoder_dim": layout.encoder_dimension, "decoder_dim": layout.policy_dimension})
    return summary


