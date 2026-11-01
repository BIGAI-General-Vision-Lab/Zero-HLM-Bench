#!/usr/bin/env python3
"""Shared simulator runtime for paired GR00T + SONIC evaluators."""

from __future__ import annotations

import argparse
import copy
import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROBOMIMIC_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
SONIC_RUN_DIR = REPO_ROOT / "scripts" / "sonic_kimodo_run"
SIM_EVALUATION_DIR = REPO_ROOT.parent / "configs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "envs") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "envs"))
if str(SONIC_RUN_DIR) not in sys.path:
    sys.path.insert(0, str(SONIC_RUN_DIR))
if str(ROBOMIMIC_DIR) not in sys.path:
    sys.path.insert(0, str(ROBOMIMIC_DIR))

from isaaclab.app import AppLauncher

from wbc_play_groot_sonic import (  # noqa: E402
    DEFAULT_B1_GS_PLY,
    DEFAULT_GS_ROOT,
    DEFAULT_HEADCAM_CALIB,
    DEFAULT_MODELS_DIR,
    DEFAULT_TASK,
    DEFAULT_TWO_D_GS_REAL2SIM_ROOT,
    HEAD_JOINT_NAMES,
    LEFT_ARM_JOINT_NAMES,
    LEFT_HAND_6D_TO_ACTIVE,
    LEFT_LEG_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_HAND_6D_TO_ACTIVE,
    RIGHT_LEG_JOINT_NAMES,
    WAIST_JOINT_NAMES,
    HeadCameraCalibration,
    RobotInferenceClient,
    SonicActionChunk,
    apply_head_action_targets,
    apply_head_camera_mode,
    build_gr00t_observation,
    detect_gs_backend_from_ply,
    hard_lock_joint_state,
    init_gs_renderer,
    load_processor_modality_configs,
    modality_delta_indices,
    modality_keys,
    parse_sonic_actions,
    quat_normalize_np,
    repeat_to_horizon,
    resolve_joint_pos_limits,
    scalar_hand_targets,
    summarize_modality_config,
    wait_for_head_camera_output,
    write_json,
    yaw_to_quat_wxyz_np,
)

from task1_evaluation_contract import (  # noqa: E402
    TASK1_PROMPT,
    binarize_right_gripper as contract_binarize_right_gripper,
)
from evaluation_runner import EvaluationTaskAdapter, run_evaluation
from evaluation_video import EpisodeVideoRecorder, capture_third_person, diagnostic_video_required


# The release evaluator consumes the immutable 100-row manifest.
MANIFEST_ROW_COUNT = 100


DEFAULT_OUTPUT_ROOT = REPO_ROOT.parent / "outputs" / "task1"
DEFAULT_IN_DIS_803_INIT_POSE = REPO_ROOT.parent / "assets/task1/assets/evaluation/task1_initial_body_pose_episode_000000.json"
DEFAULT_IN_DIS_803_LEROBOT = (
    Path(os.environ["TASK1_INDIS_INIT_DATASET"]).expanduser()
    if os.environ.get("TASK1_INDIS_INIT_DATASET")
    else None
)
DEFAULT_IN_DIS_803_SCENE_CONTEXT_HDF5 = (
    Path(os.environ["TASK1_GS_PROBE_SOURCE_HDF5"]).expanduser()
    if os.environ.get("TASK1_GS_PROBE_SOURCE_HDF5")
    else None
)
TERMINAL_REASONS = {"success", "fall", "horizon", "error"}


LEGACY_TASK1_HEADCAM_OFFSET_POS = (0.0416, -0.0115, -0.02044)
LEGACY_TASK1_HEADCAM_OFFSET_ROT_WXYZ = (-0.29884, 0.64086, -0.64086, 0.29884)

# Tight manual diagnostic distribution in the canonical Task1 scene frame.
# Keep yaw fixed until position-only evaluation is stable; yaw robustness can
# then be measured independently instead of being confounded with translation.
MANUAL_INIT_X_RANGE = (0.52625, 0.72625)
MANUAL_INIT_Y_RANGE = (-0.30, -0.10)
MANUAL_INIT_YAW_CENTER_DEG = 180.0
MANUAL_INIT_YAW_JITTER_DEG = (0.0, 0.0)
EVAL_OBJECT_POSE_RANGE = {
    "x": (-0.10, 0.10),
    "y": (-0.10, 0.10),
    "yaw": (-math.radians(10.0), math.radians(10.0)),
}
EVAL_PLACEMENT_POSE_RANGE = {
    "x": (-0.10, 0.10),
    "y": (-0.10, 0.10),
}

# LeRobot compact-state order deliberately differs from SONIC's IsaacLab order.
# These names are the 29 articulated body entries after removing the two hand
# scalars and the three projected-gravity entries from observation.state.
IN_DIS_COMPACT_BODY_JOINT_NAMES = tuple(
    list(LEFT_ARM_JOINT_NAMES)
    + list(RIGHT_ARM_JOINT_NAMES)
    + list(WAIST_JOINT_NAMES)
    + list(LEFT_LEG_JOINT_NAMES)
    + list(RIGHT_LEG_JOINT_NAMES)
)
IN_DIS_EXPECTED_STATE_NAMES = (
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow", "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "left_hand_gripper", "right_hand_gripper",
    "kWaistYaw", "kWaistRoll", "kWaistPitch",
    "kLeftHipPitch", "kLeftHipRoll", "kLeftHipYaw", "kLeftKnee", "kLeftAnklePitch", "kLeftAnkleRoll",
    "kRightHipPitch", "kRightHipRoll", "kRightHipYaw", "kRightKnee", "kRightAnklePitch", "kRightAnkleRoll",
    "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
)


def load_initial_body_pose_json(
    path: Path,
    sonic_joint_names: Iterable[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a portable, schema-checked Task1 body-joint initialization."""

    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported initial-pose schema in {path}: {payload.get('schema_version')}")
    names = tuple(str(name) for name in payload["joint_names"])
    values = np.asarray(payload["joint_positions"], dtype=np.float32)
    if len(names) != len(values) or not np.isfinite(values).all():
        raise ValueError(f"Invalid initial-pose joint arrays in {path}: names={len(names)} values={values.shape}")
    requested_names = tuple(sonic_joint_names)
    if set(names) != set(requested_names):
        raise ValueError(
            f"Initial-pose joints do not match SONIC joints in {path}: "
            f"pose={names}, sonic={requested_names}"
        )
    by_name = dict(zip(names, values, strict=True))
    target = np.asarray([by_name[name] for name in requested_names], dtype=np.float32)
    provenance = {
        "source": "portable_in_dis_episode_first_frame",
        "path": str(path),
        "source_dataset": payload.get("source_dataset"),
        "source_episode": payload.get("source_episode"),
        "source_frame": payload.get("source_frame"),
        "sonic_joint_order": list(requested_names),
        "body_joint_target": [float(value) for value in target],
    }
    return target, provenance


def load_indis_initial_body_pose(
    dataset_root: Path,
    episode_count: int,
    sonic_joint_names: Iterable[str],
    *,
    mode: str,
    episode_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a natural standing body target from in-dis LeRobot episode starts.

    Root pose remains scenario-specific.  Only the 29 body joints are copied,
    using a per-joint median across episode frame zero to suppress reset noise.
    """

    dataset_root = Path(dataset_root).expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"in-dis init-pose metadata not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    names = tuple(info["features"]["observation.state"]["names"][0])
    if names != IN_DIS_EXPECTED_STATE_NAMES:
        raise ValueError(
            "Unexpected in-dis observation.state schema; refusing to apply a joint template. "
            f"Expected {IN_DIS_EXPECTED_STATE_NAMES}, got {names}"
        )
    parquet_paths = sorted((dataset_root / "data").rglob("*.parquet"))[: max(1, int(episode_count))]
    if not parquet_paths:
        raise FileNotFoundError(f"No LeRobot Parquet episodes found below {dataset_root / 'data'}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - evaluator env is required to provide pyarrow.
        raise RuntimeError("PyArrow is required to load the in-dis initial joint template") from exc
    starts = []
    for path in parquet_paths:
        state = np.asarray(pq.read_table(path, columns=["observation.state"]).column(0).to_pylist()[0], dtype=np.float32)
        if state.shape != (34,) or not np.isfinite(state).all():
            raise ValueError(f"Invalid frame-zero observation.state in {path}: shape={state.shape}")
        starts.append(state)
    if mode == "episode_first_frame":
        if episode_index < 0 or episode_index >= len(starts):
            raise IndexError(
                f"in-dis init episode index {episode_index} is outside loaded range [0, {len(starts) - 1}]"
            )
        compact = np.asarray(starts[episode_index], dtype=np.float32)
        source = "in_dis_episode_first_frame"
    elif mode == "median":
        compact = np.median(np.asarray(starts, dtype=np.float32), axis=0)
        source = "in_dis_episode_frame_zero_median"
    else:
        raise ValueError(f"Unsupported in-dis initial pose mode: {mode}")
    body_compact = np.concatenate((compact[:14], compact[16:31]), axis=0).astype(np.float32)
    by_name = dict(zip(IN_DIS_COMPACT_BODY_JOINT_NAMES, body_compact, strict=True))
    requested_names = tuple(sonic_joint_names)
    if set(requested_names) != set(IN_DIS_COMPACT_BODY_JOINT_NAMES):
        raise ValueError(
            "SONIC body-joint order is incompatible with the in-dis compact-state template: "
            f"sonic={requested_names}, compact={IN_DIS_COMPACT_BODY_JOINT_NAMES}"
        )
    target = np.asarray([by_name[name] for name in requested_names], dtype=np.float32)
    provenance = {
        "source": source,
        "dataset_root": str(dataset_root),
        "episodes": len(parquet_paths),
        "episode_index": int(episode_index) if mode == "episode_first_frame" else None,
        "compact_joint_order": list(IN_DIS_COMPACT_BODY_JOINT_NAMES),
        "sonic_joint_order": list(requested_names),
        "body_joint_target": [float(value) for value in target],
    }
    return target, provenance


def is_fallback_trigger(exc: BaseException) -> bool:
    message = str(exc).lower()
    resource_like = any(token in message for token in ("cuda", "out of memory", "memory", "vram", "resource"))
    renderer_capability_gap = "env_id-aware renderer" in message
    return bool(resource_like or renderer_capability_gap)


@dataclass(frozen=True)
class Pose:
    pos: list[float]
    quat_wxyz: list[float]

    @classmethod
    def from_arrays(cls, pos: Iterable[float], quat_wxyz: Iterable[float]) -> "Pose":
        return cls(
            pos=[float(v) for v in np.asarray(list(pos), dtype=np.float64).reshape(3)],
            quat_wxyz=[float(v) for v in quat_normalize_np(np.asarray(list(quat_wxyz), dtype=np.float64).reshape(4))],
        )


@dataclass(frozen=True)
class SceneSE2:
    """Per-episode Task1 world transform used by the production replay path."""

    translation_xy: list[float]
    delta_yaw: float

    @classmethod
    def identity(cls) -> "SceneSE2":
        return cls(translation_xy=[0.0, 0.0], delta_yaw=0.0)

    @classmethod
    def from_arrays(cls, translation_xy: Iterable[float], delta_yaw: float) -> "SceneSE2":
        return cls(
            translation_xy=[float(v) for v in np.asarray(list(translation_xy), dtype=np.float64).reshape(2)],
            delta_yaw=float(delta_yaw),
        )


@dataclass(frozen=True)
class ScenarioRow:
    schema_version: int
    row_index: int
    scenario_id: str
    seed: int
    robot: Pose
    object: Pose
    placement: Pose
    scene_se2: SceneSE2 = field(default_factory=SceneSE2.identity)
    row_hash: str = ""

    def canonical_payload(self, *, include_hash: bool = False) -> dict[str, Any]:
        payload = {
            "schema_version": int(self.schema_version),
            "row_index": int(self.row_index),
            "scenario_id": str(self.scenario_id),
            "seed": int(self.seed),
            "robot": asdict(self.robot),
            "object": asdict(self.object),
            "placement": asdict(self.placement),
            "scene_se2": asdict(self.scene_se2),
        }
        if include_hash:
            payload["row_hash"] = str(self.row_hash)
        return payload

    def with_hash(self) -> "ScenarioRow":
        payload = canonical_json(self.canonical_payload(include_hash=False))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return ScenarioRow(
            schema_version=self.schema_version,
            row_index=self.row_index,
            scenario_id=self.scenario_id,
            seed=self.seed,
        robot=self.robot,
        object=self.object,
        placement=self.placement,
        scene_se2=self.scene_se2,
            row_hash=digest,
        )


@dataclass
class EnvSlot:
    env_id: int
    scenario: ScenarioRow | None = None
    policy: Any | None = None
    last_action: Any | None = None
    left_hand_state_open01: float = 1.0
    right_hand_state_open01: float = 1.0
    chunk: SonicActionChunk | None = None
    chunk_pos: int = 0
    steps: int = 0
    query_count: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    min_root_height: float = math.inf
    max_object_lift: float = 0.0
    initial_object_z: float | None = None
    final_delta: list[float] | None = None
    final_object_speed: float | None = None
    success_state_history: dict[str, dict[str, list[np.ndarray]]] = field(default_factory=dict)
    success_result: dict[str, Any] = field(default_factory=dict)
    fall: bool = False
    terminal_reason: str | None = None
    error: str | None = None
    video: Any | None = None
    first_frame: np.ndarray | None = None
    final_frame: np.ndarray | None = None
    started_s: float = 0.0

    @property
    def active(self) -> bool:
        return self.scenario is not None and self.terminal_reason is None


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=REPO_ROOT) != 0
        return {"commit": commit, "dirty": bool(dirty)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def manifest_hash(rows: list[ScenarioRow]) -> str:
    text = "".join(canonical_json(row.canonical_payload(include_hash=True)) + "\n" for row in rows)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scenario_from_json(payload: dict[str, Any]) -> ScenarioRow:
    stored_hash = str(payload.get("row_hash", ""))
    raw_payload = {key: value for key, value in payload.items() if key != "row_hash"}
    expected_hash = hashlib.sha256(canonical_json(raw_payload).encode("utf-8")).hexdigest()
    if stored_hash and stored_hash != expected_hash:
        raise ValueError(
            f"Manifest row {payload.get('row_index')} hash mismatch: {stored_hash} != {expected_hash}"
        )
    row = ScenarioRow(
        schema_version=int(payload["schema_version"]),
        row_index=int(payload["row_index"]),
        scenario_id=str(payload["scenario_id"]),
        seed=int(payload["seed"]),
        robot=Pose.from_arrays(payload["robot"]["pos"], payload["robot"]["quat_wxyz"]),
        object=Pose.from_arrays(payload["object"]["pos"], payload["object"]["quat_wxyz"]),
        placement=Pose.from_arrays(payload["placement"]["pos"], payload["placement"]["quat_wxyz"]),
        scene_se2=SceneSE2.from_arrays(
            payload.get("scene_se2", {}).get("translation_xy", [0.0, 0.0]),
            payload.get("scene_se2", {}).get("delta_yaw", 0.0),
        ),
        row_hash=stored_hash,
    )
    return row if row.row_hash else row.with_hash()


def write_manifest(path: Path, rows: list[ScenarioRow]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical_json(row.with_hash().canonical_payload(include_hash=True)) + "\n")
    os.replace(tmp, path)


def read_manifest(path: Path, expected_count: int | None = MANIFEST_ROW_COUNT) -> list[ScenarioRow]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = scenario_from_json(json.loads(line))
            if row.row_index != len(rows):
                raise ValueError(f"Manifest row order mismatch at line {line_no}: row_index={row.row_index}")
            if row.scenario_id != f"task1_{row.row_index:05d}":
                raise ValueError(f"Unexpected scenario_id at line {line_no}: {row.scenario_id}")
            rows.append(row)
    if expected_count is not None and len(rows) != int(expected_count):
        raise ValueError(f"Manifest must contain exactly {expected_count} rows, found {len(rows)}")
    return rows


def select_manifest_rows(
    rows: list[ScenarioRow],
    *,
    start: int = 0,
    stop: int = 0,
    limit: int = 0,
) -> list[ScenarioRow]:
    """Select a stable manifest shard after validating the complete manifest."""

    if start < 0:
        raise ValueError(f"manifest_row_start must be non-negative, got {start}")
    if stop < 0:
        raise ValueError(f"manifest_row_stop must be non-negative, got {stop}")
    if stop and stop <= start:
        raise ValueError(f"manifest_row_stop must exceed manifest_row_start, got [{start}, {stop})")
    selected = rows[start : stop or None]
    if limit > 0:
        selected = selected[:limit]
    if not selected:
        raise ValueError(
            f"Manifest row selection is empty: total={len(rows)} start={start} stop={stop} limit={limit}"
        )
    return selected


def pending_rows(rows: list[ScenarioRow], episode_csv: Path, resume: bool) -> list[ScenarioRow]:
    if not resume or not episode_csv.is_file():
        return rows
    completed: set[str] = set()
    with episode_csv.open("r", encoding="utf-8", newline="") as f:
        for item in csv.DictReader(f):
            reason = item.get("terminal_reason")
            if reason in TERMINAL_REASONS:
                completed.add(str(item["scenario_id"]))
    return [row for row in rows if row.scenario_id not in completed]


def parse_args(adapter: EvaluationTaskAdapter | None = None) -> argparse.Namespace:
    if adapter is None:
        adapter = make_default_adapter()
    parser = argparse.ArgumentParser(description=f"{adapter.name} paired multi-env GR00T + SONIC evaluator.")
    parser.add_argument("--task", type=str, default=adapter.task_id)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--model_name", type=str, choices=("real", "sim", "custom"), default="custom")
    parser.add_argument("--embodiment", type=str, default="unitree_g1_sonic")
    parser.add_argument("--manifest", type=Path, default=adapter.default_manifest, required=adapter.default_manifest is None)
    parser.add_argument("--generate_manifest", action="store_true")
    parser.add_argument("--manifest_count", type=int, default=adapter.manifest_row_count)
    parser.add_argument("--output_root", type=Path, default=adapter.output_root)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--fallback_env_counts", type=int, nargs="*", default=[1])
    parser.add_argument("--max_control_steps", type=int, default=2000)
    parser.add_argument("--chunk_horizon", type=int, default=adapter.default_chunk_horizon)
    parser.add_argument("--control_hz", type=float, default=50.0)
    parser.add_argument("--diagnostic_video_interval", type=int, default=10,
                        help="Record synchronized ego and third-person videos every N manifest episodes (one-based).")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--fall_height", type=float, default=0.35)
    parser.add_argument("--stop_on_fall", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gr00t_host", type=str, default="127.0.0.1")
    parser.add_argument("--gr00t_port", type=int, default=5555)
    parser.add_argument("--gr00t_timeout_ms", type=int, default=30000)
    parser.add_argument("--task_description", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--random_robot_init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample the manual Task1 diagnostic rectangle. Use --no-random_robot_init only for a fixed-pose diagnostic.",
    )
    parser.add_argument("--init_x_range", type=float, nargs=2, default=None)
    parser.add_argument("--init_y_range", type=float, nargs=2, default=None)
    parser.add_argument("--init_yaw_center_deg", type=float, default=None)
    parser.add_argument("--init_yaw_jitter_deg", type=float, nargs=2, default=None)
    parser.add_argument("--init_robot_z", type=float, default=None)
    parser.add_argument("--init_min_start_distance_m", type=float, default=0.05)
    parser.add_argument("--object_x_range", type=float, nargs=2, default=None)
    parser.add_argument("--object_y_range", type=float, nargs=2, default=None)
    parser.add_argument(
        "--object_yaw_range_deg",
        type=float,
        nargs=2,
        default=None,
    )
    parser.add_argument("--placement_x_range", type=float, nargs=2, default=None)
    parser.add_argument("--placement_y_range", type=float, nargs=2, default=None)
    parser.add_argument("--models_dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--sonic_cpu_onnx", action="store_true")
    parser.add_argument("--sonic_actuator_profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sonic_waist_stiffness_scale", type=float, default=None)
    parser.add_argument("--clip_body_targets_to_limits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset_to_sonic_default", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--init_joint_state_from_indis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize the 29 SONIC body joints from the in-dis LeRobot natural-standing first-frame median.",
    )
    parser.add_argument(
        "--init_joint_state_json",
        type=Path,
        default=adapter.default_init_joint_state_json,
        help="Portable 29-joint initialization. An explicit --init_joint_state_dataset takes precedence.",
    )
    parser.add_argument("--init_joint_state_dataset", type=Path, default=adapter.default_init_joint_state_dataset)
    parser.add_argument(
        "--source_initial_state_json",
        type=Path,
        default=None,
        help=(
            "Optional portable per-scenario frame-zero robot state. When present, its joint position, "
            "joint velocity, and root velocity replace the shared standing template after each reset."
        ),
    )
    parser.add_argument("--init_joint_state_episodes", type=int, default=128)
    parser.add_argument(
        "--init_joint_state_mode",
        choices=("episode_first_frame", "median"),
        default="episode_first_frame",
        help="Use episode_000000 frame 0 by default; median preserves the previous diagnostic behavior.",
    )
    parser.add_argument("--init_joint_state_episode_index", type=int, default=0)
    parser.add_argument("--enable_head_hand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--head_lock_rad", type=float, default=0.0)
    parser.add_argument(
        "--gripper_state_source",
        choices=("command_binary", "physical"),
        default="command_binary",
        help=(
            "Build gripper observation.state from the last binary command, matching the Task1 "
            "training parquet contract, or from continuous physical finger positions for diagnostics."
        ),
    )
    parser.add_argument("--headcam_calib", type=Path, default=DEFAULT_HEADCAM_CALIB)
    parser.add_argument(
        "--headcam_extrinsic_mode",
        choices=("root_calib_each_frame", "root_calib_attached_local", "task_cfg", "legacy_task_cfg"),
        default="task_cfg",
    )
    parser.add_argument("--gs_scene_path", type=Path, default=DEFAULT_B1_GS_PLY)
    parser.add_argument("--gs_backend", choices=("auto", "2dgs", "3dgs"), default="auto")
    parser.add_argument(
        "--gs_foreground_mode",
        choices=("isaac_mesh",),
        default="isaac_mesh",
        help="Render the robot and task objects as meshes over the Gaussian scene.",
    )
    parser.add_argument(
        "--gs_apply_training_color_match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the fixed RGB mapping used by the Task1 LeRobot training videos.",
    )
    parser.add_argument("--gs_root", type=Path, default=DEFAULT_GS_ROOT)
    parser.add_argument("--two_d_gs_real2sim_root", type=Path, default=DEFAULT_TWO_D_GS_REAL2SIM_ROOT)
    parser.add_argument("--gs_semantic_tolerance", type=int, default=12)
    parser.add_argument(
        "--camera_warmup_renders",
        type=int,
        default=10,
        help="Render without stepping physics after reset so camera pose and semantic buffers belong to the new episode.",
    )
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--max_manifest_rows", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--manifest_row_start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--manifest_row_stop", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--schema_only", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--untiled_head_camera_debug", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--vector_env_spacing",
        type=float,
        default=20.0,
        help="Distance in meters between cloned environments during vectorized evaluation.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.diagnostic_video_interval <= 0:
        raise ValueError("--diagnostic_video_interval must be positive")
    args.enable_cameras = True
    args.enable_gs_rollout = True
    args.require_gs = True
    args.save_obs_video = False
    if int(args.num_envs) < 1:
        raise ValueError("--num_envs must be positive")
    if int(args.max_control_steps) != 2000:
        print(f"[WARN] max_control_steps={args.max_control_steps}; contract default is 2000.", flush=True)
    if int(args.chunk_horizon) != int(adapter.default_chunk_horizon):
        print(
            f"[WARN] chunk_horizon={args.chunk_horizon}; {adapter.name} evaluation default is "
            f"{adapter.default_chunk_horizon}.",
            flush=True,
        )
    return args


def sample_robot_pose(args: argparse.Namespace, row_index: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(args.seed) + int(row_index))
    x0, x1 = sorted(float(v) for v in args.init_x_range)
    y0, y1 = sorted(float(v) for v in args.init_y_range)
    yaw0, yaw1 = sorted(float(v) for v in args.init_yaw_jitter_deg)
    for _ in range(10_000):
        x = float(rng.uniform(x0, x1))
        y = float(rng.uniform(y0, y1))
        yaw = math.radians(float(args.init_yaw_center_deg) + float(rng.uniform(yaw0, yaw1)))
        return np.asarray([x, y, float(args.init_robot_z)], dtype=np.float64), yaw_to_quat_wxyz_np(yaw)
    raise RuntimeError("Unable to sample a robot pose for manifest generation")


def _tensor_row(value: Any, env_id: int) -> np.ndarray:
    if hasattr(value, "detach"):
        return value[int(env_id)].detach().cpu().numpy()
    return np.asarray(value)[int(env_id)]


def asset_pose(env, asset_name: str, env_id: int) -> Pose:
    asset = env.scene[asset_name]
    state = _tensor_row(asset.data.root_state_w[:, :7], env_id)
    origin = _tensor_row(env.scene.env_origins, env_id)
    return Pose.from_arrays(state[:3] - origin, state[3:7])


def robot_pose(env, env_id: int) -> Pose:
    robot = env.scene["robot"]
    state = _tensor_row(robot.data.root_state_w[:, :7], env_id)
    origin = _tensor_row(env.scene.env_origins, env_id)
    return Pose.from_arrays(state[:3] - origin, state[3:7])


def reset_robot_root_env(
    env,
    robot,
    env_id: int,
    pose: Pose,
    root_velocity: np.ndarray | None = None,
) -> None:
    import torch

    env_ids = torch.as_tensor([int(env_id)], device=env.device, dtype=torch.long)
    pos_w = np.asarray(pose.pos, dtype=np.float32) + _tensor_row(env.scene.env_origins, env_id)
    root_pose = torch.as_tensor([pos_w.tolist() + pose.quat_wxyz], device=env.device, dtype=robot.data.root_state_w.dtype)
    root_velocity = np.zeros((6,), dtype=np.float32) if root_velocity is None else np.asarray(root_velocity, dtype=np.float32).reshape(6)
    root_vel = torch.as_tensor(root_velocity[None, :], device=env.device, dtype=robot.data.root_state_w.dtype)
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    joint_pos = robot.data.joint_pos[int(env_id) : int(env_id) + 1].clone()
    joint_vel = torch.zeros_like(robot.data.joint_vel[int(env_id) : int(env_id) + 1])
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    with contextlib.suppress(Exception):
        robot.set_joint_position_target(joint_pos, env_ids=env_ids)


def reset_asset_pose_env(
    env,
    asset_name: str,
    env_id: int,
    pose: Pose,
    root_velocity: np.ndarray | None = None,
) -> None:
    import torch

    asset = env.scene[asset_name]
    env_ids = torch.as_tensor([int(env_id)], device=env.device, dtype=torch.long)
    pos_w = np.asarray(pose.pos, dtype=np.float32) + _tensor_row(env.scene.env_origins, env_id)
    root_pose = torch.as_tensor([pos_w.tolist() + pose.quat_wxyz], device=env.device, dtype=asset.data.root_state_w.dtype)
    root_velocity = np.zeros((6,), dtype=np.float32) if root_velocity is None else np.asarray(root_velocity, dtype=np.float32).reshape(6)
    root_vel = torch.as_tensor(root_velocity[None, :], device=env.device, dtype=asset.data.root_state_w.dtype)
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)


def restore_reset_joint_state(
    runtime: dict[str, Any],
    env_id: int,
) -> None:
    """Restore the configured natural-standing body state."""

    import torch

    robot = runtime["robot"]
    env = runtime["env"]
    env_ids = torch.as_tensor([int(env_id)], device=env.device, dtype=torch.long)
    joint_pos = robot.data.joint_pos[int(env_id) : int(env_id) + 1].clone()
    joint_vel = robot.data.joint_vel[int(env_id) : int(env_id) + 1].clone()
    joint_pos[:, runtime["body_joint_sim_ids_t"]] = torch.as_tensor(
        runtime["init_body_joint_target"],
        device=runtime["device"],
        dtype=joint_pos.dtype,
    ).view(1, -1)
    joint_vel[:, runtime["body_joint_sim_ids_t"]] = 0.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    with contextlib.suppress(Exception):
        robot.set_joint_position_target(joint_pos, env_ids=env_ids)


def load_source_initial_states(path: Path, action_joint_names: list[str]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Load schema-checked per-scenario robot states extracted from the source HDF5."""

    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported source initial-state schema in {path}: {payload.get('schema_version')}")
    source_names = [str(name) for name in payload["action_joint_names"]]
    if set(source_names) != set(action_joint_names):
        raise ValueError(
            "Source initial-state joints do not match the environment action joints: "
            f"source={source_names}, environment={action_joint_names}"
        )
    source_index = {name: index for index, name in enumerate(source_names)}
    reorder = np.asarray([source_index[name] for name in action_joint_names], dtype=np.int64)
    states: dict[str, dict[str, np.ndarray]] = {}
    for scenario_id, raw in payload["scenarios"].items():
        joint_pos = np.asarray(raw["joint_position"], dtype=np.float32)
        joint_vel = np.asarray(raw["joint_velocity"], dtype=np.float32)
        root_vel = np.asarray(raw["root_velocity"], dtype=np.float32)
        if joint_pos.shape != (len(source_names),) or joint_vel.shape != (len(source_names),):
            raise ValueError(f"Invalid joint state for {scenario_id}: pos={joint_pos.shape}, vel={joint_vel.shape}")
        if root_vel.shape != (6,) or not all(np.isfinite(value).all() for value in (joint_pos, joint_vel, root_vel)):
            raise ValueError(f"Invalid root/joint values for {scenario_id}")
        states[str(scenario_id)] = {
            "joint_position": joint_pos[reorder],
            "joint_velocity": joint_vel[reorder],
            "root_velocity": root_vel,
        }
    provenance = {
        "path": str(path),
        "sha256": sha256_file(path),
        "scenario_count": len(states),
        "source_hdf5": payload.get("source_hdf5"),
    }
    return states, provenance


def restore_source_initial_state(runtime: dict[str, Any], env_id: int, row: ScenarioRow) -> None:
    """Restore the row-correlated dynamic state used at training frame zero."""

    import torch

    state = runtime.get("source_initial_states", {}).get(row.scenario_id)
    if state is None:
        raise KeyError(f"Source initial-state sidecar has no entry for {row.scenario_id}")
    env = runtime["env"]
    robot = runtime["robot"]
    env_ids = torch.as_tensor([int(env_id)], device=env.device, dtype=torch.long)
    joint_pos = robot.data.joint_pos[int(env_id) : int(env_id) + 1].clone()
    joint_vel = robot.data.joint_vel[int(env_id) : int(env_id) + 1].clone()
    joint_pos[:, runtime["action_joint_sim_ids_t"]] = torch.as_tensor(
        state["joint_position"], device=runtime["device"], dtype=joint_pos.dtype
    ).view(1, -1)
    joint_vel[:, runtime["action_joint_sim_ids_t"]] = torch.as_tensor(
        state["joint_velocity"], device=runtime["device"], dtype=joint_vel.dtype
    ).view(1, -1)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.write_root_velocity_to_sim(
        torch.as_tensor(state["root_velocity"], device=runtime["device"], dtype=robot.data.root_state_w.dtype).view(1, 6),
        env_ids=env_ids,
    )
    with contextlib.suppress(Exception):
        robot.set_joint_position_target(joint_pos, env_ids=env_ids)


def reset_env_slot_to_manifest(
    env,
    env_id: int,
    row: ScenarioRow,
) -> None:
    import torch

    env_ids = torch.as_tensor([int(env_id)], device=env.device, dtype=torch.long)
    if hasattr(env, "_reset_idx"):
        env._reset_idx(env_ids=env_ids)
    elif hasattr(env, "reset"):
        with contextlib.suppress(TypeError):
            env.reset(env_ids=env_ids)
    robot = env.scene["robot"]
    reset_robot_root_env(
        env,
        robot,
        env_id,
        row.robot,
    )
    reset_asset_pose_env(
        env,
        "object",
        env_id,
        row.object,
    )
    reset_asset_pose_env(
        env,
        "placement",
        env_id,
        row.placement,
    )
    if hasattr(env, "sim") and hasattr(env.sim, "forward"):
        env.sim.forward()
    env.scene.update(dt=env.physics_dt)




def generate_manifest_with_env(args: argparse.Namespace, env) -> list[ScenarioRow]:
    import torch

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    rows: list[ScenarioRow] = []
    robot = env.scene["robot"]
    count = int(args.manifest_count)
    for start in range(0, count, int(args.num_envs)):
        wave = list(range(start, min(count, start + int(args.num_envs))))
        env_ids = torch.as_tensor(list(range(len(wave))), device=env.device, dtype=torch.long)
        if hasattr(env, "_reset_idx"):
            env._reset_idx(env_ids=env_ids)
        else:
            env.reset()
        if args.random_robot_init:
            for local_env_id, row_index in enumerate(wave):
                root_pos, root_quat = sample_robot_pose(args, row_index)
                reset_robot_root_env(env, robot, local_env_id, Pose.from_arrays(root_pos, root_quat))
        if hasattr(env, "sim") and hasattr(env.sim, "forward"):
            env.sim.forward()
        env.scene.update(dt=env.physics_dt)
        for local_env_id, row_index in enumerate(wave):
            rows.append(
                ScenarioRow(
                    schema_version=1,
                    row_index=row_index,
                    scenario_id=f"task1_{row_index:05d}",
                    seed=int(args.seed) + row_index,
                    robot=robot_pose(env, local_env_id),
                    object=asset_pose(env, "object", local_env_id),
                    placement=asset_pose(env, "placement", local_env_id),
                ).with_hash()
            )
    return rows


def env_robot_state(env, env_id: int, body_joint_sim_ids_t: Any):
    from run_kimodo_sonic_isaaclab import torch_to_np
    from sonic_policy import G1_DEFAULT_ANGLES_ISAACLAB, RobotState, projected_gravity_from_root_quat

    robot = env.scene["robot"]
    root_quat = torch_to_np(robot.data.root_quat_w[int(env_id)]).astype(np.float32)
    joint_pos_abs = robot.data.joint_pos[int(env_id), body_joint_sim_ids_t]
    joint_vel = robot.data.joint_vel[int(env_id), body_joint_sim_ids_t]
    root_ang_vel_b = getattr(robot.data, "root_ang_vel_b", None)
    base_ang_vel = (
        torch_to_np(root_ang_vel_b[int(env_id)]).astype(np.float32)
        if root_ang_vel_b is not None
        else np.zeros((3,), dtype=np.float32)
    )
    return RobotState(
        root_quat_wxyz=root_quat,
        joint_pos_rel_isaaclab=torch_to_np(joint_pos_abs).astype(np.float32) - G1_DEFAULT_ANGLES_ISAACLAB,
        joint_vel_isaaclab=torch_to_np(joint_vel).astype(np.float32),
        base_ang_vel=base_ang_vel,
        gravity_dir=projected_gravity_from_root_quat(root_quat),
    )


def make_dummy_reference(env, env_id: int, frames: int = 96):
    from run_kimodo_sonic_isaaclab import torch_to_np
    from sonic_policy import G1_DEFAULT_ANGLES_MUJOCO, ReferenceMotion

    robot = env.scene["robot"]
    root_state = torch_to_np(robot.data.root_state_w[int(env_id)]).astype(np.float32)
    qpos = np.zeros((int(frames), 36), dtype=np.float32)
    qpos[:, :3] = root_state[:3]
    qpos[:, 3:7] = root_state[3:7]
    qpos[:, 7:] = G1_DEFAULT_ANGLES_MUJOCO
    return ReferenceMotion.from_qpos(
        qpos,
        source_fps=50.0,
        control_hz=50.0,
        source_path=f"task1_paired_dummy_reference_env_{env_id}",
    )


def share_decoder_session(policies: list[Any]) -> bool:
    if len(policies) < 2:
        return False
    first = policies[0]
    session = getattr(first, "decoder_session", None)
    if session is None:
        return False
    for policy in policies[1:]:
        if hasattr(policy, "decoder_session"):
            old = policy.decoder_session
            policy.decoder_session = session
            with contextlib.suppress(Exception):
                if old is not session and hasattr(old, "__del__"):
                    old.__del__()
    return True


def build_gr00t_state_values_env(robot, env_id: int, sim_joint_names: list[str], joint_limits: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    import torch
    try:
        from isaaclab.utils.math import quat_apply_inverse
    except ImportError:
        # IsaacLab 2.3 exposes the same operation under its older name.
        from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
    from wbc_play_groot_sonic import gather_hand_open01, gather_joint_values

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


def renderer_supports_env_id(gs_render: Any) -> bool:
    return any(hasattr(gs_render, name) for name in ("render_env", "render_from_env", "render_with_env_id"))


def _mirror_env_into_foreground_sandbox(env, target_env_id: int, sandbox_env_id: int) -> None:
    """Copy one rollout clone into the dedicated single-camera foreground clone.

    Isaac Sim 4.5's TiledCamera semantic path can return stale/wrong clone
    pixels after vectorized resets.  The sandbox uses the same local scene
    state but only its proven single-camera output is read for mesh overlay.
    """

    import torch

    target_env_id = int(target_env_id)
    sandbox_env_id = int(sandbox_env_id)
    if target_env_id == sandbox_env_id:
        return
    env_ids = torch.as_tensor([sandbox_env_id], device=env.device, dtype=torch.long)
    source_origin = env.scene.env_origins[target_env_id]
    sandbox_origin = env.scene.env_origins[sandbox_env_id]

    robot = env.scene["robot"]
    root_state = robot.data.root_state_w[target_env_id : target_env_id + 1].clone()
    root_state[:, :3] += (sandbox_origin - source_origin).view(1, 3)
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
    joint_pos = robot.data.joint_pos[target_env_id : target_env_id + 1].clone()
    joint_vel = robot.data.joint_vel[target_env_id : target_env_id + 1].clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)

    for asset_name in ("object", "placement"):
        asset = env.scene[asset_name]
        asset_state = asset.data.root_state_w[target_env_id : target_env_id + 1].clone()
        asset_state[:, :3] += (sandbox_origin - source_origin).view(1, 3)
        asset.write_root_pose_to_sim(asset_state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(asset_state[:, 7:13], env_ids=env_ids)

    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    calibration = getattr(env, "_task1_headcam_calib", None)
    if calibration is not None:
        apply_head_camera_mode(
            env,
            calibration,
            getattr(env, "_task1_headcam_extrinsic_mode", "root_calib_attached_local"),
            render=False,
        )


def _capture_gs_frame_direct(env, gs_render: Any, robot, sim_joint_names: list[str], env_id: int) -> np.ndarray:
    if gs_render is None:
        raise RuntimeError("Mandatory GS renderer is not initialized")
    wait_for_head_camera_output(env)
    for name in ("render_env", "render_from_env", "render_with_env_id"):
        method = getattr(gs_render, name, None)
        if method is None:
            continue
        frame = method(env=env, env_id=int(env_id), robot=robot, sim_joint_names=sim_joint_names)
        if isinstance(frame, tuple):
            frame = frame[0]
        return np.asarray(frame, dtype=np.uint8)[..., :3]
    if int(env_id) == 0:
        _, _, frame = build_gr00t_observation(
            env,
            robot,
            sim_joint_names,
            resolve_joint_pos_limits(robot, sim_joint_names),
            {"video": {"delta_indices": [0], "modality_keys": ["ego_view"]}, "state": {"delta_indices": [0], "modality_keys": []}},
            TASK1_PROMPT,
            gs_render,
        )
        return np.asarray(frame, dtype=np.uint8)[..., :3]
    raise RuntimeError(
        "Multi-env GS rendering requires an env_id-aware renderer API. "
        "This renderer supports env_id=0 only."
    )


def capture_gs_frame_env(
    env,
    gs_render: Any,
    robot,
    sim_joint_names: list[str],
    env_id: int,
    adapter: EvaluationTaskAdapter | None = None,
) -> np.ndarray:
    """Render GS background with unconditional Isaac mesh foreground overlay.

    When enabled, foreground pixels are read only from a reserved single-env
    sandbox clone.  This avoids the Isaac 4.5 vectorized semantic-camera bug
    while preserving the target clone's exact local robot/object/placement
    state and calibrated camera pose.
    """

    sandbox_env_id = getattr(env, "_task1_mesh_foreground_sandbox_env_id", None)
    if sandbox_env_id is None or int(env_id) == int(sandbox_env_id):
        return _capture_gs_frame_direct(env, gs_render, robot, sim_joint_names, env_id)
    mirror = (
        adapter.mirror_env_into_foreground_sandbox
        if adapter is not None and adapter.mirror_env_into_foreground_sandbox is not None
        else _mirror_env_into_foreground_sandbox
    )
    mirror(env, int(env_id), int(sandbox_env_id))
    return _capture_gs_frame_direct(env, gs_render, robot, sim_joint_names, int(sandbox_env_id))


def refresh_eval_head_camera(args: argparse.Namespace, runtime: dict[str, Any]) -> None:
    """Apply the production replay camera pose immediately before rendering."""

    calibration = runtime.get("headcam_calib")

    # after reset. Reapplying it every frame would cancel d455 link motion.
    if calibration is not None and args.headcam_extrinsic_mode == "root_calib_each_frame":
        apply_head_camera_mode(
            runtime["env"],
            calibration,
            args.headcam_extrinsic_mode,
            render=False,
        )


def build_gr00t_observation_env(
    env,
    gs_render: Any,
    robot,
    sim_joint_names: list[str],
    joint_limits: dict[str, tuple[float, float]],
    modality_configs: dict[str, Any],
    task_description: str,
    env_id: int,
    hand_state_open01: tuple[float, float] | None = None,
    adapter: EvaluationTaskAdapter | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    if int(env_id) == 0 and not renderer_supports_env_id(gs_render):
        observation, frame_source, frame = build_gr00t_observation(
            env,
            robot,
            sim_joint_names,
            joint_limits,
            modality_configs,
            task_description,
            gs_render,
        )
        if frame_source != "gs_zbuffer":
            raise RuntimeError(f"Mandatory GS observation returned source={frame_source!r}")
        if hand_state_open01 is not None:
            state_horizon = len(modality_delta_indices(modality_configs["state"]))
            observation["state"]["left_hand_gripper"] = repeat_to_horizon(
                np.asarray([[hand_state_open01[0]]], dtype=np.float32), state_horizon
            )
            observation["state"]["right_hand_gripper"] = repeat_to_horizon(
                np.asarray([[hand_state_open01[1]]], dtype=np.float32), state_horizon
            )
        return observation, np.asarray(frame, dtype=np.uint8)[..., :3]

    frame = capture_gs_frame_env(env, gs_render, robot, sim_joint_names, env_id, adapter)
    state_values = build_gr00t_state_values_env(robot, env_id, sim_joint_names, joint_limits)
    if hand_state_open01 is not None:
        state_values["left_hand_gripper"] = np.asarray([hand_state_open01[0]], dtype=np.float32)
        state_values["right_hand_gripper"] = np.asarray([hand_state_open01[1]], dtype=np.float32)
    observation: dict[str, Any] = {"video": {}, "state": {}, "language": {}}
    video_horizon = len(modality_delta_indices(modality_configs["video"]))
    for key in modality_keys(modality_configs["video"]):
        if key != "ego_view":
            raise KeyError(f"Unsupported GR00T video key for this evaluator: {key}")
        observation["video"][key] = repeat_to_horizon(frame[None, ...].astype(np.uint8), video_horizon)
    state_horizon = len(modality_delta_indices(modality_configs["state"]))
    for key in modality_keys(modality_configs["state"]):
        if key not in state_values:
            raise KeyError(f"Unsupported GR00T state key for this evaluator: {key}")
        observation["state"][key] = repeat_to_horizon(state_values[key][None, ...].astype(np.float32), state_horizon)
    language_cfg = modality_configs.get("language", {"delta_indices": [0], "modality_keys": ["annotation.human.action.task_description"]})
    language_horizon = len(modality_delta_indices(language_cfg))
    for key in modality_keys(language_cfg):
        observation["language"][key] = [[task_description for _ in range(language_horizon)]]
    return observation, frame


def binarize_right_gripper(open01: float) -> float:
    return float(contract_binarize_right_gripper(open01))


def apply_manifest_initialization(args: argparse.Namespace, env, rows: list[ScenarioRow]) -> None:
    for env_id, row in enumerate(rows):
        reset_env_slot_to_manifest(env, env_id, row)
    if args.enable_head_hand:
        hard_lock_joint_state(
            env.scene["robot"],
            sim_joint_names=list(env.scene["robot"].data.joint_names),
            joint_names=HEAD_JOINT_NAMES,
            value=float(args.head_lock_rad),
        )
    if hasattr(env, "sim") and hasattr(env.sim, "forward"):
        env.sim.forward()
    env.scene.update(dt=env.physics_dt)


def update_success_state(runtime: dict[str, Any], slot: EnvSlot) -> dict[str, Any]:
    """Append CFG-declared asset state and evaluate the task-owned contract."""

    env = runtime["env"]
    for asset_name in runtime["success_state_assets"]:
        state = env.scene[asset_name].data.root_state_w[int(slot.env_id)].detach().cpu().numpy()
        history = slot.success_state_history.setdefault(asset_name, {"root_pose": [], "root_velocity": []})
        history["root_pose"].append(np.asarray(state[:7], dtype=np.float64).copy())
        history["root_velocity"].append(np.asarray(state[7:13], dtype=np.float64).copy())
    payload: dict[str, np.ndarray] = {}
    for asset_name, streams in slot.success_state_history.items():
        payload[f"{asset_name}_root_pose"] = np.stack(streams["root_pose"], axis=0)
        payload[f"{asset_name}_root_velocity"] = np.stack(streams["root_velocity"], axis=0)
    result = runtime["replay_success_evaluator"](payload, runtime["replay_success_cfg"])
    if not isinstance(result, dict) or "ok" not in result:
        raise TypeError("replay_success evaluator must return a dict containing 'ok'")
    return result


def write_png(path: Path, frame_rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.asarray(frame_rgb, dtype=np.uint8)[..., :3]
    if not cv2.imwrite(str(path), frame[..., ::-1]):
        raise RuntimeError(f"Failed to write PNG: {path}")



def episode_paths(
    run_dir: Path,
    row: ScenarioRow,
    diagnostic_video_fn=diagnostic_video_required,
) -> tuple[Path, Path | None]:
    final_png = run_dir / "final_png" / f"{row.scenario_id}.png"
    video = run_dir / "videos" / row.scenario_id / "ego.mp4" if diagnostic_video_fn(row.row_index) else None
    return final_png, video


def finalize_slot(
    run_dir: Path,
    slot: EnvSlot,
    model_name: str,
    checkpoint_dir: Path,
    manifest_digest: str,
    adapter: EvaluationTaskAdapter | None = None,
) -> dict[str, Any]:
    if slot.scenario is None:
        raise RuntimeError("Cannot finalize empty slot")
    row = slot.scenario
    if slot.video is not None:
        slot.video.close()
    final_png = run_dir / "final_png" / f"{row.scenario_id}.png"
    video_path = slot.video.ego.path if slot.video is not None else None
    third_path = slot.video.third_person.path if slot.video is not None else None
    first_png = run_dir / "first_png" / f"{row.scenario_id}.png"
    if slot.first_frame is not None:
        write_png(first_png, slot.first_frame)
    else:
        first_png = Path("")
    if slot.final_frame is not None:
        write_png(final_png, slot.final_frame)
    else:
        final_png = Path("")
    lat = np.asarray(slot.latencies_ms, dtype=np.float64)
    terminal = slot.terminal_reason or "error"
    if terminal not in TERMINAL_REASONS:
        terminal = "error"
    return {
        "model": model_name,
        "checkpoint_dir": str(checkpoint_dir),
        "manifest_hash": manifest_digest,
        "row_hash": row.row_hash,
        "scenario_id": row.scenario_id,
        "row_index": row.row_index,
        "seed": row.seed,
        "env_id": slot.env_id,
        "robot_pos": json.dumps(row.robot.pos),
        "robot_quat_wxyz": json.dumps(row.robot.quat_wxyz),
        "object_pos": json.dumps(row.object.pos),
        "object_quat_wxyz": json.dumps(row.object.quat_wxyz),
        "placement_pos": json.dumps(row.placement.pos),
        "placement_quat_wxyz": json.dumps(row.placement.quat_wxyz),
        "success": terminal == "success",
        "terminal_reason": terminal,
        "control_steps": int(slot.steps),
        "gr00t_query_count": int(slot.query_count),
        "elapsed_s": float(time.time() - slot.started_s) if slot.started_s else 0.0,
        "gr00t_latency_mean_ms": float(np.mean(lat)) if lat.size else 0.0,
        "gr00t_latency_p50_ms": float(np.percentile(lat, 50)) if lat.size else 0.0,
        "gr00t_latency_p95_ms": float(np.percentile(lat, 95)) if lat.size else 0.0,
        "min_robot_root_height": float(slot.min_root_height) if np.isfinite(slot.min_root_height) else None,
        "fall": bool(slot.fall),
        "max_object_lift": float(slot.max_object_lift),
        "final_object_placement_delta": json.dumps(slot.final_delta or [None, None, None]),
        "final_object_speed": slot.final_object_speed,
        "success_details": canonical_json(slot.success_result),
        "first_image_path": str(first_png) if str(first_png) else "",
        "final_image_path": str(final_png) if str(final_png) else "",
        "diagnostic_video_path": str(video_path) if video_path is not None and video_path.is_file() else "",
        "third_person_video_path": str(third_path) if third_path is not None and third_path.is_file() else "",
        "video_frames": slot.video.frames if slot.video is not None else 0,
        "video_fps": slot.video.fps if slot.video is not None else None,
        "error": slot.error or "",
    }


def append_episode_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if exists:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            previous_fields = reader.fieldnames or []
            previous_rows = list(reader)
        if any(None in row for row in previous_rows):
            raise ValueError(f"CSV contains rows wider than its header: {path}")
        fields = list(dict.fromkeys(previous_fields + fields))
        if fields != previous_fields:
            temporary = path.with_suffix(".csv.tmp")
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(previous_rows)
                writer.writerows(rows)
            temporary.replace(path)
            return
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_tables(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = run_dir / "episodes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["scenario_id"])
        writer.writeheader()
        writer.writerows(rows)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(run_dir / "episodes.parquet", index=False)
    except Exception as exc:
        raise RuntimeError("Per-episode Parquet output is mandatory; install/use the repo environment with parquet support.") from exc


def read_episode_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def confidence_interval(successes: int, total: int) -> dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    p = successes / total
    z = 1.96
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return {"low": float(max(0.0, centre - half)), "high": float(min(1.0, centre + half))}


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    success_count = sum(str(row.get("success")).lower() in {"true", "1"} for row in rows)
    reason_counts: dict[str, int] = {}
    steps = []
    latencies = []
    for row in rows:
        reason = str(row.get("terminal_reason", ""))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        with contextlib.suppress(Exception):
            steps.append(float(row.get("control_steps", 0)))
        with contextlib.suppress(Exception):
            latencies.append(float(row.get("gr00t_latency_p50_ms", 0)))
    return {
        "episode_count": total,
        "success_count": success_count,
        "success_rate": float(success_count / total) if total else 0.0,
        "success_rate_wilson_95": confidence_interval(success_count, total),
        "terminal_reason_counts": reason_counts,
        "steps_mean": float(np.mean(steps)) if steps else 0.0,
        "steps_p50": float(np.percentile(steps, 50)) if steps else 0.0,
        "steps_p95": float(np.percentile(steps, 95)) if steps else 0.0,
        "gr00t_latency_p50_ms_mean": float(np.mean(latencies)) if latencies else 0.0,
    }


def write_aggregate(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    aggregate = aggregate_rows(rows)
    write_json(run_dir / "aggregate.json", aggregate)
    with (run_dir / "aggregate.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in aggregate.items():
            writer.writerow({"metric": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value})


def checkpoint_hash(path: Path) -> str | None:
    path = path.expanduser()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        return None
    candidates = sorted(p for p in path.rglob("*") if p.is_file() and p.stat().st_size < 1024 * 1024 * 1024)
    digest = hashlib.sha256()
    for item in candidates:
        rel = item.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_metadata(args: argparse.Namespace, run_dir: Path, manifest_digest: str, chosen_envs: int, fallback_reason: str | None, modality_configs: dict[str, Any]) -> None:
    assets = {
        "gs_scene_path": str(args.gs_scene_path),
        "headcam_calib": str(args.headcam_calib),
        "sonic_models_dir": str(args.models_dir),
    }
    asset_hashes = {
        key: sha256_file(Path(value).expanduser().resolve())
        for key, value in assets.items()
        if Path(value).expanduser().is_file()
    }
    payload = {
        "created_at_unix_s": time.time(),
        "model_name": args.model_name,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_hash": checkpoint_hash(args.checkpoint_dir),
        "git": git_revision(),
        "manifest_path": str(args.manifest),
        "manifest_hash": manifest_digest,
        "task": args.task,
        "task_description": args.task_description,
        "embodiment": args.embodiment,
        "evaluator_config": {
            "num_envs": int(args.num_envs),
            "chosen_env_count": int(chosen_envs),
            "fallback_reason": fallback_reason,
            "chunk_horizon": int(args.chunk_horizon),
            "control_hz": float(args.control_hz),
            "diagnostic_video_interval": int(args.diagnostic_video_interval),
            "diagnostic_video_views": ["ego", "third_person"],
            "max_control_steps": int(args.max_control_steps),
            "right_gripper_rule": "<0.6 -> 0, >=0.6 -> 1",
            "gripper_state_source": str(args.gripper_state_source),
            "head_lock_rad": float(args.head_lock_rad),
            "gs_foreground_mode": str(args.gs_foreground_mode),
            "gs_apply_training_color_match": bool(args.gs_apply_training_color_match),
            "terminal_reasons": sorted(TERMINAL_REASONS),
            "init_joint_state": getattr(args, "_init_joint_state_provenance", {"source": "sonic_default"}),
            "source_initial_states": getattr(args, "_source_initial_state_provenance", None),
            "initialization_source": "manual_manifest_rectangle",
        },
        "assets": assets,
        "asset_hashes": asset_hashes,
        "camera_mode": args.headcam_extrinsic_mode,
        "modality_config": summarize_modality_config(modality_configs),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    write_json(run_dir / "metadata.json", payload)


def validate_required_assets(args: argparse.Namespace) -> None:
    paths = [args.checkpoint_dir, args.gs_scene_path, args.headcam_calib, args.models_dir]
    if args.init_joint_state_from_indis and args.init_joint_state_dataset is None:
        paths.append(args.init_joint_state_json)
    missing = [str(Path(p).expanduser()) for p in paths if not Path(p).expanduser().exists()]
    if missing:
        raise FileNotFoundError("Mandatory evaluation assets are missing: " + ", ".join(missing))
    detected_backend = detect_gs_backend_from_ply(Path(args.gs_scene_path).expanduser().resolve())
    backend = detected_backend if args.gs_backend == "auto" else args.gs_backend
    if backend == "2dgs":
        backend_probe = Path(args.two_d_gs_real2sim_root).expanduser() / "z_buffer_merger_with_robot.py"
        renderer_root = Path(args.two_d_gs_real2sim_root).expanduser().parent / "2d-gaussian-splatting"
        backend_missing = [str(path) for path in (backend_probe, renderer_root) if not path.exists()]
        if backend_missing:
            raise FileNotFoundError("Mandatory 2DGS backend files are missing: " + ", ".join(backend_missing))
    elif backend == "3dgs" and not Path(args.gs_root).expanduser().exists():
        raise FileNotFoundError(f"Mandatory 3DGS backend root is missing: {args.gs_root}")
    scene_names = {
        "Isaac-Final-Walk-PickPlace-Render-Abs-v0": "B1_0716_aligned_0711.ply",
        "Isaac-Final-Task2-PickUpHat-Render-Abs-v0": "B1_0711.ply",
    }
    if args.task not in scene_names:
        raise RuntimeError(f"No static GS scene contract for task: {args.task}")
    expected_scene = scene_names[args.task]
    if Path(args.gs_scene_path).name != expected_scene:
        raise RuntimeError(
            f"{args.task} requires {expected_scene}, got {args.gs_scene_path}"
        )
    print(
        f"[GS] mandatory scene={args.gs_scene_path} backend={backend} "
        f"foreground={args.gs_foreground_mode}",
        flush=True,
    )


def prepare_env(
    args: argparse.Namespace,
    num_envs: int,
    adapter: EvaluationTaskAdapter | None = None,
):
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import pick_place  # noqa: F401
    # Importing these packages registers IsaacLab tasks; their local names are
    # intentionally discarded after registration.
    del isaaclab_tasks, pick_place
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from run_kimodo_sonic_isaaclab import apply_sonic_actuator_profile, disable_time_out
    from wbc_play_groot_sonic import patch_head_camera_calibration, patch_head_camera_for_gs

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=int(num_envs))
    evaluation_cfg = copy.deepcopy(getattr(env_cfg, "sim_evaluation", {}))
    if not evaluation_cfg:
        raise RuntimeError(f"Task {args.task} does not define env_cfg.sim_evaluation")
    replay_success_cfg = getattr(env_cfg, "replay_success", None)
    if not isinstance(replay_success_cfg, dict) or not callable(replay_success_cfg.get("evaluator")):
        raise RuntimeError(f"Task {args.task} replay_success must define a callable evaluator")
    if args.task_description is None:
        args.task_description = str(evaluation_cfg["task_description"])
    robot_init = evaluation_cfg["robot_init"]
    if args.init_x_range is None:
        args.init_x_range = tuple(robot_init["x_range"])
    if args.init_y_range is None:
        args.init_y_range = tuple(robot_init["y_range"])
    if args.init_yaw_center_deg is None:
        args.init_yaw_center_deg = float(robot_init["yaw_center_deg"])
    if args.init_yaw_jitter_deg is None:
        args.init_yaw_jitter_deg = tuple(robot_init["yaw_jitter_deg"])
    if args.init_robot_z is None:
        args.init_robot_z = float(robot_init["z"])
    if args.gs_foreground_mode == "isaac_mesh" and int(num_envs) > 1:
        env_cfg.scene.env_spacing = float(args.vector_env_spacing)
        print(
            f"[GS] isolated vectorized foreground scenes with env_spacing={env_cfg.scene.env_spacing:.1f}m.",
            flush=True,
        )
    if args.untiled_head_camera_debug:
        from isaaclab.sensors import Camera

        env_cfg.scene.head_camera.class_type = Camera
        print("[GS] using per-clone Camera render products for vectorized foreground debug.", flush=True)
    env_cfg.decimation = 4
    env_cfg.sim.dt = 1.0 / 200.0
    env_cfg.sim.render_interval = env_cfg.decimation
    # Keep this narrower banana distribution local to the paired evaluator.
    # The shared Task1 CFG remains the source of truth for collection/generation.
    reset_event_name = str(evaluation_cfg["reset_event_name"])
    reset_pair = getattr(env_cfg.events, reset_event_name)
    reset_pair.params = copy.deepcopy(reset_pair.params)
    reset_pair.params.update(copy.deepcopy(evaluation_cfg.get("reset_pose_ranges", {})))
    object_pose_range = reset_pair.params.get("object_pose_range", {})
    placement_pose_range = reset_pair.params.get("placement_pose_range", {})
    if args.object_x_range is not None:
        object_pose_range["x"] = tuple(float(value) for value in args.object_x_range)
    if args.object_y_range is not None:
        object_pose_range["y"] = tuple(float(value) for value in args.object_y_range)
    if args.object_yaw_range_deg is not None:
        object_pose_range["yaw"] = tuple(math.radians(float(value)) for value in args.object_yaw_range_deg)
    if args.placement_x_range is not None:
        placement_pose_range["x"] = tuple(float(value) for value in args.placement_x_range)
    if args.placement_y_range is not None:
        placement_pose_range["y"] = tuple(float(value) for value in args.placement_y_range)
    disable_time_out(env_cfg)
    headcam_calib = HeadCameraCalibration(args.headcam_calib) if args.headcam_calib is not None else None
    patch_head_camera_for_gs(env_cfg, enable_gs=True)
    patch_head_camera_calibration(env_cfg, headcam_calib)
    if args.headcam_extrinsic_mode == "root_calib_each_frame":
        # Match the 4090 replay/render pipeline exactly: detach the camera from
        # the robot articulation before construction, then author its calibrated
        # world pose immediately before every render.
        head_camera_cfg = env_cfg.scene.head_camera
        head_camera_cfg.prim_path = "{ENV_REGEX_NS}/head_camera"
        with contextlib.suppress(Exception):
            head_camera_cfg.offset = head_camera_cfg.offset.replace(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            )
        head_camera_cfg.offset.pos = (0.0, 0.0, 0.0)
        head_camera_cfg.offset.rot = (1.0, 0.0, 0.0, 0.0)
        head_camera_cfg.offset.convention = "world"
    elif args.headcam_extrinsic_mode == "legacy_task_cfg":
        # The old render task used the ROS offset below.  This mode exists

        env_cfg.scene.head_camera.offset.pos = LEGACY_TASK1_HEADCAM_OFFSET_POS
        env_cfg.scene.head_camera.offset.rot = LEGACY_TASK1_HEADCAM_OFFSET_ROT_WXYZ
        env_cfg.scene.head_camera.offset.convention = "ros"
    if args.sonic_actuator_profile:
        actuator_profile = (
            adapter.apply_actuator_profile
            if adapter is not None and adapter.apply_actuator_profile is not None
            else apply_sonic_actuator_profile
        )
        actuator_profile(
            env_cfg.scene.robot,
            preserve_aux_actuators=True,
            waist_stiffness_scale=args.sonic_waist_stiffness_scale,
        )
    from isaaclab.sim import PinholeCameraCfg
    import torch

    camera_cfg = copy.deepcopy(env_cfg.scene.head_camera)
    camera_cfg.prim_path = "{ENV_REGEX_NS}/evaluation_camera"
    camera_cfg.offset.pos = (0.0, 0.0, 0.0)
    camera_cfg.offset.rot = (1.0, 0.0, 0.0, 0.0)
    camera_cfg.offset.convention = "world"
    camera_cfg.data_types = ["rgb"]
    camera_cfg.width, camera_cfg.height = 960, 720
    camera_cfg.update_latest_camera_pose = True
    camera_cfg.spawn = PinholeCameraCfg(
        focal_length=1.93, focus_distance=3.0,
        horizontal_aperture=3.896, vertical_aperture=2.453,
        clipping_range=(0.01, 100.0), lock_camera=False,
    )
    env_cfg.scene.evaluation_camera = camera_cfg
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()
    origins = env.scene.env_origins
    eyes = origins + torch.tensor(env_cfg.viewer.eye, device=env.device)
    targets = origins + torch.tensor(env_cfg.viewer.lookat, device=env.device)
    env.scene["evaluation_camera"].set_world_poses_from_view(eyes=eyes, targets=targets)
    return env, env_cfg, headcam_calib


def setup_runtime(
    args: argparse.Namespace,
    num_envs: int,
    adapter: EvaluationTaskAdapter | None = None,
):
    import torch
    from run_kimodo_sonic_isaaclab import build_name_indices, resolve_env_action_order, resolve_robot_names
    from sonic_policy import G1_ISAACLAB_BODY_JOINT_NAMES

    env, env_cfg, headcam_calib = prepare_env(args, num_envs, adapter)
    robot = env.scene["robot"]
    sim_joint_names, _ = resolve_robot_names(robot)
    fallback_action_names = list(getattr(env_cfg, "env_action_joint_names", sim_joint_names))
    action_joint_names, action_joint_sim_ids, action_dim = resolve_env_action_order(env, sim_joint_names, fallback_action_names)
    body_joint_sim_ids = build_name_indices(sim_joint_names, list(G1_ISAACLAB_BODY_JOINT_NAMES), "SONIC body joint")
    body_action_ids = build_name_indices(action_joint_names, list(G1_ISAACLAB_BODY_JOINT_NAMES), "SONIC action joint")
    device = torch.device(env.device)
    if args.init_joint_state_from_indis:
        if args.init_joint_state_dataset is not None:
            init_body_joint_target, init_joint_provenance = load_indis_initial_body_pose(
                args.init_joint_state_dataset,
                args.init_joint_state_episodes,
                G1_ISAACLAB_BODY_JOINT_NAMES,
                mode=args.init_joint_state_mode,
                episode_index=args.init_joint_state_episode_index,
            )
        elif args.init_joint_state_json is not None:
            if args.init_joint_state_mode != "episode_first_frame":
                raise ValueError("--init_joint_state_mode median requires --init_joint_state_dataset")
            init_body_joint_target, init_joint_provenance = load_initial_body_pose_json(
                args.init_joint_state_json,
                G1_ISAACLAB_BODY_JOINT_NAMES,
            )
        else:
            raise ValueError(
                "--init_joint_state_from_indis requires --init_joint_state_json or "
                "--init_joint_state_dataset. Use --no-init_joint_state_from_indis only "
                "when the SONIC default pose is intentional."
            )
    else:
        from sonic_policy import G1_DEFAULT_ANGLES_ISAACLAB

        init_body_joint_target = np.asarray(G1_DEFAULT_ANGLES_ISAACLAB, dtype=np.float32).copy()
        init_joint_provenance = {
            "source": "sonic_default",
            "sonic_joint_order": list(G1_ISAACLAB_BODY_JOINT_NAMES),
            "body_joint_target": [float(value) for value in init_body_joint_target],
        }
    args._init_joint_state_provenance = init_joint_provenance
    print(
        "[INIT] body_joint_state=" + str(init_joint_provenance["source"])
        + f" episodes={init_joint_provenance.get('episodes', 0)} "
        + "target=" + np.array2string(init_body_joint_target, precision=3, separator=", "),
        flush=True,
    )
    runtime = {
        "env": env,
        "env_cfg": env_cfg,
        "headcam_calib": headcam_calib,
        "robot": robot,
        "device": device,
        "sim_joint_names": sim_joint_names,
        "action_joint_names": action_joint_names,
        "action_joint_sim_ids_t": torch.as_tensor(action_joint_sim_ids, device=device, dtype=torch.long),
        "action_dim": int(action_dim),
        "body_joint_sim_ids_t": torch.as_tensor(body_joint_sim_ids, device=device, dtype=torch.long),
        "body_joint_names": list(G1_ISAACLAB_BODY_JOINT_NAMES),
        "body_action_ids": body_action_ids,
        "joint_limits": resolve_joint_pos_limits(robot, sim_joint_names),
        "action_name_to_idx": {name: idx for idx, name in enumerate(action_joint_names)},
        "available_action_names": set(action_joint_names),
        "init_body_joint_target": init_body_joint_target,
        "evaluation_cfg": copy.deepcopy(env_cfg.sim_evaluation),
        "replay_success_cfg": env_cfg.replay_success,
        "replay_success_evaluator": env_cfg.replay_success["evaluator"],
        "success_state_assets": tuple(env_cfg.sim_evaluation["success_state_assets"]),
    }
    if args.source_initial_state_json is not None:
        states, provenance = load_source_initial_states(args.source_initial_state_json, action_joint_names)
        runtime["source_initial_states"] = states
        args._source_initial_state_provenance = provenance
        print(
            f"[INIT] source_initial_states={provenance['scenario_count']} path={provenance['path']}",
            flush=True,
        )
    env._task1_headcam_calib = headcam_calib
    env._task1_headcam_extrinsic_mode = str(args.headcam_extrinsic_mode)
    if args.clip_body_targets_to_limits:
        limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if limits is not None:
            runtime["body_soft_limits"] = limits[0, runtime["body_joint_sim_ids_t"]].to(device=device, dtype=torch.float32)
    return runtime


def make_policies(args: argparse.Namespace, runtime: dict[str, Any], num_envs: int) -> list[Any]:
    from sonic_policy import SonicPolicy

    first = SonicPolicy(
        make_dummy_reference(runtime["env"], 0),
        args.models_dir,
        use_cuda=not bool(args.sonic_cpu_onnx),
        load_encoder=False,
    )
    policies = [first]
    for env_id in range(1, int(num_envs)):
        policy = copy.copy(first)
        policy.reference_motion = make_dummy_reference(runtime["env"], env_id)
        for name, value in vars(first).items():
            if isinstance(value, np.ndarray):
                setattr(policy, name, value.copy())
        policy.init_ref_quat = policy.reference_motion.root_quat_wxyz[0].copy()
        policies.append(policy)
    print(f"[SONIC] policies={len(policies)} shared_decoder_session={len(policies) > 1}", flush=True)
    return policies


def reset_slot(
    args: argparse.Namespace,
    runtime: dict[str, Any],
    slot: EnvSlot,
    row: ScenarioRow,
    policy: Any,
    run_dir: Path,
    adapter: EvaluationTaskAdapter | None = None,
) -> None:
    import torch

    env = runtime["env"]
    robot = runtime["robot"]
    print(f"[RESET] begin row={row.row_index} env={slot.env_id}", flush=True)
    reset_fn = adapter.reset_env_slot_to_manifest if adapter is not None else reset_env_slot_to_manifest
    reset_fn(env, slot.env_id, row)
    print(f"[RESET] manifest state applied row={row.row_index} env={slot.env_id}", flush=True)
    if args.reset_to_sonic_default:
        restore_reset_joint_state(runtime, slot.env_id)
    if args.source_initial_state_json is not None:
        restore_source_initial_state(runtime, slot.env_id, row)
    if args.enable_head_hand:
        hard_lock_joint_state(robot, sim_joint_names=runtime["sim_joint_names"], joint_names=HEAD_JOINT_NAMES, value=float(args.head_lock_rad))
    print(f"[RESET] robot/camera state applied row={row.row_index} env={slot.env_id}", flush=True)
    if hasattr(env, "sim") and hasattr(env.sim, "forward"):
        env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    print(f"[RESET] camera warmup begin row={row.row_index} env={slot.env_id}", flush=True)
    wait_for_head_camera_output(env, render_count=args.camera_warmup_renders)
    print(f"[RESET] camera warmup complete row={row.row_index} env={slot.env_id}", flush=True)
    policy.reset(env_robot_state(env, slot.env_id, runtime["body_joint_sim_ids_t"]), frame_idx=0)
    slot.scenario = row
    slot.policy = policy
    slot.last_action = robot.data.joint_pos[slot.env_id, runtime["action_joint_sim_ids_t"]].to(device=runtime["device"], dtype=torch.float32).clone()
    slot.left_hand_state_open01 = 1.0
    slot.right_hand_state_open01 = 1.0
    slot.chunk = None
    slot.chunk_pos = 0
    slot.steps = 0
    slot.query_count = 0
    slot.latencies_ms.clear()
    slot.min_root_height = math.inf
    slot.initial_object_z = float(env.scene["object"].data.root_state_w[slot.env_id, 2].detach().cpu().item())
    slot.max_object_lift = 0.0
    slot.final_delta = None
    slot.final_object_speed = None
    slot.success_state_history.clear()
    success_fn = (
        adapter.update_success_state
        if adapter is not None and adapter.update_success_state is not None
        else update_success_state
    )
    slot.success_result = success_fn(runtime, slot)
    slot.fall = False
    slot.terminal_reason = None
    slot.error = None
    slot.first_frame = None
    slot.started_s = time.time()
    slot.final_frame = None
    slot.video = (
        EpisodeVideoRecorder(run_dir / "videos" / row.scenario_id, fps=float(args.control_hz))
        if diagnostic_video_required(row.row_index, args.diagnostic_video_interval) else None
    )


def request_chunk(
    args: argparse.Namespace,
    runtime: dict[str, Any],
    client: RobotInferenceClient,
    modality_configs: dict[str, Any],
    gs_render: Any,
    slot: EnvSlot,
    adapter: EvaluationTaskAdapter | None = None,
) -> None:
    print(f"[QUERY] observation begin row={slot.scenario.row_index if slot.scenario else -1} env={slot.env_id}", flush=True)
    refresh_eval_head_camera(args, runtime)
    observation, frame = build_gr00t_observation_env(
        runtime["env"],
        gs_render,
        runtime["robot"],
        runtime["sim_joint_names"],
        runtime["joint_limits"],
        modality_configs,
        args.task_description,
        slot.env_id,
        (
            (slot.left_hand_state_open01, slot.right_hand_state_open01)
            if args.gripper_state_source == "command_binary"
            else None
        ),
        adapter,
    )
    print(f"[QUERY] observation complete row={slot.scenario.row_index if slot.scenario else -1} env={slot.env_id}", flush=True)
    if slot.first_frame is None:
        slot.first_frame = np.asarray(frame, dtype=np.uint8).copy()
    started = time.perf_counter()
    action_dict, _ = client.get_action(observation, options=None)
    print(f"[QUERY] inference complete row={slot.scenario.row_index if slot.scenario else -1} env={slot.env_id}", flush=True)
    slot.latencies_ms.append((time.perf_counter() - started) * 1000.0)
    chunk = parse_sonic_actions(action_dict)
    if chunk.horizon < int(args.chunk_horizon):
        raise RuntimeError(f"GR00T returned {chunk.horizon} steps, expected at least {args.chunk_horizon}")
    slot.chunk = chunk
    slot.chunk_pos = 0
    slot.query_count += 1
    slot.final_frame = frame


def synchronize_reset_wave(args: argparse.Namespace, runtime: dict[str, Any], slots: list[EnvSlot]) -> None:
    """Publish reset poses to Fabric before the first policy observation."""

    import torch

    env = runtime["env"]
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    # The attached-camera local transform depends on body poses.  Those poses
    # are stale until every clone's reset writes have reached PhysX/Fabric.
    # Compute it only after the full reset wave is synchronized.
    if runtime["headcam_calib"] is not None:
        apply_head_camera_mode(env, runtime["headcam_calib"], args.headcam_extrinsic_mode, render=False)
    wait_for_head_camera_output(env, render_count=args.camera_warmup_renders)
    for slot in slots:
        if slot.scenario is None or slot.policy is None:
            continue
        slot.policy.reset(
            env_robot_state(env, slot.env_id, runtime["body_joint_sim_ids_t"]),
            frame_idx=0,
        )
        slot.last_action = runtime["robot"].data.joint_pos[
            slot.env_id, runtime["action_joint_sim_ids_t"]
        ].to(device=runtime["device"], dtype=torch.float32).clone()


def mark_slot_error(slot: EnvSlot, exc: BaseException) -> None:
    slot.terminal_reason = "error"
    slot.error = f"{type(exc).__name__}: {exc}"


def step_active_slots(
    args: argparse.Namespace,
    runtime: dict[str, Any],
    slots: list[EnvSlot],
    gs_render: Any,
    adapter: EvaluationTaskAdapter | None = None,
) -> None:
    import torch

    env = runtime["env"]
    robot = runtime["robot"]
    action_batch = robot.data.joint_pos[:, runtime["action_joint_sim_ids_t"]].to(device=runtime["device"], dtype=torch.float32).clone()
    for slot in slots:
        if not slot.active:
            continue
        if slot.chunk is None or slot.chunk_pos >= int(args.chunk_horizon):
            raise RuntimeError(f"Active slot env_id={slot.env_id} has no action chunk")
        chunk_idx = slot.chunk_pos
        state = env_robot_state(env, slot.env_id, runtime["body_joint_sim_ids_t"])
        token = slot.chunk.token[0, chunk_idx]
        sonic_action = slot.policy.act_with_token(state, token, frame_idx=slot.steps, advance=True)
        q_target = torch.as_tensor(sonic_action.q_target_isaaclab.astype(np.float32, copy=False), device=runtime["device"], dtype=torch.float32)
        if "body_soft_limits" in runtime:
            q_target = torch.clamp(q_target, runtime["body_soft_limits"][:, 0], runtime["body_soft_limits"][:, 1])
        env_action = slot.last_action.clone()
        for local_idx, action_idx in enumerate(runtime["body_action_ids"]):
            env_action[action_idx] = q_target[local_idx]
        left_open = float(slot.chunk.left_hand_gripper[0, chunk_idx, 0])
        right_open = binarize_right_gripper(float(slot.chunk.right_hand_gripper[0, chunk_idx, 0]))
        slot.left_hand_state_open01 = binarize_right_gripper(left_open)
        slot.right_hand_state_open01 = right_open
        for name, target in scalar_hand_targets(left_open, "left", runtime["available_action_names"], runtime["joint_limits"]).items():
            env_action[runtime["action_name_to_idx"][name]] = float(target)
        for name, target in scalar_hand_targets(right_open, "right", runtime["available_action_names"], runtime["joint_limits"]).items():
            env_action[runtime["action_name_to_idx"][name]] = float(target)
        apply_head_action_targets(env_action, runtime["action_name_to_idx"], enabled=bool(args.enable_head_hand), value=float(args.head_lock_rad))
        action_batch[slot.env_id] = env_action
        slot.last_action = env_action
        slot.chunk_pos += 1

    env.step(action_batch)
    if args.enable_head_hand:
        hard_lock_joint_state(robot, sim_joint_names=runtime["sim_joint_names"], joint_names=HEAD_JOINT_NAMES, value=float(args.head_lock_rad))

    for slot in slots:
        if not slot.active:
            continue
        slot.steps += 1
        root_pos = robot.data.root_pos_w[slot.env_id].detach().cpu().numpy().astype(np.float32)
        slot.min_root_height = min(slot.min_root_height, float(root_pos[2]))
        success_fn = adapter.update_success_state if adapter is not None and adapter.update_success_state is not None else update_success_state
        metrics = success_fn(runtime, slot)
        slot.success_result = metrics
        slot.max_object_lift = float(metrics.get("max_object_lift", 0.0))
        slot.final_delta = metrics.get("final_object_placement_delta")
        slot.final_object_speed = metrics.get("final_object_speed")
        if slot.video is not None:
            refresh_eval_head_camera(args, runtime)
            frame = capture_gs_frame_env(
                env,
                gs_render,
                robot,
                runtime["sim_joint_names"],
                slot.env_id,
                adapter,
            )
            third_frame = capture_third_person(env, slot.env_id)
            slot.video.write(frame, third_frame)
            slot.final_frame = frame
        if metrics["ok"]:
            slot.terminal_reason = "success"
        else:
            if (not np.isfinite(root_pos).all()) or float(root_pos[2]) < float(args.fall_height):
                slot.fall = True
                if args.stop_on_fall:
                    slot.terminal_reason = "fall"
            if slot.terminal_reason is None and slot.steps >= int(args.max_control_steps):
                slot.terminal_reason = "horizon"








def aggregate_paired(output_root: Path, manifest_digest: str) -> dict[str, Any]:
    by_model = {}
    for model in ("real", "sim", "custom"):
        rows = read_episode_csv(output_root / model / "sim_eval" / "episodes.csv")
        if rows:
            by_model[model] = rows
    paired = {}
    if "real" in by_model and "sim" in by_model:
        real = {row["scenario_id"]: row for row in by_model["real"]}
        sim = {row["scenario_id"]: row for row in by_model["sim"]}
        ids = sorted(set(real) & set(sim))
        wins = losses = ties = 0
        for sid in ids:
            r = str(real[sid].get("success")).lower() in {"true", "1"}
            s = str(sim[sid].get("success")).lower() in {"true", "1"}
            wins += int(s and not r)
            losses += int(r and not s)
            ties += int(r == s)
        paired = {
            "paired_count": len(ids),
            "sim_wins": wins,
            "sim_losses": losses,
            "ties": ties,
            "missing_real": sorted(set(sim) - set(real)),
            "missing_sim": sorted(set(real) - set(sim)),
        }
    aggregate = {
        "manifest_hash": manifest_digest,
        "models": {model: aggregate_rows(rows) for model, rows in by_model.items()},
        "paired": paired,
    }
    write_json(output_root / "paired_aggregate.json", aggregate)
    return aggregate


def make_default_adapter() -> EvaluationTaskAdapter:
    """Build the immutable Task1 adapter consumed by the shared runner."""

    return EvaluationTaskAdapter(
        name="Task1",
        task_id=DEFAULT_TASK,
        manifest_row_count=MANIFEST_ROW_COUNT,
        output_root=DEFAULT_OUTPUT_ROOT,
        default_init_joint_state_json=DEFAULT_IN_DIS_803_INIT_POSE,
        default_init_joint_state_dataset=DEFAULT_IN_DIS_803_LEROBOT,
        default_manifest=REPO_ROOT.parent / "configs/manifests/task1_heldout100_independent_xy5cm.jsonl",
        default_chunk_horizon=16,
        parse_args=parse_args,
        validate_required_assets=validate_required_assets,
        load_processor_modality_configs=load_processor_modality_configs,
        read_manifest=read_manifest,
        write_manifest=write_manifest,
        manifest_hash=manifest_hash,
        select_manifest_rows=select_manifest_rows,
        pending_rows=pending_rows,
        reset_env_slot_to_manifest=reset_env_slot_to_manifest,
        setup_runtime=setup_runtime,
        generate_manifest_with_env=generate_manifest_with_env,
        make_policies=make_policies,
        init_gs_renderer=init_gs_renderer,
        reset_slot=reset_slot,
        synchronize_reset_wave=synchronize_reset_wave,
        request_chunk=request_chunk,
        step_active_slots=step_active_slots,
        capture_gs_frame_env=capture_gs_frame_env,
        mirror_env_into_foreground_sandbox=_mirror_env_into_foreground_sandbox,
        finalize_slot=finalize_slot,
        append_episode_rows=append_episode_rows,
        read_episode_csv=read_episode_csv,
        write_tables=write_tables,
        write_aggregate=write_aggregate,
        write_metadata=write_metadata,
        diagnostic_video_required=diagnostic_video_required,
        env_slot_type=EnvSlot,
        robot_inference_client=RobotInferenceClient,
        summarize_modality_config=summarize_modality_config,
        update_success_state=update_success_state,
        aggregate_paired=aggregate_paired,
        refresh_eval_head_camera=refresh_eval_head_camera,
    )


def main() -> None:
    adapter = make_default_adapter()
    run_evaluation(parse_args(adapter), adapter)


if __name__ == "__main__":
    main()
