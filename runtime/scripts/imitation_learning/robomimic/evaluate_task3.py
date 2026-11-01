#!/usr/bin/env python3
"""Independent Task3 adapter for the paired GR00T + SONIC rollout engine."""

from __future__ import annotations

import hashlib
import atexit
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

import evaluation_runtime as shared
from evaluation_runner import EvaluationTaskAdapter, run_evaluation


TASK3_SCENARIO_COUNT = 100
TASK3_RENDER_TASK_ID = "Isaac-Final-Task3-CarryBasketKickDoor-Render-Abs-v0"
TASK3_ASSETS_ROOT = Path(
    os.environ.get(
        "TASK3_EVAL_BUNDLE_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "assets/task3"),
    )
).expanduser()
TASK3_SCENE_USD_PATH = Path(
    os.environ.get(
        "TASK3_SCENE_USD_PATH",
        str(Path(__file__).resolve().parents[3].parent / "assets/task3/scene.usdc"),
    )
).expanduser()
TASK3_DOOR_PLY = Path(
    os.environ.get(
        "TASK3_EVAL_DOOR_PLY",
        str(TASK3_ASSETS_ROOT / "assets/GS_Assets/B1_art/obj_1_gs_task3_world.ply"),
    )
).expanduser()
TASK3_JOINT_INFO = Path(
    os.environ.get(
        "TASK3_EVAL_JOINT_INFO",
        str(TASK3_ASSETS_ROOT / "assets/GS_Assets/B1_art/joint_info.json"),
    )
).expanduser()
TASK3_SCENARIO_BANK = Path(
    os.environ.get(
        "TASK3_EVAL_SCENARIO_BANK",
        str(TASK3_ASSETS_ROOT / "assets/evaluation/task3_training_scenarios_386_full_init.jsonl"),
    )
).expanduser()
TASK3_INITIAL_BODY_POSE = Path(
    os.environ.get(
        "TASK3_INIT_JOINT_STATE_JSON",
        str(TASK3_ASSETS_ROOT / "assets/evaluation/task3_initial_body_pose_episode_000000.json"),
    )
).expanduser()

SCENE_BASE_POS = (0.15325, 0.10805, 1.50718)
SCENE_BASE_ROT_WXYZ = (0.70710678, 0.0, 0.0, -0.70710678)


class Task3CarryboxColorMatcher:
    """Apply the exact carrybox training-video filter to online RGB frames."""

    def __init__(self) -> None:
        from scripts.tools.match_gs_video_colors import CARRYBOX_REAL_FILTER

        self._filter = CARRYBOX_REAL_FILTER
        self._process: subprocess.Popen[bytes] | None = None
        self._shape: tuple[int, int] | None = None

    def _start(self, height: int, width: int) -> None:
        self.close()
        self._shape = (int(height), int(width))
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-i",
                "pipe:0",
                "-vf",
                self._filter,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        frame = np.ascontiguousarray(rgb, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Task3 RGB frame must have shape HxWx3, got {frame.shape}")
        height, width = frame.shape[:2]
        if self._process is None or self._shape != (height, width):
            self._start(height, width)
        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write(frame.tobytes())
        self._process.stdin.flush()
        output = bytearray()
        remaining = frame.nbytes
        while remaining:
            chunk = self._process.stdout.read(remaining)
            if not chunk:
                raise RuntimeError("Task3 carrybox ffmpeg color matcher exited early")
            output.extend(chunk)
            remaining -= len(chunk)
        return np.frombuffer(output, dtype=np.uint8).reshape(frame.shape).copy()

    def close(self) -> None:
        process, self._process = self._process, None
        self._shape = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


_TASK3_COLOR_MATCHER: Task3CarryboxColorMatcher | None = None


def _prepare_task3_color_matcher() -> None:
    """Fork ffmpeg before Isaac/CUDA starts any worker threads."""

    global _TASK3_COLOR_MATCHER
    if os.environ.get("TASK3_APPLY_CARRYBOX_COLOR_MATCH", "1") == "0":
        return
    if _TASK3_COLOR_MATCHER is None:
        _TASK3_COLOR_MATCHER = Task3CarryboxColorMatcher()
        _TASK3_COLOR_MATCHER._start(height=480, width=640)
        atexit.register(_TASK3_COLOR_MATCHER.close)


@dataclass(frozen=True)
class ScenarioRow:
    schema_version: int
    row_index: int
    scenario_id: str
    seed: int
    robot: shared.Pose
    object: shared.Pose
    placement: shared.Pose
    scene: shared.Pose
    source_scenario_index: int = -1
    source_demo: str = ""
    robot_root_velocity: tuple[float, ...] = (0.0,) * 6
    robot_body_joint_names: tuple[str, ...] = ()
    robot_body_joint_position: tuple[float, ...] = ()
    robot_body_joint_velocity: tuple[float, ...] = ()
    robot_hand_open: tuple[float, ...] = (1.0, 1.0)
    object_root_velocity: tuple[float, ...] = (0.0,) * 6
    placement_root_velocity: tuple[float, ...] = (0.0,) * 6
    door_joint_position: float = 0.0
    door_joint_velocity: float = 0.0
    door_contact_latched: bool = False
    scene_se2: shared.SceneSE2 = field(default_factory=shared.SceneSE2.identity)
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
            "scene": asdict(self.scene),
            "source_scenario_index": int(self.source_scenario_index),
            "source_demo": str(self.source_demo),
            "robot_root_velocity": list(self.robot_root_velocity),
            "robot_body_joint_names": list(self.robot_body_joint_names),
            "robot_body_joint_position": list(self.robot_body_joint_position),
            "robot_body_joint_velocity": list(self.robot_body_joint_velocity),
            "robot_hand_open": list(self.robot_hand_open),
            "object_root_velocity": list(self.object_root_velocity),
            "placement_root_velocity": list(self.placement_root_velocity),
            "door_joint_position": float(self.door_joint_position),
            "door_joint_velocity": float(self.door_joint_velocity),
            "door_contact_latched": bool(self.door_contact_latched),
            "scene_se2": asdict(self.scene_se2),
        }
        if include_hash:
            payload["row_hash"] = str(self.row_hash)
        return payload

    def with_hash(self) -> "ScenarioRow":
        digest = hashlib.sha256(
            shared.canonical_json(self.canonical_payload(include_hash=False)).encode("utf-8")
        ).hexdigest()
        return ScenarioRow(
            schema_version=self.schema_version,
            row_index=self.row_index,
            scenario_id=self.scenario_id,
            seed=self.seed,
            robot=self.robot,
            object=self.object,
            placement=self.placement,
            scene=self.scene,
            source_scenario_index=self.source_scenario_index,
            source_demo=self.source_demo,
            robot_root_velocity=self.robot_root_velocity,
            robot_body_joint_names=self.robot_body_joint_names,
            robot_body_joint_position=self.robot_body_joint_position,
            robot_body_joint_velocity=self.robot_body_joint_velocity,
            robot_hand_open=self.robot_hand_open,
            object_root_velocity=self.object_root_velocity,
            placement_root_velocity=self.placement_root_velocity,
            door_joint_position=self.door_joint_position,
            door_joint_velocity=self.door_joint_velocity,
            door_contact_latched=self.door_contact_latched,
            scene_se2=self.scene_se2,
            row_hash=digest,
        )


def _pose(payload: dict[str, Any], name: str) -> shared.Pose:
    value = payload[name]
    return shared.Pose.from_arrays(value["pos"], value["quat_wxyz"])


def scenario_from_json(payload: dict[str, Any]) -> ScenarioRow:
    stored_hash = str(payload.get("row_hash", ""))
    raw_payload = {key: value for key, value in payload.items() if key != "row_hash"}
    expected_hash = hashlib.sha256(shared.canonical_json(raw_payload).encode("utf-8")).hexdigest()
    if stored_hash and stored_hash != expected_hash:
        raise ValueError(f"Manifest row {payload.get('row_index')} hash mismatch")
    scene_se2 = payload.get("scene_se2", {})
    row = ScenarioRow(
        schema_version=int(payload["schema_version"]),
        row_index=int(payload["row_index"]),
        scenario_id=str(payload["scenario_id"]),
        seed=int(payload["seed"]),
        robot=_pose(payload, "robot"),
        object=_pose(payload, "object"),
        placement=_pose(payload, "placement"),
        scene=_pose(payload, "scene"),
        source_scenario_index=int(payload.get("source_scenario_index", -1)),
        source_demo=str(payload.get("source_demo", "")),
        robot_root_velocity=tuple(float(value) for value in payload.get("robot_root_velocity", (0.0,) * 6)),
        robot_body_joint_names=tuple(str(value) for value in payload.get("robot_body_joint_names", ())),
        robot_body_joint_position=tuple(float(value) for value in payload.get("robot_body_joint_position", ())),
        robot_body_joint_velocity=tuple(float(value) for value in payload.get("robot_body_joint_velocity", ())),
        robot_hand_open=tuple(float(value) for value in payload.get("robot_hand_open", (1.0, 1.0))),
        object_root_velocity=tuple(float(value) for value in payload.get("object_root_velocity", (0.0,) * 6)),
        placement_root_velocity=tuple(float(value) for value in payload.get("placement_root_velocity", (0.0,) * 6)),
        door_joint_position=float(payload.get("door_joint_position", 0.0)),
        door_joint_velocity=float(payload.get("door_joint_velocity", 0.0)),
        door_contact_latched=bool(payload.get("door_contact_latched", False)),
        scene_se2=shared.SceneSE2.from_arrays(
            scene_se2.get("translation_xy", (0.0, 0.0)),
            scene_se2.get("delta_yaw", 0.0),
        ),
        row_hash=stored_hash,
    )
    return row if row.row_hash else row.with_hash()


def read_manifest(path: Path, expected_count: int | None = TASK3_SCENARIO_COUNT) -> list[ScenarioRow]:
    rows: list[ScenarioRow] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = scenario_from_json(json.loads(line))
            if row.row_index != len(rows) or row.scenario_id != f"task3_carry_basket_{row.row_index:05d}":
                raise ValueError(f"Task3 manifest order/id mismatch at line {line_no}")
            rows.append(row)
    if expected_count is not None and len(rows) != int(expected_count):
        raise ValueError(f"Manifest must contain exactly {expected_count} rows, found {len(rows)}")
    return rows


def _reset_scene_articulation(env, env_id: int, row: ScenarioRow) -> None:
    import torch

    scene = env.scene["scene"]
    env_ids = torch.as_tensor([int(env_id)], device=env.device, dtype=torch.long)
    shared.reset_asset_pose_env(env, "scene", env_id, row.scene)
    joint_pos = scene.data.joint_pos[int(env_id) : int(env_id) + 1].clone()
    joint_vel = torch.zeros_like(scene.data.joint_vel[int(env_id) : int(env_id) + 1])
    joint_ids, _ = scene.find_joints("joint_1")
    joint_pos[:, joint_ids[0]] = float(row.door_joint_position)
    joint_vel[:, joint_ids[0]] = float(row.door_joint_velocity)
    scene.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    if not hasattr(env, "_task3_door_contact_latched"):
        env._task3_door_contact_latched = torch.zeros(
            scene.num_instances, device=scene.device, dtype=torch.bool
        )
    env._task3_door_contact_latched[env_ids] = bool(row.door_contact_latched)
    if hasattr(env, "_task3_door_close_step"):
        env._task3_door_close_step[env_ids] = -1
    if hasattr(env, "_task3_door_was_touching"):
        env._task3_door_was_touching[env_ids] = False
    if not hasattr(env, "_task3_eval_scene_se2"):
        env._task3_eval_scene_se2 = {}
    env._task3_eval_scene_se2[int(env_id)] = row.scene_se2


def reset_env_slot_to_manifest(env, env_id: int, row: ScenarioRow) -> None:
    import torch

    env_ids = torch.as_tensor([int(env_id)], device=env.device, dtype=torch.long)
    if hasattr(env, "_reset_idx"):
        env._reset_idx(env_ids=env_ids)
    else:
        env.reset()
    robot = env.scene["robot"]
    shared.reset_robot_root_env(
        env, robot, env_id, row.robot, np.asarray(row.robot_root_velocity, dtype=np.float32)
    )
    if not row.robot_body_joint_names:
        raise ValueError(f"Task3 scenario {row.scenario_id} is missing its per-source body joint state")
    if len(row.robot_body_joint_names) != len(row.robot_body_joint_position):
        raise ValueError(f"Task3 scenario {row.scenario_id} has mismatched body joint names/positions")
    if len(row.robot_body_joint_names) != len(row.robot_body_joint_velocity):
        raise ValueError(f"Task3 scenario {row.scenario_id} has mismatched body joint names/velocities")
    sim_joint_index = {name: index for index, name in enumerate(robot.data.joint_names)}
    missing = [name for name in row.robot_body_joint_names if name not in sim_joint_index]
    if missing:
        raise KeyError(f"Task3 scenario {row.scenario_id} body joints missing in simulation: {missing}")
    body_ids = torch.as_tensor(
        [sim_joint_index[name] for name in row.robot_body_joint_names],
        device=env.device,
        dtype=torch.long,
    )
    joint_pos = robot.data.joint_pos[int(env_id) : int(env_id) + 1].clone()
    joint_vel = robot.data.joint_vel[int(env_id) : int(env_id) + 1].clone()
    joint_pos[:, body_ids] = torch.as_tensor(
        row.robot_body_joint_position, device=env.device, dtype=joint_pos.dtype
    ).view(1, -1)
    joint_vel[:, body_ids] = torch.as_tensor(
        row.robot_body_joint_velocity, device=env.device, dtype=joint_vel.dtype
    ).view(1, -1)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    shared.reset_asset_pose_env(
        env, "object", env_id, row.object, np.asarray(row.object_root_velocity, dtype=np.float32)
    )
    shared.reset_asset_pose_env(
        env,
        "placement",
        env_id,
        row.placement,
        np.asarray(row.placement_root_velocity, dtype=np.float32),
    )
    object_view = env.scene["object"].root_physx_view
    if not hasattr(env, "_task3_eval_object_material_baseline"):
        env._task3_eval_object_material_baseline = (
            object_view.get_material_properties().clone()
        )
    else:
        object_view.set_material_properties(
            env._task3_eval_object_material_baseline,
            torch.as_tensor([int(env_id)], dtype=torch.long, device="cpu"),
        )
    _reset_scene_articulation(env, env_id, row)
    env.scene.write_data_to_sim()
    if hasattr(env, "sim") and hasattr(env.sim, "forward"):
        env.sim.forward()
    env.scene.update(dt=env.physics_dt)


def generate_manifest_with_env(args, env) -> list[ScenarioRow]:
    import torch

    bank = []
    with TASK3_SCENARIO_BANK.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            required = (
                "source_index",
                "source_demo",
                "robot_pose_wxyz",
                "object_pose_wxyz",
                "placement_pose_wxyz",
                "scene_translation_xy",
                "scene_delta_yaw",
                "robot_root_velocity",
                "robot_body_joint_names",
                "robot_body_joint_position",
                "robot_body_joint_velocity",
                "robot_hand_open",
                "object_root_velocity",
                "placement_root_velocity",
            )
            missing = [name for name in required if name not in payload]
            if missing:
                raise ValueError(f"Task3 scenario bank line {line_no} missing {missing}")
            bank.append(payload)
    if int(args.manifest_count) > len(bank):
        raise ValueError(
            f"Task3 manifest_count={args.manifest_count} exceeds the {len(bank)} correlated training scenarios"
        )
    selection = np.random.default_rng(int(args.seed)).permutation(len(bank))[
        : int(args.manifest_count)
    ]
    rows: list[ScenarioRow] = []
    for start in range(0, int(args.manifest_count), int(args.num_envs)):
        wave = list(range(start, min(int(args.manifest_count), start + int(args.num_envs))))
        env_ids = torch.as_tensor(range(len(wave)), device=env.device, dtype=torch.long)
        if hasattr(env, "_reset_idx"):
            env._reset_idx(env_ids=env_ids)
        for local_env_id, row_index in enumerate(wave):
            source = bank[int(selection[row_index])]
            robot_pose = np.asarray(source["robot_pose_wxyz"], dtype=np.float64)
            object_pose = np.asarray(source["object_pose_wxyz"], dtype=np.float64)
            placement_pose = np.asarray(source["placement_pose_wxyz"], dtype=np.float64)
            scene_translation = np.asarray(source["scene_translation_xy"], dtype=np.float64)
            scene_delta_yaw = float(source["scene_delta_yaw"])
            scene_se2 = shared.SceneSE2.from_arrays(scene_translation, scene_delta_yaw)
            row = ScenarioRow(
                schema_version=3,
                row_index=row_index,
                scenario_id=f"task3_carry_basket_{row_index:05d}",
                seed=int(args.seed) + row_index,
                robot=shared.Pose.from_arrays(robot_pose[:3], robot_pose[3:7]),
                object=shared.Pose.from_arrays(object_pose[:3], object_pose[3:7]),
                placement=shared.Pose.from_arrays(placement_pose[:3], placement_pose[3:7]),
                scene=shared.Pose.from_arrays(
                    (
                        SCENE_BASE_POS[0] + scene_translation[0],
                        SCENE_BASE_POS[1] + scene_translation[1],
                        SCENE_BASE_POS[2],
                    ),
                    SCENE_BASE_ROT_WXYZ,
                ),
                source_scenario_index=int(source["source_index"]),
                source_demo=str(source["source_demo"]),
                robot_root_velocity=tuple(float(value) for value in source["robot_root_velocity"]),
                robot_body_joint_names=tuple(str(value) for value in source["robot_body_joint_names"]),
                robot_body_joint_position=tuple(float(value) for value in source["robot_body_joint_position"]),
                robot_body_joint_velocity=tuple(float(value) for value in source["robot_body_joint_velocity"]),
                robot_hand_open=tuple(float(value) for value in source["robot_hand_open"]),
                object_root_velocity=tuple(float(value) for value in source["object_root_velocity"]),
                placement_root_velocity=tuple(float(value) for value in source["placement_root_velocity"]),
                scene_se2=scene_se2,
            ).with_hash()
            reset_env_slot_to_manifest(env, local_env_id, row)
            rows.append(row)
    return rows


def update_success_state(runtime: dict[str, Any], slot: shared.EnvSlot) -> dict[str, Any]:
    env = runtime["env"]
    for asset_name in runtime["success_state_assets"]:
        state = env.scene[asset_name].data.root_state_w[int(slot.env_id)].detach().cpu().numpy()
        history = slot.success_state_history.setdefault(asset_name, {"root_pose": [], "root_velocity": []})
        history["root_pose"].append(np.asarray(state[:7], dtype=np.float64).copy())
        history["root_velocity"].append(np.asarray(state[7:13], dtype=np.float64).copy())
    door = env.scene["scene"]
    joint_ids, _ = door.find_joints("joint_1")
    joint_id = joint_ids[0]
    door_history = slot.success_state_history.setdefault(
        "__door__", {"joint_position": [], "joint_velocity": [], "contact_latched": []}
    )
    door_history["joint_position"].append(float(door.data.joint_pos[slot.env_id, joint_id].item()))
    door_history["joint_velocity"].append(float(door.data.joint_vel[slot.env_id, joint_id].item()))
    latched = getattr(env, "_task3_door_contact_latched", None)
    door_history["contact_latched"].append(bool(latched[slot.env_id].item()) if latched is not None else False)

    payload: dict[str, np.ndarray] = {}
    for asset_name in runtime["success_state_assets"]:
        streams = slot.success_state_history[asset_name]
        payload[f"{asset_name}_root_pose"] = np.stack(streams["root_pose"], axis=0)
        payload[f"{asset_name}_root_velocity"] = np.stack(streams["root_velocity"], axis=0)
    payload["door_joint_position"] = np.asarray(door_history["joint_position"], dtype=np.float64)
    payload["door_joint_velocity"] = np.asarray(door_history["joint_velocity"], dtype=np.float64)
    payload["door_contact_latched"] = np.asarray(door_history["contact_latched"], dtype=bool)
    result = runtime["replay_success_evaluator"](payload, runtime["replay_success_cfg"])
    if not isinstance(result, dict) or "ok" not in result:
        raise TypeError("Task3 replay_success evaluator must return a dict containing 'ok'")
    return result


class Task3DynamicDoorRenderer:
    """Add the native-hinge door GS to the standard Isaac-mesh compositor."""

    def __init__(self, args, env):
        from wbc_play_groot_sonic import Online2DGSRenderer, get_head_camera

        task3_dir = shared.REPO_ROOT / "scripts/sonic_kimodo_run/task3_carry_basket"
        if str(task3_dir) not in sys.path:
            sys.path.insert(0, str(task3_dir))
        from dynamic_door_gs import (
            DEFAULT_GS_WORLD_YAW_CORRECTION_DEG,
            DoorHingeTransformCache,
            authored_gs_to_world_matrix,
            load_videoartgs_hinge_axis,
            se2_matrix,
            transform_axis,
            yaw_about_world_point_matrix,
        )
        from render_dynamic_door_gs_background import CombinedDynamicDoorModel

        common = dict(
            camera=get_head_camera(env),
            real2sim_root=args.two_d_gs_real2sim_root,
            semantic_tolerance=args.gs_semantic_tolerance,
            foreground_mode="isaac_mesh",
            apply_training_color_match=False,
        )
        self.base = Online2DGSRenderer(args.gs_scene_path, **common)
        door_loader = Online2DGSRenderer(TASK3_DOOR_PLY, **common)
        self.dynamic = CombinedDynamicDoorModel(self.base.renderer, door_loader.renderer)
        self._scene_xyz0 = self.base._scene_base_xyz.detach().clone()
        raw_origin, raw_direction = load_videoartgs_hinge_axis(TASK3_JOINT_INFO)
        baked_world = yaw_about_world_point_matrix(DEFAULT_GS_WORLD_YAW_CORRECTION_DEG)
        axis_origin, axis_direction = transform_axis(
            raw_origin,
            raw_direction,
            baked_world @ authored_gs_to_world_matrix(),
        )
        self.hinge_cache = DoorHingeTransformCache(0.0, axis_origin, axis_direction)
        self._se2_matrix = se2_matrix
        self._last_scene_key: tuple[float, float, float] | None = None
        self._last_hinge: float | None = None
        self.color_matcher = _TASK3_COLOR_MATCHER
        if (
            os.environ.get("TASK3_APPLY_CARRYBOX_COLOR_MATCH", "1") != "0"
            and self.color_matcher is None
        ):
            raise RuntimeError("Task3 color matcher was not prepared before Isaac startup")
        print(
            "[GS] Task3 carrybox-real-v1 whole-frame color match="
            f"{self.color_matcher is not None}",
            flush=True,
        )

    def render_env(self, *, env, env_id: int, robot, sim_joint_names: list[str]) -> np.ndarray:
        row_se2 = getattr(env, "_task3_eval_scene_se2", {}).get(
            int(env_id), shared.SceneSE2.identity()
        )
        tx, ty = (float(value) for value in row_se2.translation_xy)
        yaw = float(row_se2.delta_yaw)
        if abs(yaw) > 1.0e-9:
            raise RuntimeError("Task3 online renderer currently requires scene delta_yaw=0")
        scene_key = (tx, ty, yaw)
        door = env.scene["scene"]
        joint_ids, _ = door.find_joints("joint_1")
        hinge = float(door.data.joint_pos[int(env_id), joint_ids[0]].item())
        if scene_key != self._last_scene_key or self._last_hinge is None or abs(hinge - self._last_hinge) > 1.0e-8:
            torch = self.base.zbuf.torch
            with torch.no_grad():
                translation = torch.as_tensor(
                    (tx, ty, 0.0), device=self._scene_xyz0.device, dtype=self._scene_xyz0.dtype
                )
                self.dynamic.model._xyz[: self.dynamic.scene_count].copy_(self._scene_xyz0 + translation)
            hinge_transform, _ = self.hinge_cache.transform(hinge)
            self.dynamic.set_relative_world_transform(
                self._se2_matrix((tx, ty), yaw) @ hinge_transform
            )
            self._last_scene_key = scene_key
            self._last_hinge = hinge
        rgb = self.base.render_env(
            env=env, env_id=env_id, robot=robot, sim_joint_names=sim_joint_names
        )
        return self.color_matcher.apply(rgb) if self.color_matcher is not None else rgb


def init_gs_renderer(args, env):
    if not args.enable_gs_rollout:
        return None
    return Task3DynamicDoorRenderer(args, env)


_shared_mirror_foreground = shared._mirror_env_into_foreground_sandbox


def load_processor_modality_configs(checkpoint_dir: Path, embodiment: str) -> dict[str, Any]:
    """Support final-only checkpoints that keep processor files in /processor."""

    root = Path(checkpoint_dir).expanduser().resolve()
    config_path = root / "processor_config.json"
    if not config_path.is_file():
        config_path = root / "processor/processor_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload["processor_kwargs"]["modality_configs"][embodiment]


def mirror_env_into_foreground_sandbox(env, target_env_id: int, sandbox_env_id: int) -> None:
    _shared_mirror_foreground(env, target_env_id, sandbox_env_id)
    if int(target_env_id) == int(sandbox_env_id):
        return
    # Task3's supported launcher uses one environment. Refuse silent dynamic
    # door desynchronization if a future caller enables the mesh sandbox.
    raise RuntimeError("Task3 dynamic-door evaluation currently supports NUM_ENVS=1")


def validate_required_assets(args) -> None:
    required = (
        args.checkpoint_dir,
        args.gs_scene_path,
        TASK3_DOOR_PLY,
        TASK3_JOINT_INFO,
        TASK3_SCENARIO_BANK,
        TASK3_INITIAL_BODY_POSE,
        args.headcam_calib,
        args.models_dir,
        TASK3_ASSETS_ROOT / "usd/robots/final_robot_usd_align_sonic_portable.usd",
        TASK3_SCENE_USD_PATH,
        TASK3_ASSETS_ROOT / "usd/assets/B1/task3_lanzi_Model_1_scaled_centered.usd",
        TASK3_ASSETS_ROOT / "usd/assets/B1/dengzi_Model_2_scaled_centered.usd",
    )
    missing = [str(Path(path).expanduser()) for path in required if not Path(path).expanduser().exists()]
    if missing:
        raise FileNotFoundError("Mandatory Task3 evaluation assets are missing: " + ", ".join(missing))
    print(
        f"[GS] Task3 dynamic scene={args.gs_scene_path} door={TASK3_DOOR_PLY} "
        "foreground=isaac_mesh",
        flush=True,
    )


def write_metadata(args, run_dir, manifest_digest, chosen_envs, fallback_reason, modality_configs) -> None:
    shared.write_metadata(
        args, run_dir, manifest_digest, chosen_envs, fallback_reason, modality_configs
    )
    metadata_path = Path(run_dir) / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    usd_root = TASK3_ASSETS_ROOT / "usd"
    assets = {
        "robot_usd": usd_root / "robots/final_robot_usd_align_sonic_portable.usd",
        "scene_usd": TASK3_SCENE_USD_PATH,
        "basket_usd": usd_root / "assets/B1/task3_lanzi_Model_1_scaled_centered.usd",
        "stool_usd": usd_root / "assets/B1/dengzi_Model_2_scaled_centered.usd",
        "scene_gs": Path(args.gs_scene_path),
        "door_gs": TASK3_DOOR_PLY,
        "door_joint_info": TASK3_JOINT_INFO,
        "training_scenario_bank": TASK3_SCENARIO_BANK,
        "initial_body_pose": TASK3_INITIAL_BODY_POSE,
    }
    payload["task3_assets"] = {name: str(path) for name, path in assets.items()}
    payload["task3_asset_hashes"] = {
        name: shared.sha256_file(path) for name, path in assets.items()
    }
    payload["task3_render_contract"] = {
        "foreground": ["isaac_mesh_robot", "isaac_mesh_basket", "isaac_mesh_stool"],
        "background": "dynamic_B1_art_scene_and_hinged_door_2dgs",
        "depth_or_zbuffer_foreground_test": False,
        "training_color_match": os.environ.get("TASK3_APPLY_CARRYBOX_COLOR_MATCH", "1") != "0",
        "training_color_profile": "carrybox-real-v1",
    }
    shared.write_json(metadata_path, payload)


def diagnostic_video_required(row_index: int) -> bool:
    return shared.diagnostic_video_required(row_index)

def _apply_task3_post_lift_friction(runtime: dict[str, Any], slot: shared.EnvSlot) -> None:
    """Reproduce the online basket-friction contract used to generate Task3."""

    import torch

    env = runtime["env"]
    object_asset = env.scene["object"]
    baseline = getattr(env, "_task3_eval_object_material_baseline", None)
    if baseline is None or slot.initial_object_z is None or slot.chunk is None:
        return
    state = slot.success_state_history.setdefault(
        "__task3_object_friction__",
        {"active": False, "restored": False, "seen_closed": False},
    )
    chunk_index = min(int(slot.chunk_pos), int(slot.chunk.left_hand_gripper.shape[1]) - 1)
    left_open = float(slot.chunk.left_hand_gripper[0, chunk_index, 0])
    right_open = float(slot.chunk.right_hand_gripper[0, chunk_index, 0])
    if left_open <= 0.2 and right_open <= 0.2:
        state["seen_closed"] = True

    env_id = int(slot.env_id)
    env_ids_cpu = torch.as_tensor([env_id], dtype=torch.long, device="cpu")
    if not bool(state["active"]):
        object_z = float(object_asset.data.root_state_w[env_id, 2].detach().cpu().item())
        lift_m = object_z - float(slot.initial_object_z)
        if lift_m >= 0.10:
            materials = object_asset.root_physx_view.get_material_properties().clone()
            materials[env_id, :, 0] = 2.0
            materials[env_id, :, 1] = 2.0
            object_asset.root_physx_view.set_material_properties(materials, env_ids_cpu)
            state["active"] = True
            state["trigger_step"] = int(slot.steps)
            state["trigger_lift_m"] = float(lift_m)
            print(
                f"[TASK3_FRICTION] row={slot.scenario.row_index} activated "
                f"step={slot.steps} lift_m={lift_m:.3f} static=2.0 dynamic=2.0",
                flush=True,
            )
    elif (
        not bool(state["restored"])
        and bool(state["seen_closed"])
        and left_open >= 0.8
        and right_open >= 0.8
    ):
        object_asset.root_physx_view.set_material_properties(baseline, env_ids_cpu)
        state["restored"] = True
        state["restore_step"] = int(slot.steps)
        print(
            f"[TASK3_FRICTION] row={slot.scenario.row_index} restored at model release "
            f"step={slot.steps}",
            flush=True,
        )


def _install_task3_friction_override():
    """Return the Task3 friction callback for the shared runner lifecycle."""

    def before_step(args, runtime, slots) -> None:
        del args
        for slot in slots:
            if slot.active:
                _apply_task3_post_lift_friction(runtime, slot)

    return before_step


def _install_task3_actuator_profile_override():
    """Return the Task3 actuator callback without changing process globals."""

    def apply_task3_profile(robot_cfg, **kwargs) -> None:
        import run_kimodo_sonic_isaaclab as sonic_runtime

        sonic_runtime.apply_sonic_actuator_profile(robot_cfg, **kwargs)
        applied = sonic_runtime.apply_task3_replay_waist_override(
            robot_cfg, TASK3_RENDER_TASK_ID
        )
        print(f"[TASK3] matched generation waist stiffness: {applied}", flush=True)

    return apply_task3_profile


def _install_task3_chunk_diagnostics():
    """Return the Task3 predicted-grasp diagnostic callback."""

    def after_chunk(args, runtime, slot) -> None:
        del runtime
        horizon = min(int(args.chunk_horizon), int(slot.chunk.left_hand_gripper.shape[1]))
        left = np.asarray(slot.chunk.left_hand_gripper[0, :horizon, 0], dtype=np.float32)
        right = np.asarray(slot.chunk.right_hand_gripper[0, :horizon, 0], dtype=np.float32)
        print(
            "[TASK3_GRIPPER] "
            f"row={slot.scenario.row_index} step={slot.steps} horizon={horizon} "
            f"left=[{left.min():.3f},{left.max():.3f}] "
            f"right=[{right.min():.3f},{right.max():.3f}]",
            flush=True,
        )

    return after_chunk


def make_task_adapter() -> EvaluationTaskAdapter:
    """Return Task3 overrides and lifecycle hooks without sibling mutation."""

    return replace(
        shared.make_default_adapter(),
        name="Task3 carry basket",
        task_id=TASK3_RENDER_TASK_ID,
        manifest_row_count=TASK3_SCENARIO_COUNT,
        output_root=shared.REPO_ROOT.parent / "outputs/task3",
        default_chunk_horizon=40,
        default_init_joint_state_json=TASK3_INITIAL_BODY_POSE,
        default_init_joint_state_dataset=None,
        default_manifest=shared.REPO_ROOT.parent / "configs/manifests/task3_hdf5_success386_perturbed100_seed20260831.jsonl",
        read_manifest=read_manifest,
        reset_env_slot_to_manifest=reset_env_slot_to_manifest,
        generate_manifest_with_env=generate_manifest_with_env,
        reset_slot=shared.reset_slot,
        update_success_state=update_success_state,
        init_gs_renderer=init_gs_renderer,
        mirror_env_into_foreground_sandbox=mirror_env_into_foreground_sandbox,
        validate_required_assets=validate_required_assets,
        write_metadata=write_metadata,
        diagnostic_video_required=diagnostic_video_required,
        load_processor_modality_configs=load_processor_modality_configs,
        before_step=_install_task3_friction_override(),
        after_chunk=_install_task3_chunk_diagnostics(),
        apply_actuator_profile=_install_task3_actuator_profile_override(),
    )


if __name__ == "__main__":
    _prepare_task3_color_matcher()
    adapter = make_task_adapter()
    run_evaluation(shared.parse_args(adapter), adapter)
