#!/usr/bin/env python3
"""Task2 scene, reset, rendering, and success evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

import evaluation_runtime as shared
from evaluation_runner import EvaluationTaskAdapter, run_evaluation


TASK2_SCENARIO_COUNT = 100
FINAL_TASK_2_PICK_UP_HAT_RENDER_TASK_ID = "Isaac-Final-Task2-PickUpHat-Render-Abs-v0"
TASK2_CABINET_FIXED_POS = (0.85, 1.32666, 0.0)
TASK2_CABINET_FIXED_ROT_WXYZ = (0.9515756268, 0.0, 0.0, -0.3074147468)
TASK2_ASSETS_ROOT = Path(
    os.environ.get(
        "TASK2_EVAL_BUNDLE_ROOT",
        str(shared.REPO_ROOT.parent / "assets/task2"),
    )
).expanduser()


@dataclass(frozen=True)
class ScenarioRow:
    schema_version: int
    row_index: int
    scenario_id: str
    seed: int
    robot: shared.Pose
    object: shared.Pose
    placement: shared.Pose
    hanger: shared.Pose
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
            "hanger": asdict(self.hanger),
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
            hanger=self.hanger,
            scene_se2=self.scene_se2,
            row_hash=digest,
        )


def scenario_from_json(payload: dict[str, Any]) -> ScenarioRow:
    stored_hash = str(payload.get("row_hash", ""))
    raw_payload = {key: value for key, value in payload.items() if key != "row_hash"}
    expected_hash = hashlib.sha256(shared.canonical_json(raw_payload).encode("utf-8")).hexdigest()
    if stored_hash and stored_hash != expected_hash:
        raise ValueError(f"Manifest row {payload.get('row_index')} hash mismatch")

    def pose(name: str) -> shared.Pose:
        value = payload[name]
        return shared.Pose.from_arrays(value["pos"], value["quat_wxyz"])

    row = ScenarioRow(
        schema_version=int(payload["schema_version"]),
        row_index=int(payload["row_index"]),
        scenario_id=str(payload["scenario_id"]),
        seed=int(payload["seed"]),
        robot=pose("robot"),
        object=pose("object"),
        placement=pose("placement"),
        hanger=pose("hanger"),
        scene_se2=shared.SceneSE2.from_arrays(
            payload.get("scene_se2", {}).get("translation_xy", [0.0, 0.0]),
            payload.get("scene_se2", {}).get("delta_yaw", 0.0),
        ),
        row_hash=stored_hash,
    )
    return row if row.row_hash else row.with_hash()


def read_manifest(path: Path, expected_count: int | None = TASK2_SCENARIO_COUNT) -> list[ScenarioRow]:
    rows: list[ScenarioRow] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = scenario_from_json(json.loads(line))
            if row.row_index != len(rows) or row.scenario_id != f"task2_cabinet_{row.row_index:05d}":
                raise ValueError(f"Task2 manifest order/id mismatch at line {line_no}")
            rows.append(row)
    if expected_count is not None and len(rows) != int(expected_count):
        raise ValueError(f"Manifest must contain exactly {expected_count} rows, found {len(rows)}")
    return rows


def reset_env_slot_to_manifest(env, env_id: int, row: ScenarioRow) -> None:
    import torch

    env_ids = torch.as_tensor([int(env_id)], device=env.device, dtype=torch.long)
    if hasattr(env, "_reset_idx"):
        env._reset_idx(env_ids=env_ids)
    else:
        env.reset()
    shared.reset_robot_root_env(env, env.scene["robot"], env_id, row.robot)
    for asset_name in ("object", "placement", "hanger"):
        shared.reset_asset_pose_env(env, asset_name, env_id, getattr(row, asset_name))
    if hasattr(env, "sim") and hasattr(env.sim, "forward"):
        env.sim.forward()
    env.scene.update(dt=env.physics_dt)


def generate_manifest_with_env(args, env) -> list[ScenarioRow]:
    import torch

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    rows: list[ScenarioRow] = []
    count = int(args.manifest_count)
    cabinet_pose = shared.Pose.from_arrays(TASK2_CABINET_FIXED_POS, TASK2_CABINET_FIXED_ROT_WXYZ)
    for start in range(0, count, int(args.num_envs)):
        wave = list(range(start, min(count, start + int(args.num_envs))))
        env_ids = torch.as_tensor(range(len(wave)), device=env.device, dtype=torch.long)
        if hasattr(env, "_reset_idx"):
            env._reset_idx(env_ids=env_ids)
        else:
            env.reset()
        for local_env_id, row_index in enumerate(wave):
            root_pos, root_quat = shared.sample_robot_pose(args, row_index)
            shared.reset_robot_root_env(
                env,
                env.scene["robot"],
                local_env_id,
                shared.Pose.from_arrays(root_pos, root_quat),
            )
            shared.reset_asset_pose_env(env, "hanger", local_env_id, cabinet_pose)
        if hasattr(env, "sim") and hasattr(env.sim, "forward"):
            env.sim.forward()
        env.scene.update(dt=env.physics_dt)
        for local_env_id, row_index in enumerate(wave):
            rows.append(
                ScenarioRow(
                    schema_version=2,
                    row_index=row_index,
                    scenario_id=f"task2_cabinet_{row_index:05d}",
                    seed=int(args.seed) + row_index,
                    robot=shared.robot_pose(env, local_env_id),
                    object=shared.asset_pose(env, "object", local_env_id),
                    placement=shared.asset_pose(env, "placement", local_env_id),
                    hanger=cabinet_pose,
                ).with_hash()
            )
    return rows


_CURVES = (
    ((0.0, 0.071, 0.098, 0.169, 0.616, 0.737, 1.0), (0.0, 0.106, 0.149, 0.200, 0.682, 0.820, 1.0)),
    ((0.0, 0.098, 0.137, 0.176, 0.655, 0.765, 1.0), (0.0, 0.129, 0.173, 0.196, 0.678, 0.820, 1.0)),
    ((0.0, 0.082, 0.149, 0.173, 0.690, 0.780, 1.0), (0.0, 0.098, 0.157, 0.173, 0.671, 0.831, 1.0)),
)


def _match_training_background_color(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    output = np.empty_like(image)
    for channel, (points_in, points_out) in enumerate(_CURVES):
        output[..., channel] = np.interp(image[..., channel], points_in, points_out)
    return np.clip(np.rint(output * 255.0), 0, 255).astype(np.uint8)


_shared_init_gs_renderer = shared.init_gs_renderer
_shared_mirror_foreground = shared._mirror_env_into_foreground_sandbox


def init_gs_renderer(args, env):
    renderer = _shared_init_gs_renderer(args, env)
    if renderer is not None:
        renderer.background_rgb_transform = _match_training_background_color
    return renderer


def mirror_env_into_foreground_sandbox(env, target_env_id: int, sandbox_env_id: int) -> None:
    """Extend the Task1 mesh sandbox copy with the cabinet state."""

    import torch

    _shared_mirror_foreground(env, target_env_id, sandbox_env_id)
    if int(target_env_id) == int(sandbox_env_id):
        return
    env_ids = torch.as_tensor([int(sandbox_env_id)], device=env.device, dtype=torch.long)
    source_origin = env.scene.env_origins[int(target_env_id)]
    sandbox_origin = env.scene.env_origins[int(sandbox_env_id)]
    cabinet = env.scene["hanger"]
    state = cabinet.data.root_state_w[int(target_env_id) : int(target_env_id) + 1].clone()
    state[:, :3] += (sandbox_origin - source_origin).view(1, 3)
    cabinet.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    cabinet.write_root_velocity_to_sim(state[:, 7:13], env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)


def write_metadata(args, run_dir, manifest_digest, chosen_envs, fallback_reason, modality_configs) -> None:
    shared.write_metadata(
        args,
        run_dir,
        manifest_digest,
        chosen_envs,
        fallback_reason,
        modality_configs,
    )
    metadata_path = Path(run_dir) / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    from pick_place.final_task_2_pick_up_hat_env_cfg import (
        HANGER_USD_PATH,
        OBJECT_USD_PATH,
        PLACEMENT_USD_PATH,
        ROBOT_USD_PATH,
        SCENE_USD_PATH,
    )

    task2_assets = {
        "robot_usd": ROBOT_USD_PATH,
        "scene_usd": SCENE_USD_PATH,
        "hat_usd": OBJECT_USD_PATH,
        "little_desk_usd": PLACEMENT_USD_PATH,
        "cabinet_usd": HANGER_USD_PATH,
    }
    payload["task2_assets"] = {name: str(path) for name, path in task2_assets.items()}
    payload["task2_asset_hashes"] = {
        name: shared.sha256_file(path) for name, path in task2_assets.items()
    }
    payload["task2_render_contract"] = {
        "foreground": "isaac_mesh_robot_and_rigid_assets",
        "background": "B1_0711_2dgs",
        "background_color_profile": "task2_training",
        "robot_gs": False,
        "depth_or_zbuffer_foreground_test": False,
    }
    shared.write_json(metadata_path, payload)


def diagnostic_video_required(row_index: int) -> bool:
    return shared.diagnostic_video_required(row_index)

def make_task_adapter() -> EvaluationTaskAdapter:
    """Return Task2 overrides without mutating the Task1 module."""

    return replace(
        shared.make_default_adapter(),
        name="Task2 cabinet",
        task_id=FINAL_TASK_2_PICK_UP_HAT_RENDER_TASK_ID,
        manifest_row_count=TASK2_SCENARIO_COUNT,
        output_root=shared.REPO_ROOT.parent / "outputs/task2",
        default_chunk_horizon=40,
        default_init_joint_state_json=TASK2_ASSETS_ROOT / "assets/evaluation/task2_initial_body_pose_episode_000000.json",
        default_init_joint_state_dataset=(
            Path(os.environ["TASK2_INIT_JOINT_DATASET"]).expanduser()
            if os.environ.get("TASK2_INIT_JOINT_DATASET")
            else None
        ),
        default_manifest=shared.REPO_ROOT.parent / "configs/manifests/task2_cabinet_xy15cm_hat_xy5cm_100.jsonl",
        read_manifest=read_manifest,
        reset_env_slot_to_manifest=reset_env_slot_to_manifest,
        generate_manifest_with_env=generate_manifest_with_env,
        reset_slot=shared.reset_slot,
        init_gs_renderer=init_gs_renderer,
        mirror_env_into_foreground_sandbox=mirror_env_into_foreground_sandbox,
        write_metadata=write_metadata,
        diagnostic_video_required=diagnostic_video_required,
    )


if __name__ == "__main__":
    adapter = make_task_adapter()
    run_evaluation(shared.parse_args(adapter), adapter)
