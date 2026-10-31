"""Shared closed-loop runner for the paired GR00T + SONIC task evaluators.

Task modules own scene data and task-specific physics/rendering details.  This
module owns the evaluation lifecycle and receives those details through an
explicit adapter.  Keeping the runner independent of a task module prevents a
Task2/Task3 import from changing Task1's module globals for the duration of a
process.
"""

from __future__ import annotations

import argparse
import contextlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from zerohlm_bench.protocol import validate_policy_modalities
except ModuleNotFoundError:  # Keep CPU-only AST/runtime contracts importable in-place.
    def validate_policy_modalities(expected, received):
        from zerohlm_bench.protocol import validate_policy_modalities as validator

        return validator(expected, received)


Callback = Callable[..., Any]


@dataclass(frozen=True)
class EvaluationTaskAdapter:
    """Task-owned callbacks and release defaults consumed by :func:`run_evaluation`.

    The callbacks intentionally cover the seams where tasks differ: manifest
    rows and reset state, environment setup, GS composition, success history,
    metadata, and per-step lifecycle hooks.  The rollout queue and fallback
    policy stay in this module so those behaviors cannot drift between tasks.
    """

    name: str
    task_id: str
    manifest_row_count: int
    output_root: Path
    default_init_joint_state_json: Path | None
    default_init_joint_state_dataset: Path | None
    default_manifest: Path | None
    default_chunk_horizon: int
    parse_args: Callback
    validate_required_assets: Callback
    load_processor_modality_configs: Callback
    read_manifest: Callback
    write_manifest: Callback
    manifest_hash: Callback
    select_manifest_rows: Callback
    pending_rows: Callback
    reset_env_slot_to_manifest: Callback
    setup_runtime: Callback
    generate_manifest_with_env: Callback
    make_policies: Callback
    init_gs_renderer: Callback
    reset_slot: Callback
    synchronize_reset_wave: Callback
    request_chunk: Callback
    step_active_slots: Callback
    capture_gs_frame_env: Callback
    mirror_env_into_foreground_sandbox: Callback | None
    finalize_slot: Callback
    append_episode_rows: Callback
    read_episode_csv: Callback
    write_tables: Callback
    write_aggregate: Callback
    write_metadata: Callback
    diagnostic_video_required: Callback
    env_slot_type: type
    robot_inference_client: Callback
    summarize_modality_config: Callback | None = None
    update_success_state: Callback | None = None
    aggregate_paired: Callback | None = None
    before_step: Callback | None = None
    after_chunk: Callback | None = None
    after_episode: Callback | None = None
    apply_actuator_profile: Callback | None = None
    configure_runtime: Callback | None = None
    refresh_eval_head_camera: Callback | None = None

    def with_defaults(self, args: argparse.Namespace) -> argparse.Namespace:
        """Apply task defaults only where the parser did not provide one."""

        if getattr(args, "task", None) is None:
            args.task = self.task_id
        if getattr(args, "manifest", None) is None and self.default_manifest is not None:
            args.manifest = self.default_manifest
        if getattr(args, "manifest_count", None) is None:
            args.manifest_count = int(self.manifest_row_count)
        if getattr(args, "output_root", None) is None:
            args.output_root = self.output_root
        if getattr(args, "chunk_horizon", None) is None:
            args.chunk_horizon = int(self.default_chunk_horizon)
        if getattr(args, "init_joint_state_json", None) is None:
            args.init_joint_state_json = self.default_init_joint_state_json
        if getattr(args, "init_joint_state_dataset", None) is None:
            args.init_joint_state_dataset = self.default_init_joint_state_dataset
        return args


def parse_args(adapter: EvaluationTaskAdapter) -> argparse.Namespace:
    """Parse a task's CLI through its adapter-owned parser callback."""

    args = adapter.parse_args(adapter)
    return adapter.with_defaults(args)


def _call_setup(adapter: EvaluationTaskAdapter, args: argparse.Namespace, count: int):
    """Call setup callbacks with the adapter when supported by the task code."""

    return adapter.setup_runtime(args, int(count), adapter)


def _call_reset(adapter: EvaluationTaskAdapter, args, runtime, slot, row, policy, run_dir):
    return adapter.reset_slot(args, runtime, slot, row, policy, run_dir, adapter)


def _call_finalize(adapter: EvaluationTaskAdapter, run_dir, slot, model_name, checkpoint_dir, digest):
    return adapter.finalize_slot(run_dir, slot, model_name, checkpoint_dir, digest, adapter)


def run_checkpoint(
    args: argparse.Namespace,
    runtime: dict[str, Any],
    rows: list[Any],
    manifest_digest: str,
    chosen_envs: int,
    fallback_reason: str | None,
    adapter: EvaluationTaskAdapter,
) -> None:
    """Run all pending rows for one simulator allocation.

    Protocol validation intentionally precedes renderer construction.  The
    policy server and checkpoint schema must agree before Isaac allocates GS
    resources or starts the closed-loop rollout.
    """

    run_dir = Path(args.output_root) / str(args.model_name) / "sim_eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    episode_csv = run_dir / "episodes.csv"
    remaining = adapter.pending_rows(rows, episode_csv, args.resume)
    if not remaining:
        print("[RUN] no pending rows; refreshing aggregate and metadata", flush=True)
        all_rows = adapter.read_episode_csv(episode_csv)
        adapter.write_tables(run_dir, all_rows)
        adapter.write_aggregate(run_dir, all_rows)
        return

    modality_configs = adapter.load_processor_modality_configs(
        args.checkpoint_dir, args.embodiment
    )
    client = adapter.robot_inference_client(
        args.gr00t_host, args.gr00t_port, args.gr00t_timeout_ms
    )
    gs_render = None
    slots = []
    try:
        print(f"[GR00T] ping={client.ping()}", flush=True)
        client.reset()
        server_cfg = client.get_modality_config()
        validate_policy_modalities(modality_configs, server_cfg)
        if adapter.configure_runtime is not None:
            adapter.configure_runtime(args, runtime, modality_configs)
        print("[SCHEMA] policy modalities validated", flush=True)
        gs_render = adapter.init_gs_renderer(args, runtime["env"])
        if gs_render is None:
            raise RuntimeError("Mandatory GS renderer did not initialize")
        runtime["gs_render"] = gs_render
        policies = adapter.make_policies(args, runtime, chosen_envs)
        sandbox_env_id = getattr(runtime["env"], "_task1_mesh_foreground_sandbox_env_id", None)
        active_env_ids = [
            env_id
            for env_id in range(int(chosen_envs))
            if sandbox_env_id is None or env_id != int(sandbox_env_id)
        ]
        slots = [adapter.env_slot_type(env_id=env_id) for env_id in active_env_ids]
        if sandbox_env_id is not None:
            print(
                f"[GS] reserved env={sandbox_env_id} as single-camera mesh foreground sandbox; "
                f"active_rollout_envs={active_env_ids}",
                flush=True,
            )
        queue = iter(remaining)
        completed_rows: list[dict[str, Any]] = []

        def fill_empty_slots() -> None:
            reset_any = False
            for slot in slots:
                if slot.scenario is None:
                    with contextlib.suppress(StopIteration):
                        _call_reset(
                            adapter,
                            args,
                            runtime,
                            slot,
                            next(queue),
                            policies[slot.env_id],
                            run_dir,
                        )
                        reset_any = True
            if reset_any:
                adapter.synchronize_reset_wave(args, runtime, slots)

        fill_empty_slots()
        while any(slot.scenario is not None for slot in slots):
            active = [slot for slot in slots if slot.active]
            for slot in sorted(
                active, key=lambda item: item.scenario.row_index if item.scenario else -1
            ):
                if slot.chunk is None or slot.chunk_pos >= int(args.chunk_horizon):
                    try:
                        adapter.request_chunk(
                            args,
                            runtime,
                            client,
                            modality_configs,
                            gs_render,
                            slot,
                            adapter,
                        )
                        if adapter.after_chunk is not None:
                            adapter.after_chunk(args, runtime, slot)
                    except BaseException as exc:
                        traceback.print_exc()
                        if _is_fallback_trigger(exc):
                            raise
                        _mark_slot_error(slot, exc)
            try:
                if adapter.before_step is not None:
                    adapter.before_step(args, runtime, slots)
                adapter.step_active_slots(args, runtime, slots, gs_render, adapter)
            except Exception as exc:
                if _is_fallback_trigger(exc):
                    raise
                for slot in slots:
                    if slot.active:
                        _mark_slot_error(slot, exc)
            finalized = []
            for slot in slots:
                if slot.scenario is not None and slot.terminal_reason is not None:
                    if slot.final_frame is None:
                        if adapter.refresh_eval_head_camera is not None:
                            adapter.refresh_eval_head_camera(args, runtime)
                        slot.final_frame = adapter.capture_gs_frame_env(
                            runtime["env"],
                            gs_render,
                            runtime["robot"],
                            runtime["sim_joint_names"],
                            slot.env_id,
                            adapter,
                        )
                    row = _call_finalize(
                        adapter,
                        run_dir,
                        slot,
                        args.model_name,
                        args.checkpoint_dir,
                        manifest_digest,
                    )
                    completed_rows.append(row)
                    finalized.append(slot)
                    if adapter.after_episode is not None:
                        adapter.after_episode(args, runtime, slot, row)
                    print(
                        f"[EP] {row['scenario_id']} env={slot.env_id} reason={row['terminal_reason']} "
                        f"steps={row['control_steps']} success={row['success']}",
                        flush=True,
                    )
                    slot.scenario = None
                    slot.policy = None
                    slot.chunk = None
                    slot.video = None
            if finalized:
                adapter.append_episode_rows(episode_csv, completed_rows)
                completed_rows = []
                fill_empty_slots()
            if active and min(slot.steps for slot in active) % max(1, int(args.log_every)) == 0:
                print(
                    f"[RUN] active={len(active)} pending_or_active="
                    f"{sum(s.scenario is not None for s in slots)}",
                    flush=True,
                )
        all_rows = adapter.read_episode_csv(episode_csv)
        adapter.write_tables(run_dir, all_rows)
        adapter.write_aggregate(run_dir, all_rows)
        adapter.write_metadata(
            args,
            run_dir,
            manifest_digest,
            chosen_envs,
            fallback_reason,
            modality_configs,
        )
    finally:
        for slot in slots:
            if slot.video is not None:
                slot.video.close()
        runtime.pop("gs_render", None)
        if gs_render is not None and hasattr(gs_render, "close"):
            with contextlib.suppress(Exception):
                gs_render.close()
        client.close()


def run_with_fallback(
    args: argparse.Namespace,
    rows: list[Any],
    manifest_digest: str,
    simulation_app: Any,
    adapter: EvaluationTaskAdapter,
) -> None:
    """Retry simulator construction with configured lower environment counts."""

    del simulation_app
    attempts = [int(args.num_envs)] + [
        int(value)
        for value in args.fallback_env_counts
        if int(value) < int(args.num_envs) and int(value) >= 1
    ]
    last_exc: BaseException | None = None
    for index, count in enumerate(attempts):
        runtime = None
        try:
            runtime = _call_setup(adapter, args, count)
            fallback_reason = (
                None
                if index == 0
                else f"previous_attempt_failed: {type(last_exc).__name__}: {last_exc}"
            )
            run_checkpoint(
                args,
                runtime,
                rows,
                manifest_digest,
                int(count),
                fallback_reason,
                adapter,
            )
            return
        except (RuntimeError, MemoryError) as exc:
            last_exc = exc
            if index + 1 >= len(attempts) or not _is_fallback_trigger(exc):
                raise
            print(
                f"[FALLBACK] num_envs={count} failed: {exc}; trying {attempts[index + 1]}",
                flush=True,
            )
        finally:
            if runtime is not None:
                with contextlib.suppress(Exception):
                    runtime["env"].close()
    if last_exc is not None:
        raise last_exc


def run_evaluation(
    args: argparse.Namespace,
    adapter: EvaluationTaskAdapter,
    simulation_app: Any | None = None,
) -> None:
    """Execute schema, manifest, simulator, and aggregate phases for a task."""

    args = adapter.with_defaults(args)
    if not Path(args.checkpoint_dir).expanduser().exists():
        raise FileNotFoundError(f"Checkpoint directory is missing: {args.checkpoint_dir}")
    checkpoint_configs = adapter.load_processor_modality_configs(
        args.checkpoint_dir, args.embodiment
    )
    if args.schema_only:
        summary = (
            adapter.summarize_modality_config(checkpoint_configs)
            if adapter.summarize_modality_config is not None
            else checkpoint_configs
        )
        print(summary, flush=True)
        return

    if args.aggregate_only:
        rows = adapter.read_manifest(args.manifest, expected_count=int(args.manifest_count))
        rows = adapter.select_manifest_rows(
            rows,
            start=int(args.manifest_row_start),
            stop=int(args.manifest_row_stop),
            limit=int(args.max_manifest_rows),
        )
        digest = adapter.manifest_hash(rows)
        if adapter.aggregate_paired is not None:
            print(adapter.aggregate_paired(args.output_root, digest), flush=True)
        return

    adapter.validate_required_assets(args)
    if simulation_app is None:
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(args, multi_gpu=False).app
    print("[BOOT] Isaac application initialized", flush=True)
    try:
        if args.generate_manifest:
            runtime = _call_setup(adapter, args, int(args.num_envs))
            try:
                rows = adapter.generate_manifest_with_env(args, runtime["env"])
                adapter.write_manifest(args.manifest, rows)
                print(
                    {
                        "manifest": str(args.manifest),
                        "rows": len(rows),
                        "manifest_hash": adapter.manifest_hash(rows),
                    },
                    flush=True,
                )
            finally:
                runtime["env"].close()
            return

        print(f"[BOOT] reading manifest: {args.manifest}", flush=True)
        rows = adapter.read_manifest(args.manifest, expected_count=int(args.manifest_count))
        rows = adapter.select_manifest_rows(
            rows,
            start=int(args.manifest_row_start),
            stop=int(args.manifest_row_stop),
            limit=int(args.max_manifest_rows),
        )
        print(f"[BOOT] manifest parsed rows={len(rows)}", flush=True)
        digest = adapter.manifest_hash(rows)
        print(f"[BOOT] manifest hash={digest}", flush=True)
        print(
            f"[BOOT] starting evaluation rows={len(rows)} num_envs={args.num_envs}",
            flush=True,
        )
        run_with_fallback(args, rows, digest, simulation_app, adapter)
        if adapter.aggregate_paired is not None:
            adapter.aggregate_paired(args.output_root, digest)
    finally:
        simulation_app.close()


def _is_fallback_trigger(exc: BaseException) -> bool:
    message = str(exc).lower()
    resource_like = any(
        token in message for token in ("cuda", "out of memory", "memory", "vram", "resource")
    )
    return bool(resource_like or "env_id-aware renderer" in message)


def _mark_slot_error(slot: Any, exc: BaseException) -> None:
    slot.terminal_reason = "error"
    slot.error = f"{type(exc).__name__}: {exc}"
