"""Simulation-independent contracts for the shared paired-evaluation runner."""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "runtime" / "scripts" / "imitation_learning" / "robomimic"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from evaluation_runner import EvaluationTaskAdapter  # noqa: E402
import evaluation_runner as runner  # noqa: E402


RUNNER = HERE / "evaluation_runner.py"
TASK_ENTRYPOINTS = (
    HERE / "evaluate_task1.py",
    HERE / "evaluate_task2.py",
    HERE / "evaluate_task3.py",
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_runner_exposes_explicit_task_adapter_and_common_checkpoint_runner() -> None:
    tree = _tree(RUNNER)
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "EvaluationTaskAdapter" in classes
    assert {"run_checkpoint", "run_evaluation", "parse_args"} <= functions

    checkpoint = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_checkpoint")
    parameters = {arg.arg for arg in (*checkpoint.args.posonlyargs, *checkpoint.args.args, *checkpoint.args.kwonlyargs)}
    assert "adapter" in parameters


def test_task_entrypoints_delegate_without_mutating_sibling_globals() -> None:
    for path in TASK_ENTRYPOINTS:
        source = path.read_text(encoding="utf-8")
        tree = _tree(path)
        assert "_install_adapter" not in source
        assert " as base" not in source
        assert "evaluate_task1" not in source
        if path.name != "evaluate_task1.py":
            assert "evaluation_runtime as shared" in source
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"run_evaluation", "run_task"}
            for node in ast.walk(tree)
        ), path.name

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                assert not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in {"base", "task1"}
                ), f"sibling-global assignment in {path.name}"


def test_runtime_manifest_defaults_are_the_released_100_rows() -> None:
    for path in TASK_ENTRYPOINTS:
        tree = _tree(path)
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
                if target in {"MANIFEST_ROW_COUNT", "TASK2_SCENARIO_COUNT", "TASK3_SCENARIO_COUNT"}:
                    values[target] = ast.literal_eval(node.value)
        assert values, path.name
        assert set(values.values()) == {100}, path.name


def test_task_chunk_defaults_preserve_task1_16_and_task2_task3_40() -> None:
    expected = {
        "evaluate_task1.py": 16,
        "evaluate_task2.py": 40,
        "evaluate_task3.py": 40,
    }
    for path in TASK_ENTRYPOINTS:
        tree = _tree(path)
        values = [
            keyword.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "replace"
            for keyword in node.keywords
            if keyword.arg == "default_chunk_horizon"
        ]
        if path.name == "evaluate_task1.py":
            source = (HERE / "evaluation_runtime.py").read_text(encoding="utf-8")
            assert "default_chunk_horizon=16" in source
        else:
            assert values and ast.literal_eval(values[0]) == expected[path.name]


def test_task3_lifecycle_overrides_are_named_adapter_hooks() -> None:
    source = (HERE / "evaluate_task3.py").read_text(encoding="utf-8")
    for name in (
        "_apply_task3_post_lift_friction",
        "_install_task3_friction_override",
        "_install_task3_actuator_profile_override",
        "_install_task3_chunk_diagnostics",
    ):
        assert name in source
    assert "before_step" in source
    assert "after_chunk" in source


def _adapter(**overrides) -> EvaluationTaskAdapter:
    callbacks = {
        name: lambda *args, **kwargs: None
        for name in (
            "parse_args",
            "validate_required_assets",
            "load_processor_modality_configs",
            "read_manifest",
            "write_manifest",
            "manifest_hash",
            "select_manifest_rows",
            "pending_rows",
            "reset_env_slot_to_manifest",
            "setup_runtime",
            "generate_manifest_with_env",
            "make_policies",
            "init_gs_renderer",
            "reset_slot",
            "synchronize_reset_wave",
            "request_chunk",
            "step_active_slots",
            "capture_gs_frame_env",
            "mirror_env_into_foreground_sandbox",
            "finalize_slot",
            "append_episode_rows",
            "read_episode_csv",
            "write_tables",
            "write_aggregate",
            "write_metadata",
            "diagnostic_video_required",
            "robot_inference_client",
        )
    }
    callbacks.update(
        name="test",
        task_id="test-task",
        manifest_row_count=100,
        output_root=Path("outputs/test"),
        default_init_joint_state_json=None,
        default_init_joint_state_dataset=None,
        default_manifest=Path("manifest.jsonl"),
        default_chunk_horizon=16,
        env_slot_type=object,
    )
    callbacks.update(overrides)
    return EvaluationTaskAdapter(**callbacks)


def test_adapter_defaults_are_applied_without_sibling_global_mutation() -> None:
    adapter = _adapter()
    args = argparse.Namespace(
        task=None,
        manifest=None,
        manifest_count=None,
        output_root=None,
        chunk_horizon=None,
        init_joint_state_json=None,
        init_joint_state_dataset=None,
    )
    result = adapter.with_defaults(args)
    assert result is args
    assert args.task == "test-task"
    assert args.manifest == Path("manifest.jsonl")
    assert args.manifest_count == 100
    assert args.output_root == Path("outputs/test")
    assert args.chunk_horizon == 16


def test_schema_only_uses_task_adapter_modality_summary(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    seen = []
    adapter = _adapter(
        load_processor_modality_configs=lambda path, embodiment: seen.append((path, embodiment))
        or {"video": {"delta_indices": [0]}},
        summarize_modality_config=lambda configs: {"keys": sorted(configs)},
    )
    args = SimpleNamespace(
        checkpoint_dir=checkpoint,
        embodiment="unitree_g1_sonic",
        schema_only=True,
        aggregate_only=False,
    )
    output = io.StringIO()
    with redirect_stdout(output):
        runner.run_evaluation(args, adapter)
    assert seen == [(checkpoint, "unitree_g1_sonic")]
    assert "{'keys': ['video']}" in output.getvalue()


def test_common_runner_preserves_reset_chunk_step_and_finalize_hooks(tmp_path: Path, monkeypatch) -> None:
    events = []

    class Slot:
        def __init__(self, env_id):
            self.env_id = env_id
            self.scenario = None
            self.policy = None
            self.chunk = None
            self.chunk_pos = 0
            self.terminal_reason = None
            self.final_frame = None
            self.video = None
            self.steps = 0

        @property
        def active(self):
            return self.scenario is not None and self.terminal_reason is None

    class Client:
        def ping(self):
            events.append("ping")
            return True

        def reset(self):
            events.append("client_reset")

        def get_modality_config(self):
            return {"video": {"delta_indices": [0]}}

        def close(self):
            events.append("client_close")

    renderer = SimpleNamespace(close=lambda: events.append("renderer_close"))
    row = SimpleNamespace(row_index=0, scenario_id="row-0")
    adapter = _adapter(
        pending_rows=lambda *args: [row],
        load_processor_modality_configs=lambda *args: {"video": {"delta_indices": [0]}},
        robot_inference_client=lambda *args: Client(),
        init_gs_renderer=lambda *args: renderer,
        make_policies=lambda *args: [object()],
        env_slot_type=Slot,
        reset_slot=lambda args, runtime, slot, row, policy, run_dir, adapter: (
            events.append("reset"), setattr(slot, "scenario", row), setattr(slot, "policy", policy)
        ),
        synchronize_reset_wave=lambda *args: events.append("sync_reset"),
        request_chunk=lambda *args: (events.append("request_chunk"), setattr(args[-2], "chunk", object())),
        before_step=lambda *args: events.append("before_step"),
        after_chunk=lambda *args: events.append("after_chunk"),
        step_active_slots=lambda args, runtime, slots, gs_render, adapter: (
            events.append("step"), setattr(slots[0], "steps", 1),
            setattr(slots[0], "terminal_reason", "success"),
            setattr(slots[0], "final_frame", np.zeros((1, 1, 3), dtype=np.uint8))
        ),
        finalize_slot=lambda *args: (
            events.append("finalize")
            or {
                "scenario_id": "row-0",
                "terminal_reason": "success",
                "control_steps": 1,
                "success": True,
            }
        ),
        after_episode=lambda *args: events.append("after_episode"),
        append_episode_rows=lambda *args: events.append("append"),
        read_episode_csv=lambda *args: [],
        write_tables=lambda *args: events.append("tables"),
        write_aggregate=lambda *args: events.append("aggregate"),
        write_metadata=lambda *args: events.append("metadata"),
    )
    monkeypatch.setattr(runner, "validate_policy_modalities", lambda *args: events.append("validate"))
    args = SimpleNamespace(
        output_root=tmp_path,
        model_name="custom",
        resume=False,
        checkpoint_dir=tmp_path,
        embodiment="unitree_g1_sonic",
        gr00t_host="127.0.0.1",
        gr00t_port=5555,
        gr00t_timeout_ms=100,
        chunk_horizon=16,
        log_every=100,
    )
    runner.run_checkpoint(args, {"env": object(), "robot": object(), "sim_joint_names": []}, [row], "digest", 1, None, adapter)
    assert events[:8] == [
        "ping",
        "client_reset",
        "validate",
        "reset",
        "sync_reset",
        "request_chunk",
        "after_chunk",
        "before_step",
    ]
    assert events[8:] == ["step", "finalize", "after_episode", "append", "tables", "aggregate", "metadata", "renderer_close", "client_close"]
