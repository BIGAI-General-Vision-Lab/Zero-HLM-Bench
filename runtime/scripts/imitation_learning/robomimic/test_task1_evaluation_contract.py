from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import task1_evaluation_contract as contract


def pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> tuple[float, ...]:
    return (x, y, z, 1.0, 0.0, 0.0, 0.0)


def entry(index: int, seed: int = 100) -> contract.ScenarioManifestEntry:
    return contract.ScenarioManifestEntry(
        scenario_id=f"scenario_{index:04d}",
        row_index=index,
        seed=seed + index,
        robot_pose=pose(float(index), 0.0, 0.4),
        object_pose=pose(0.2, -0.1, 0.7),
        placement_pose=pose(0.5, 0.1, 0.75),
    )


def test_manifest_entries_are_immutable_and_hash_stable(tmp_path: Path) -> None:
    rows = (entry(0), entry(1))
    manifest_path = tmp_path / "task1_manifest.jsonl"

    manifest_hash = contract.write_manifest_jsonl(rows, manifest_path)
    loaded = contract.read_manifest_jsonl(manifest_path)

    assert loaded == rows
    assert manifest_hash == contract.manifest_sha256(rows)
    assert manifest_hash == contract.manifest_sha256(loaded)
    assert len(loaded[0].row_hash) == 64
    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded[0].seed = 42  # type: ignore[misc]


def test_manifest_validation_rejects_payload_drift_and_duplicate_row_index(tmp_path: Path) -> None:
    manifest_path = tmp_path / "task1_manifest.jsonl"
    row = entry(0).manifest_row()
    row["object"]["pos"][0] = 9.0
    manifest_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(contract.ContractError, match="row_hash"):
        contract.read_manifest_jsonl(manifest_path)

    with pytest.raises(contract.ContractError, match="duplicate row_index"):
        contract.validate_manifest((entry(0), entry(1), contract.ScenarioManifestEntry(
            scenario_id="another",
            row_index=1,
            seed=10,
            robot_pose=pose(),
            object_pose=pose(),
            placement_pose=pose(),
        )))


def test_manifest_loading_rejects_non_integer_row_index_even_with_matching_hash(tmp_path: Path) -> None:
    manifest_path = tmp_path / "task1_manifest.jsonl"
    row = entry(0).manifest_row()
    row["row_index"] = 0.5
    row["row_hash"] = contract.stable_sha256({key: value for key, value in row.items() if key != "row_hash"})
    manifest_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="row_index must be an integer"):
        contract.read_manifest_jsonl(manifest_path)


def test_right_gripper_binarization_threshold_boundaries() -> None:
    assert contract.binarize_right_gripper(0.599999) == 0
    assert contract.binarize_right_gripper(0.6) == 1
    assert contract.binarize_right_gripper(1.0) == 1
    with pytest.raises(contract.ContractError, match="finite"):
        contract.binarize_right_gripper(float("nan"))


def test_task1_step_success_predicate_boundaries() -> None:
    assert contract.step_success(contract.Task1StepMetrics(-0.15, -0.10, 0.129999, 0.099999, 0.03))
    assert contract.step_success(contract.Task1StepMetrics(0.15, 0.10, 0.0, 0.0, 0.030001))
    assert not contract.step_success(contract.Task1StepMetrics(-0.150001, 0.0, 0.0, 0.0, 0.03))
    assert not contract.step_success(contract.Task1StepMetrics(0.0, 0.100001, 0.0, 0.0, 0.03))
    assert not contract.step_success(contract.Task1StepMetrics(0.0, 0.0, 0.13, 0.0, 0.03))
    assert not contract.step_success(contract.Task1StepMetrics(0.0, 0.0, 0.0, 0.10, 0.03))
    assert not contract.step_success(contract.Task1StepMetrics(0.0, 0.0, 0.0, 0.0, 0.029999))


def test_terminal_precedence_success_then_fall_then_horizon() -> None:
    assert contract.resolve_terminal(success=True, fall=True, step_index=1999) is contract.TerminalReason.SUCCESS
    assert contract.resolve_terminal(success=False, fall=True, step_index=1999) is contract.TerminalReason.FALL
    assert contract.resolve_terminal(success=False, fall=False, step_index=1999) is contract.TerminalReason.HORIZON
    assert contract.resolve_terminal(success=False, fall=False, step_index=1998) is None


def test_diagnostic_video_indices_are_every_tenth_among_100() -> None:
    indices = contract.diagnostic_video_indices()
    assert indices[:3] == (9, 19, 29)
    assert indices[-1] == 99
    assert len(indices) == 10
    assert all(0 <= index < 100 for index in indices)


def result(scenario_id: str, terminal: contract.TerminalReason, policy: str = "policy") -> contract.EvaluationResult:
    return contract.EvaluationResult(scenario_id=scenario_id, policy=policy, terminal=terminal, steps=500)


def test_aggregation_counts_terminals_and_policy_breakdown() -> None:
    rows = (
        result("a", contract.TerminalReason.SUCCESS, "base"),
        result("b", contract.TerminalReason.FALL, "base"),
        result("c", contract.TerminalReason.HORIZON, "candidate"),
        result("d", contract.TerminalReason.SUCCESS, "candidate"),
    )

    summary = contract.aggregate_results(rows)

    assert summary["episodes"] == 4
    assert summary["terminal_counts"] == {"success": 2, "fall": 1, "horizon": 1, "error": 0}
    assert summary["success_rate"] == 0.5
    assert summary["by_policy"]["candidate"] == {"success": 1, "fall": 0, "horizon": 1, "error": 0}


def test_paired_comparison_matches_by_scenario_id_and_reports_delta() -> None:
    baseline = (
        result("shared_win", contract.TerminalReason.FALL, "base"),
        result("shared_loss", contract.TerminalReason.SUCCESS, "base"),
        result("shared_tie", contract.TerminalReason.HORIZON, "base"),
        result("base_only", contract.TerminalReason.SUCCESS, "base"),
    )
    candidate = (
        result("shared_win", contract.TerminalReason.SUCCESS, "candidate"),
        result("shared_loss", contract.TerminalReason.HORIZON, "candidate"),
        result("shared_tie", contract.TerminalReason.FALL, "candidate"),
        result("candidate_only", contract.TerminalReason.SUCCESS, "candidate"),
    )

    comparison = contract.paired_comparison(baseline, candidate)

    assert comparison["paired"] == 3
    assert comparison["candidate_wins"] == 1
    assert comparison["candidate_losses"] == 1
    assert comparison["ties"] == 1
    assert comparison["net_success_delta"] == 0
    assert comparison["baseline_only"] == ("base_only",)
    assert comparison["candidate_only"] == ("candidate_only",)
    assert comparison["scenario_deltas"] == {
        "shared_loss": -1,
        "shared_tie": 0,
        "shared_win": 1,
    }


def test_paired_comparison_rejects_duplicate_scenario_results() -> None:
    rows = (
        result("dup", contract.TerminalReason.SUCCESS),
        result("dup", contract.TerminalReason.FALL),
    )
    with pytest.raises(contract.ContractError, match="duplicate baseline"):
        contract.paired_comparison(rows, ())


def test_released_task1_manifest_is_complete_and_hash_valid():
    root = Path(__file__).resolve().parents[4]
    path = root / "configs/manifests/task1_heldout100_independent_xy5cm.jsonl"
    rows = contract.read_manifest_jsonl(path)
    assert len(rows) == 100
    assert [row.row_index for row in rows] == list(range(100))
    assert len({row.scenario_id for row in rows}) == 100
    assert all(len(row.row_hash) == 64 for row in rows)


def test_released_source_states_cover_all_manifest_rows():
    root = Path(__file__).resolve().parents[4]
    path = root / "configs/manifests/task1_heldout100_independent_xy5cm_source_initial_states.json"
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    assert len(payload["scenarios"]) == 100

