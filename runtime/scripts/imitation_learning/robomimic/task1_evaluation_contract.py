#!/usr/bin/env python3
"""Pure Task1 paired-evaluation contracts.

This module intentionally imports neither Isaac nor project simulator code.  It
defines the stable data contract used to pair scenarios across policies and the
small deterministic helpers needed to score and aggregate Task1 eval outputs.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CONTRACT_VERSION = 1
POSE_SIZE = 7
TASK1_HORIZON_STEPS = 2000
TASK1_SCENARIO_COUNT = 100
DIAGNOSTIC_VIDEO_STRIDE = 10
RIGHT_GRIPPER_CLOSED_THRESHOLD = 0.6
TASK1_PROMPT = "walk to the table, pick up the object and put it to the basket"


class ContractError(ValueError):
    """Raised when a payload cannot satisfy the Task1 paired eval contract."""


class TerminalReason(str, enum.Enum):
    SUCCESS = "success"
    FALL = "fall"
    HORIZON = "horizon"
    ERROR = "error"


def _finite_float(value: Any, label: str) -> float:
    output = float(value)
    if not math.isfinite(output):
        raise ContractError(f"{label} must be finite, got {value!r}")
    return output


def _strict_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError(f"{label} must be an integer")
    return value


def _validate_pose(values: Sequence[Any], label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != POSE_SIZE:
        raise ContractError(f"{label} must contain exactly {POSE_SIZE} scalars")
    return tuple(_finite_float(value, f"{label}[{index}]") for index, value in enumerate(values))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_sha256(value: Mapping[str, Any]) -> str:
    """Return the stable SHA256 of a JSON-serializable contract payload."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScenarioManifestEntry:
    """One immutable scenario row in the paired-eval JSONL manifest."""

    scenario_id: str
    row_index: int
    seed: int
    robot_pose: tuple[float, ...]
    object_pose: tuple[float, ...]
    placement_pose: tuple[float, ...]
    scene_translation_xy: tuple[float, float] = (0.0, 0.0)
    scene_delta_yaw: float = 0.0
    schema_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise ContractError(f"unsupported schema_version {self.schema_version}")
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ContractError("scenario_id must be a non-empty string")
        _strict_int(self.row_index, "row_index")
        _strict_int(self.seed, "seed")
        if self.row_index < 0:
            raise ContractError("row_index must be a non-negative integer")
        object.__setattr__(self, "robot_pose", _validate_pose(self.robot_pose, "robot_pose"))
        object.__setattr__(self, "object_pose", _validate_pose(self.object_pose, "object_pose"))
        object.__setattr__(self, "placement_pose", _validate_pose(self.placement_pose, "placement_pose"))
        object.__setattr__(
            self,
            "scene_translation_xy",
            tuple(_finite_float(value, f"scene_translation_xy[{index}]") for index, value in enumerate(self.scene_translation_xy)),
        )
        if len(self.scene_translation_xy) != 2:
            raise ContractError("scene_translation_xy must contain exactly 2 scalars")
        object.__setattr__(self, "scene_delta_yaw", _finite_float(self.scene_delta_yaw, "scene_delta_yaw"))

    @staticmethod
    def _pose_payload(pose: tuple[float, ...]) -> dict[str, list[float]]:
        return {"pos": list(pose[:3]), "quat_wxyz": list(pose[3:])}

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "row_index": self.row_index,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "robot": self._pose_payload(self.robot_pose),
            "object": self._pose_payload(self.object_pose),
            "placement": self._pose_payload(self.placement_pose),
            "scene_se2": {
                "translation_xy": list(self.scene_translation_xy),
                "delta_yaw": self.scene_delta_yaw,
            },
        }

    @property
    def row_hash(self) -> str:
        return stable_sha256(self.payload())

    def manifest_row(self) -> dict[str, Any]:
        row = self.payload()
        row["row_hash"] = self.row_hash
        return row

    @classmethod
    def from_manifest_row(cls, row: Mapping[str, Any]) -> "ScenarioManifestEntry":
        expected_keys = {
            "schema_version",
            "row_index",
            "scenario_id",
            "seed",
            "robot",
            "object",
            "placement",
            "scene_se2",
            "row_hash",
        }
        extra = set(row) - expected_keys
        missing = expected_keys - set(row)
        if extra or missing:
            raise ContractError(f"manifest row keys mismatch extra={sorted(extra)} missing={sorted(missing)}")
        payload = {key: row[key] for key in expected_keys if key != "row_hash"}
        expected_sha = stable_sha256(payload)
        if row["row_hash"] != expected_sha:
            raise ContractError("manifest row_hash does not match payload")

        def pose(name: str) -> tuple[Any, ...]:
            value = row[name]
            if not isinstance(value, Mapping) or set(value) != {"pos", "quat_wxyz"}:
                raise ContractError(f"{name} must contain only pos and quat_wxyz")
            return tuple(value["pos"]) + tuple(value["quat_wxyz"])

        scene_se2 = row["scene_se2"]
        if not isinstance(scene_se2, Mapping) or set(scene_se2) != {"translation_xy", "delta_yaw"}:
            raise ContractError("scene_se2 must contain only translation_xy and delta_yaw")
        return cls(
            scenario_id=row["scenario_id"],
            row_index=_strict_int(row["row_index"], "row_index"),
            seed=_strict_int(row["seed"], "seed"),
            robot_pose=pose("robot"),
            object_pose=pose("object"),
            placement_pose=pose("placement"),
            scene_translation_xy=tuple(scene_se2["translation_xy"]),
            scene_delta_yaw=scene_se2["delta_yaw"],
            schema_version=_strict_int(row["schema_version"], "schema_version"),
        )


def manifest_sha256(entries: Sequence[ScenarioManifestEntry]) -> str:
    """Hash the exact canonical JSONL bytes for a validated manifest."""

    validate_manifest(entries)
    text = "".join(_canonical_json(entry.manifest_row()) + "\n" for entry in entries)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_manifest(entries: Sequence[ScenarioManifestEntry]) -> None:
    ids: set[str] = set()
    row_indices: set[int] = set()
    previous_index = -1
    for entry in entries:
        if not isinstance(entry, ScenarioManifestEntry):
            raise ContractError("manifest entries must be ScenarioManifestEntry instances")
        if entry.scenario_id in ids:
            raise ContractError(f"duplicate scenario_id {entry.scenario_id!r}")
        if entry.row_index in row_indices:
            raise ContractError(f"duplicate row_index {entry.row_index}")
        if entry.row_index <= previous_index:
            raise ContractError("manifest entries must be sorted by strictly increasing row_index")
        ids.add(entry.scenario_id)
        row_indices.add(entry.row_index)
        previous_index = entry.row_index


def write_manifest_jsonl(entries: Sequence[ScenarioManifestEntry], path: str | Path) -> str:
    validate_manifest(entries)
    lines = [_canonical_json(entry.manifest_row()) + "\n" for entry in entries]
    Path(path).write_text("".join(lines), encoding="utf-8")
    return manifest_sha256(entries)


def read_manifest_jsonl(path: str | Path) -> tuple[ScenarioManifestEntry, ...]:
    entries: list[ScenarioManifestEntry] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ContractError(f"blank manifest line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSON on manifest line {line_number}: {exc}") from exc
        entries.append(ScenarioManifestEntry.from_manifest_row(row))
    output = tuple(entries)
    validate_manifest(output)
    return output


def binarize_right_gripper(value: Any) -> int:
    """Map right gripper openness to the Task1 binary command."""

    return 0 if _finite_float(value, "right_gripper") < RIGHT_GRIPPER_CLOSED_THRESHOLD else 1


@dataclass(frozen=True)
class Task1StepMetrics:
    dx: float
    dy: float
    dz: float
    speed: float
    max_lift: float

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            object.__setattr__(self, field.name, _finite_float(getattr(self, field.name), field.name))


def step_success(metrics: Task1StepMetrics) -> bool:
    return (
        -0.15 <= metrics.dx <= 0.15
        and -0.10 <= metrics.dy <= 0.10
        and metrics.dz < 0.13
        and metrics.speed < 0.10
        and metrics.max_lift >= 0.03
    )


def resolve_terminal(*, success: bool, fall: bool, step_index: int, horizon: int = TASK1_HORIZON_STEPS) -> TerminalReason | None:
    """Resolve terminal state with precedence: success, then fall, then horizon."""

    if horizon <= 0:
        raise ContractError("horizon must be positive")
    if step_index < 0:
        raise ContractError("step_index must be non-negative")
    if success:
        return TerminalReason.SUCCESS
    if fall:
        return TerminalReason.FALL
    if step_index >= horizon - 1:
        return TerminalReason.HORIZON
    return None


def diagnostic_video_indices(
    scenario_count: int = TASK1_SCENARIO_COUNT,
    stride: int = DIAGNOSTIC_VIDEO_STRIDE,
) -> tuple[int, ...]:
    if scenario_count <= 0:
        raise ContractError("scenario_count must be positive")
    if stride <= 0:
        raise ContractError("stride must be positive")
    return tuple(range(stride - 1, scenario_count, stride))


@dataclass(frozen=True)
class EvaluationResult:
    scenario_id: str
    policy: str
    terminal: TerminalReason
    steps: int
    success_step: int | None = None

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ContractError("scenario_id is required")
        if not self.policy:
            raise ContractError("policy is required")
        if not isinstance(self.terminal, TerminalReason):
            object.__setattr__(self, "terminal", TerminalReason(self.terminal))
        if self.steps <= 0:
            raise ContractError("steps must be positive")
        if self.success_step is not None and self.success_step < 0:
            raise ContractError("success_step must be non-negative when present")

    @property
    def success(self) -> bool:
        return self.terminal is TerminalReason.SUCCESS


def aggregate_results(results: Iterable[EvaluationResult]) -> dict[str, Any]:
    rows = tuple(results)
    terminal_counts = {reason.value: 0 for reason in TerminalReason}
    by_policy: dict[str, dict[str, int]] = {}
    for row in rows:
        terminal_counts[row.terminal.value] += 1
        policy_counts = by_policy.setdefault(row.policy, {reason.value: 0 for reason in TerminalReason})
        policy_counts[row.terminal.value] += 1
    return {
        "episodes": len(rows),
        "terminal_counts": terminal_counts,
        "successes": terminal_counts[TerminalReason.SUCCESS.value],
        "success_rate": (terminal_counts[TerminalReason.SUCCESS.value] / len(rows)) if rows else 0.0,
        "by_policy": by_policy,
    }


def paired_comparison(
    baseline: Iterable[EvaluationResult],
    candidate: Iterable[EvaluationResult],
) -> dict[str, Any]:
    baseline_by_id = _unique_results_by_scenario(baseline, "baseline")
    candidate_by_id = _unique_results_by_scenario(candidate, "candidate")
    shared_ids = tuple(sorted(set(baseline_by_id) & set(candidate_by_id)))
    baseline_only = tuple(sorted(set(baseline_by_id) - set(candidate_by_id)))
    candidate_only = tuple(sorted(set(candidate_by_id) - set(baseline_by_id)))
    wins = losses = ties = 0
    deltas: dict[str, int] = {}
    for scenario_id in shared_ids:
        delta = int(candidate_by_id[scenario_id].success) - int(baseline_by_id[scenario_id].success)
        deltas[scenario_id] = delta
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            ties += 1
    return {
        "paired": len(shared_ids),
        "candidate_wins": wins,
        "candidate_losses": losses,
        "ties": ties,
        "net_success_delta": wins - losses,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "scenario_deltas": deltas,
    }


def _unique_results_by_scenario(results: Iterable[EvaluationResult], label: str) -> dict[str, EvaluationResult]:
    output: dict[str, EvaluationResult] = {}
    for result in results:
        if result.scenario_id in output:
            raise ContractError(f"duplicate {label} result for scenario_id {result.scenario_id!r}")
        output[result.scenario_id] = result
    return output
