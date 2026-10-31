"""Every rollout keeps its horizon even when falls do not stop it early."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


RUNTIME = Path(__file__).resolve().parents[1] / "runtime/scripts/imitation_learning/robomimic/evaluation_runtime.py"


@pytest.mark.parametrize(
    "stop_on_fall,height,steps,success,expected_reason,expected_fall",
    [
        (False, 0.1, 2000, False, "horizon", True),
        (False, float("nan"), 2000, False, "horizon", True),
        (False, 0.1, 1999, False, None, True),
        (True, 0.1, 2000, False, "fall", True),
        (True, 0.1, 1999, False, "fall", True),
        (True, 0.9, 2000, False, "horizon", False),
        (False, 0.9, 1999, False, None, False),
        (True, 0.1, 2000, True, "success", False),
    ],
)
def test_terminal_decision_respects_horizon(
    stop_on_fall, height, steps, success, expected_reason, expected_fall
):
    tree = ast.parse(RUNTIME.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "step_active_slots"
    )
    # Execute the actual terminal decision, without loading Isaac or stepping physics.
    decision = function.body[-1].body[-1]
    slot = SimpleNamespace(steps=steps, fall=False, terminal_reason=None)
    scope = {
        "np": np,
        "slot": slot,
        "args": SimpleNamespace(
            stop_on_fall=stop_on_fall, fall_height=0.5, max_control_steps=2000
        ),
        "root_pos": np.asarray([0.0, 0.0, height]),
        "metrics": {"ok": success},
    }
    exec(compile(ast.Module(body=[decision], type_ignores=[]), str(RUNTIME), "exec"), scope)
    assert slot.terminal_reason == expected_reason
    assert slot.fall is expected_fall
