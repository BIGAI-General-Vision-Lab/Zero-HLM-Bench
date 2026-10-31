import importlib.util
from pathlib import Path

import numpy as np


CONTRACT_PATH = Path(__file__).resolve().parents[3] / "pick_place" / "task2_evaluation_contract.py"
SPEC = importlib.util.spec_from_file_location("task2_evaluation_contract", CONTRACT_PATH)
assert SPEC is not None and SPEC.loader is not None
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)
TASK2_CABINET_REPLAY_SUCCESS_CFG = CONTRACT.TASK2_CABINET_REPLAY_SUCCESS_CFG
evaluate_task2_cabinet_success = CONTRACT.evaluate_task2_cabinet_success


def test_cabinet_success_requires_stable_tail_inside_target_region():
    frames = 10
    cabinet_pose = np.tile([0.85, 1.32666, 0.0, 0.9515756268, 0.0, 0.0, -0.3074147468], (frames, 1))
    rotation = CONTRACT._quat_wxyz_to_matrix(cabinet_pose[0, 3:7])
    local = np.array([-0.1, -0.1, 0.87])
    world = cabinet_pose[0, :3] + rotation @ local
    hat_pose = np.tile(np.r_[world, [1.0, 0.0, 0.0, 0.0]], (frames, 1))
    payload = {
        "object_root_pose": hat_pose,
        "object_root_velocity": np.zeros((frames, 6)),
        "hanger_root_pose": cabinet_pose,
    }
    result = evaluate_task2_cabinet_success(payload, TASK2_CABINET_REPLAY_SUCCESS_CFG)
    assert result["ok"]


def test_cabinet_success_rejects_hat_outside_target_region():
    frames = 10
    payload = {
        "object_root_pose": np.tile([2.0, 2.0, 0.9, 1.0, 0.0, 0.0, 0.0], (frames, 1)),
        "object_root_velocity": np.zeros((frames, 6)),
        "hanger_root_pose": np.tile([0.85, 1.32666, 0.0, 1.0, 0.0, 0.0, 0.0], (frames, 1)),
    }
    result = evaluate_task2_cabinet_success(payload, TASK2_CABINET_REPLAY_SUCCESS_CFG)
    assert not result["ok"]
    assert "inside_left_top_empty_region_xy" in result["reasons"]


def test_cabinet_success_reports_maximum_hat_lift_from_history():
    frames = 10
    hat_pose = np.tile([0.0, 0.0, 0.62, 1.0, 0.0, 0.0, 0.0], (frames, 1))
    hat_pose[4, 2] = 0.87
    payload = {
        "object_root_pose": hat_pose,
        "object_root_velocity": np.zeros((frames, 6)),
        "hanger_root_pose": np.tile([0.85, 1.32666, 0.0, 1.0, 0.0, 0.0, 0.0], (frames, 1)),
    }
    result = evaluate_task2_cabinet_success(payload, TASK2_CABINET_REPLAY_SUCCESS_CFG)
    assert np.isclose(result["max_object_lift"], 0.25)
