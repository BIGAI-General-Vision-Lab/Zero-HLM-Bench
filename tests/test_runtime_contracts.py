"""Run the evaluator's simulation-independent regression contracts."""
import importlib.util
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1] / "runtime/scripts/imitation_learning/robomimic"
for index, name in enumerate(("task1", "task2", "task3")):
    path = RUNTIME / f"test_{name}_evaluation_contract.py"
    spec = importlib.util.spec_from_file_location(f"runtime_contract_{index}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, value in vars(module).items():
        if key.startswith("test_"):
            globals()[f"test_{name}_{key[5:]}"] = value


def load_runtime_function(filename, name, scope):
    import ast
    source = RUNTIME / filename
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
    return scope[name]


def test_static_task_scene_validation(tmp_path, monkeypatch):
    import argparse
    import os
    import pytest
    from types import SimpleNamespace

    backend = tmp_path / "Real2Sim"
    backend.mkdir()
    (backend / "z_buffer_merger_with_robot.py").touch()
    (backend.parent / "2d-gaussian-splatting").mkdir()
    validate = load_runtime_function("evaluation_runtime.py", "validate_required_assets",
                                     {"Path": Path, "argparse": argparse, "os": os,
                                      "detect_gs_backend_from_ply": lambda _: "2dgs"})
    scenes = {"Isaac-Final-Walk-PickPlace-Render-Abs-v0": "B1_0716_aligned_0711.ply",
              "Isaac-Final-Task2-PickUpHat-Render-Abs-v0": "B1_0711.ply"}
    for scene in scenes.values():
        (tmp_path / scene).touch()
    args = SimpleNamespace(checkpoint_dir=tmp_path, headcam_calib=tmp_path, models_dir=tmp_path,
                           init_joint_state_from_indis=False, gs_foreground_mode="isaac_mesh",
                           gs_backend="auto", two_d_gs_real2sim_root=backend)
    monkeypatch.setenv("TASK1_ALLOW_LEGACY_GS_PROBE", "1")
    for task, expected in scenes.items():
        args.task = task
        args.gs_scene_path = tmp_path / expected
        validate(args)
        args.gs_scene_path = tmp_path / next(name for name in scenes.values() if name != expected)
        with pytest.raises(RuntimeError):
            validate(args)


def test_task3_asset_validation_requires_dynamic_door_assets(tmp_path):
    from types import SimpleNamespace
    import pytest

    files = ["scene.ply", "door.ply", "joint.json", "bank.jsonl", "pose.json",
             "usd/robots/final_robot_usd_align_sonic_portable.usd",
             "scene.usdc",
             "usd/assets/B1/task3_lanzi_Model_1_scaled_centered.usd",
             "usd/assets/B1/dengzi_Model_2_scaled_centered.usd"]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    scope = {"Path": Path, "TASK3_ASSETS_ROOT": tmp_path, "TASK3_DOOR_PLY": tmp_path / "door.ply",
             "TASK3_JOINT_INFO": tmp_path / "joint.json", "TASK3_SCENARIO_BANK": tmp_path / "bank.jsonl",
             "TASK3_INITIAL_BODY_POSE": tmp_path / "pose.json",
             "TASK3_SCENE_USD_PATH": tmp_path / "scene.usdc"}
    validate = load_runtime_function("evaluate_task3.py", "validate_required_assets", scope)
    args = SimpleNamespace(checkpoint_dir=tmp_path, gs_scene_path=tmp_path / "scene.ply",
                           headcam_calib=tmp_path, models_dir=tmp_path)
    validate(args)
    scope["TASK3_JOINT_INFO"].unlink()
    with pytest.raises(FileNotFoundError, match="joint.json"):
        validate(args)


def test_simulator_uses_only_the_selected_gpu():
    import ast
    from types import SimpleNamespace
    import pytest

    source = RUNTIME / "evaluate_task1.py"
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    args = SimpleNamespace(checkpoint_dir=str(RUNTIME), embodiment="unitree_g1_sonic",
                           schema_only=False, aggregate_only=False, device="cuda:6")
    observed = {}

    class LaunchReached(Exception):
        pass

    def launch(received, **settings):
        observed.update(device=received.device, **settings)
        raise LaunchReached

    def run_evaluation(received, adapter):
        del adapter
        launch(received, multi_gpu=False)

    scope = {
        "parse_args": lambda adapter: args,
        "make_task_adapter": lambda: object(),
        "run_evaluation": run_evaluation,
    }
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), "exec"), scope)
    with pytest.raises(LaunchReached):
        scope["main"]()
    assert observed == {"device": "cuda:6", "multi_gpu": False}


def test_task3_metadata_records_configured_scene(tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace
    import os

    scene = tmp_path / "astra/scene.usdc"
    paths = [scene, tmp_path / "usd/robots/final_robot_usd_align_sonic_portable.usd",
             tmp_path / "usd/assets/B1/task3_lanzi_Model_1_scaled_centered.usd",
             tmp_path / "usd/assets/B1/dengzi_Model_2_scaled_centered.usd"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    write_json = lambda path, value: Path(path).write_text(json.dumps(value))
    base = SimpleNamespace(sha256_file=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
                           write_json=write_json,
                           write_metadata=lambda *args: write_json(tmp_path / "metadata.json", {}))
    scope = {"Path": Path, "json": json, "os": os, "shared": base,
             "TASK3_ASSETS_ROOT": tmp_path, "TASK3_SCENE_USD_PATH": scene}
    for name in ("TASK3_DOOR_PLY", "TASK3_JOINT_INFO", "TASK3_SCENARIO_BANK", "TASK3_INITIAL_BODY_POSE"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        scope[name] = path
    write = load_runtime_function("evaluate_task3.py", "write_metadata", scope)
    write(SimpleNamespace(gs_scene_path=scene), tmp_path, None, None, None, None)
    result = json.loads((tmp_path / "metadata.json").read_text())
    assert result["task3_assets"]["scene_usd"] == str(scene)
    assert result["task3_asset_hashes"]["scene_usd"] == hashlib.sha256(scene.read_bytes()).hexdigest()
    assert "scene_wrapper_usda" not in result["task3_assets"]


def test_task3_scene_resolution_matches_environment(tmp_path, monkeypatch):
    import ast
    import os
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    adapter = RUNTIME / "evaluate_task3.py"
    environment = root / "runtime/pick_place/final_task_3_carry_basket_kick_door_env_cfg.py"
    for override in (None, str(tmp_path / "selected_scene.usdc")):
        monkeypatch.delenv("TASK3_SCENE_USD_PATH", raising=False)
        if override is not None:
            monkeypatch.setenv("TASK3_SCENE_USD_PATH", override)
        values = []
        for source, names, result_name in (
            (adapter, {"TASK3_SCENE_USD_PATH"}, "TASK3_SCENE_USD_PATH"),
            (environment, {"TASK3_DEFAULT_BUNDLE_ROOT", "SCENE_USD_PATH"}, "SCENE_USD_PATH"),
        ):
            scope = {"Path": Path, "os": os, "__file__": str(source),
                     "shared": SimpleNamespace(REPO_ROOT=root / "runtime")}
            nodes = [node for node in ast.parse(source.read_text()).body
                     if isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id in names for target in node.targets)]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
            values.append(scope[result_name])
        assert values[0] == values[1] == Path(override or root / "assets/task3/scene.usdc")


def test_append_video_columns_preserves_previous_results(tmp_path):
    import csv
    from typing import Any
    import pytest

    append = load_runtime_function('evaluation_runtime.py', 'append_episode_rows',
                                   {'Path': Path, 'Any': Any, 'csv': csv})
    path = tmp_path / 'episodes.csv'
    path.write_text('scenario_id,success,error\nold,True,\n')
    append(path, [{'scenario_id': 'new', 'success': False, 'third_person_video_path': 'third.mp4',
                   'video_frames': 5, 'video_fps': 50, 'error': 'failure'}])
    rows = list(csv.DictReader(path.open()))
    assert rows[0]['scenario_id'] == 'old'
    assert rows[0]['error'] == ''
    assert rows[0]['third_person_video_path'] == ''
    assert rows[1]['third_person_video_path'] == 'third.mp4'
    assert rows[1]['error'] == 'failure'
    assert not any(None in row for row in rows)
    append(path, [{'scenario_id': 'next', 'success': True, 'error': ''}])
    rows = list(csv.DictReader(path.open()))
    assert rows[-1]['third_person_video_path'] == ''
    assert len(rows) == 3
    before = path.read_bytes()
    path.write_text('scenario_id,error\nbad,,extra\n')
    corrupt = path.read_bytes()
    with pytest.raises(ValueError, match='CSV'):
        append(path, [{'scenario_id': 'next', 'third_person_video_path': 'third.mp4'}])
    assert path.read_bytes() == corrupt
    assert before != corrupt
