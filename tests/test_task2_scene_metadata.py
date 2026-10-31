"""CPU regressions for Task2's configured asset provenance."""

import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "runtime/scripts/imitation_learning/robomimic/evaluate_task2.py"
ENV_CONFIG = ROOT / "runtime/pick_place/final_task_2_pick_up_hat_env_cfg.py"


def configured_paths(root):
    names = {"PROJECT_ROOT", "ASSETS_USD_ROOT", "ROBOT_USD_PATH", "SCENE_USD_PATH",
             "OBJECT_USD_PATH", "PLACEMENT_USD_PATH", "HANGER_USD_PATH"}
    assignments = [node for node in ast.parse(ENV_CONFIG.read_text()).body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    namespace = {"Path": Path, "os": os,
                 "__file__": str(root / "runtime/pick_place/final_task_2_pick_up_hat_env_cfg.py")}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(ENV_CONFIG), "exec"), namespace)
    module = types.ModuleType("pick_place.final_task_2_pick_up_hat_env_cfg")
    for name in names:
        setattr(module, name, namespace[name])
    return module


class Task2MetadataTests(unittest.TestCase):
    def check_metadata(self, override):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {key: value for key, value in os.environ.items()
                           if key not in {"TASK2_EVAL_ASSETS_ROOT", "TASK2_SCENE_USD_PATH"}}
            if override:
                environment["TASK2_SCENE_USD_PATH"] = str(root / "alternate/scene.usdc")
                environment["TASK2_EVAL_ASSETS_ROOT"] = str(root / "alternate/usd")
            with patch.dict(os.environ, environment, clear=True):
                config = configured_paths(root)
                for name in ("ROBOT_USD_PATH", "SCENE_USD_PATH", "OBJECT_USD_PATH",
                             "PLACEMENT_USD_PATH", "HANGER_USD_PATH"):
                    path = getattr(config, name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(name.encode())
                run_dir = root / "run"
                run_dir.mkdir()
                original_calls = []

                def original(*args):
                    original_calls.append(args)
                    (run_dir / "metadata.json").write_text('{"existing": "preserved"}')

                shared = types.SimpleNamespace(
                    REPO_ROOT=root / "runtime",
                    sha256_file=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
                    write_json=lambda path, value: path.write_text(json.dumps(value)),
                    write_metadata=original,
                )
                namespace = {"Path": Path, "json": json, "os": os, "shared": shared,
                             "TASK2_ASSETS_ROOT": root / "assets/task2"}
                function = next(node for node in ast.parse(EVALUATOR.read_text()).body
                                if isinstance(node, ast.FunctionDef) and node.name == "write_metadata")
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(EVALUATOR), "exec"), namespace)
                package = types.ModuleType("pick_place")
                package.__path__ = []
                with patch.dict(sys.modules, {"pick_place": package,
                                              "pick_place.final_task_2_pick_up_hat_env_cfg": config}):
                    namespace["write_metadata"](None, run_dir, "manifest", 1, None, {})
                result = json.loads((run_dir / "metadata.json").read_text())
                self.assertEqual(len(original_calls), 1)
                self.assertEqual(result["existing"], "preserved")
                self.assertEqual(result["task2_assets"]["scene_usd"], str(config.SCENE_USD_PATH))
                self.assertEqual(result["task2_asset_hashes"]["scene_usd"],
                                 hashlib.sha256(config.SCENE_USD_PATH.read_bytes()).hexdigest())
                self.assertEqual(result["task2_assets"]["cabinet_usd"], str(config.HANGER_USD_PATH))
                self.assertEqual(len(result["task2_assets"]), 5)
                for name, path in result["task2_assets"].items():
                    self.assertEqual(result["task2_asset_hashes"][name],
                                     hashlib.sha256(Path(path).read_bytes()).hexdigest())

    def test_default_scene_metadata(self):
        self.check_metadata(override=False)

    def test_configured_scene_and_asset_override_metadata(self):
        self.check_metadata(override=True)


if __name__ == "__main__":
    unittest.main()
