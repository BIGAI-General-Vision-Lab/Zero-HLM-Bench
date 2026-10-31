import argparse
import ast
import __future__
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zerohlm_bench.launch import build_client, build_server, flag_arguments, main
from zerohlm_bench.protocol import validate_policy_modalities
from types import SimpleNamespace
from unittest.mock import Mock, patch


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("configs", "runtime/scripts", "assets/task2", "checkpoint", "groot/gr00t/eval"):
            (self.root / name).mkdir(parents=True)
        self.manifest = self.root / "configs/task2.jsonl"
        self.manifest.write_text(''.join(json.dumps({"row": i}) + "\n" for i in range(100)))
        (self.root / "runtime/scripts/evaluate.py").write_text("print('evaluation')\n")
        self.config = {
            "schema_version": 1, "task": "task2", "evaluator": "scripts/evaluate.py",
            "protocol": "configs/protocol/shared",
            "manifest": {"path": "configs/task2.jsonl", "rows": 100,
                         "sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest()},
            "required_paths": ["{root}/assets/task2"],
            "env": {"TASK2_ASSET_ROOT": "{root}/assets/task2"},
            "args": {"--num_envs": 1, "--fallback_env_counts": [1], "--no-resume": True,
                     "--embodiment": "unitree_g1_sonic", "--chunk_horizon": 40, "--device": "cuda:0"},
        }
        self.save()
        self.checkpoint = self.root / "checkpoint"
        modalities = {key: {"delta_indices": [0], "modality_keys": [key]}
                      for key in ("video", "state", "language", "action")}
        (self.checkpoint / "processor_config.json").write_text(json.dumps({
            "processor_kwargs": {"modality_configs": {"unitree_g1_sonic": modalities}}}))
        protocol = self.root / "configs/protocol/shared"
        protocol.mkdir(parents=True)
        shutil.copy2(self.checkpoint / "processor_config.json", protocol / "processor_config.json")
        (self.checkpoint / "config.json").write_text("{}")
        (self.checkpoint / "model.safetensors").write_bytes(b"weights")
        (self.root / "groot/gr00t/eval/run_gr00t_server.py").write_text("")
        self.args = argparse.Namespace(task="task2", checkpoint=self.checkpoint,
                                       output=self.root / "outputs/task2", host="127.0.0.1", port=5555,
                                       row_start=0, row_count=None, python=sys.executable, device=None)

    def save(self):
        (self.root / "configs/task2.json").write_text(json.dumps(self.config))

    def test_portable_full_launch(self):
        launch = build_client(self.root, self.args)
        self.assertEqual(launch.provenance["rows"], [0, 100])
        self.assertEqual(launch.env["TASK2_ASSET_ROOT"], str(self.root / "assets/task2"))
        self.assertIn("--no-resume", launch.command)
        self.assertEqual(launch.command[launch.command.index("--manifest_count") + 1], "100")
        self.assertFalse(self.args.output.exists())

    def test_gpu_override_is_recorded_without_changing_config(self):
        self.args.device = "cuda:6"
        launch = build_client(self.root, self.args)
        self.assertEqual(launch.command[launch.command.index("--device") + 1], "cuda:6")
        self.assertEqual(launch.provenance["config"]["args"]["--device"], "cuda:6")
        self.assertEqual(json.loads((self.root / "configs/task2.json").read_text())["args"]["--device"], "cuda:0")
        self.assertEqual(launch.provenance["config"]["args"]["--chunk_horizon"], 40)

    def test_cli_records_launch_in_expanded_output_directory(self):
        requested = Path("~/outputs/task2")
        resolved = self.root / "expanded_outputs/task2"

        def expand(path):
            return resolved if path == requested else path

        with patch.object(Path, "expanduser", expand), \
             patch.object(Path, "mkdir", autospec=True) as mkdir, \
             patch.object(Path, "write_text", autospec=True) as write, \
             patch("zerohlm_bench.launch.os.chdir"), \
             patch("zerohlm_bench.launch.os.execvpe") as execute, \
             patch.object(sys, "argv", ["start_client.py", "--task", "task2", "--output", str(requested)]):
            self.assertEqual(main("client", self.root), 0)
        mkdir.assert_called_once_with(resolved, parents=True, exist_ok=True)
        self.assertEqual(write.call_args.args[0], resolved / "launch.json")
        command = execute.call_args.args[1]
        self.assertEqual(command[command.index("--output_root") + 1], str(resolved))

    def test_scene_dependencies_are_verified_before_launch(self):
        asset = self.root / "assets/task2/albedo.png"
        payload = b"texture"
        entry = {"path": "assets/task2/albedo.png", "bytes": len(payload),
                 "sha256": hashlib.sha256(payload).hexdigest()}
        manifest = self.root / "configs/scene_assets.json"

        def pin(entries):
            manifest.write_text(json.dumps({"files": entries}))
            self.config["scene_assets"] = {"path": "configs/scene_assets.json",
                                           "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
            self.save()

        pin([entry])
        for contents in (None, b"broken!", b"short"):
            with self.subTest(contents=contents):
                if contents is not None:
                    asset.write_bytes(contents)
                with self.assertRaisesRegex(ValueError, "Scene asset"):
                    build_client(self.root, self.args)
        asset.write_bytes(payload)
        self.assertEqual(build_client(self.root, self.args).provenance["rows"], [0, 100])
        manifest.write_text('{"files": []}')
        with self.assertRaisesRegex(ValueError, "Scene manifest"):
            build_client(self.root, self.args)
        for path in ("../outside.png", "/outside.png"):
            pin([dict(entry, path=path)])
            with self.assertRaisesRegex(ValueError, "repository-relative"):
                build_client(self.root, self.args)

    def test_subset_retains_full_manifest(self):
        self.args.row_start, self.args.row_count = 20, 2
        launch = build_client(self.root, self.args)
        self.assertEqual(launch.provenance["rows"], [20, 22])
        self.assertEqual(launch.command[launch.command.index("--manifest_count") + 1], "100")

    def test_tampered_manifest_rejected(self):
        self.manifest.write_text(self.manifest.read_text() + "{}\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            build_client(self.root, self.args)

    def test_missing_assets_rejected(self):
        (self.root / "assets/task2").rmdir()
        with self.assertRaisesRegex(ValueError, "Missing required path"):
            build_client(self.root, self.args)

    def test_external_assets_are_checked_before_launch(self):
        asset = self.root / "assets/task2/ground.usd"
        entry = {"path": "assets/task2/ground.usd", "bytes": 6,
                 "sha256": hashlib.sha256(b"ground").hexdigest()}
        (self.root / "configs/external_assets.json").write_text(json.dumps({"files": [entry]}))
        self.config["external_assets"] = [entry["path"]]
        self.save()
        for contents in (None, b"broken", b"short"):
            if contents is not None:
                asset.write_bytes(contents)
            with self.assertRaisesRegex(ValueError, "download_external_assets.py"):
                build_client(self.root, self.args)
        asset.write_bytes(b"ground")
        launch = build_client(self.root, self.args)
        self.assertEqual(launch.provenance["rows"], [0, 100])

    def test_external_asset_must_be_declared(self):
        (self.root / "configs/external_assets.json").write_text(json.dumps({"files": []}))
        self.config["external_assets"] = ["assets/task2/unknown.usd"]
        self.save()
        with self.assertRaisesRegex(ValueError, "not declared"):
            build_client(self.root, self.args)

    def test_protocol_path_cannot_escape_repository(self):
        self.config["protocol"] = "../external_protocol"
        self.save()
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            build_client(self.root, self.args)

    def test_existing_results_rejected(self):
        self.args.output.mkdir(parents=True)
        (self.args.output / "episodes.csv").write_text("data")
        with self.assertRaisesRegex(ValueError, "Output must"):
            build_client(self.root, self.args)

    def test_invalid_ranges_rejected(self):
        for start, count in ((-1, 1), (100, 1), (1, 0), (95, 6)):
            self.args.row_start, self.args.row_count = start, count
            with self.assertRaises(ValueError):
                build_client(self.root, self.args)

    def test_fixed_options_cannot_override_manifest(self):
        self.config["args"]["--manifest"] = "other.jsonl"
        self.save()
        with self.assertRaisesRegex(ValueError, "launch-owned"):
            build_client(self.root, self.args)

    def test_single_environment_required(self):
        self.config["args"]["--num_envs"] = 8
        self.save()
        with self.assertRaisesRegex(ValueError, "one environment"):
            build_client(self.root, self.args)

    def test_sidecar_integrity(self):
        self.config["integrity"] = [{"path": "configs/task2.jsonl", "sha256": "0" * 64}]
        self.save()
        with self.assertRaisesRegex(ValueError, "Input SHA-256"):
            build_client(self.root, self.args)

    def test_server_launch(self):
        self.args.groot_root = self.root / "groot"
        self.args.qwen_config = None
        self.args.embodiment = "unitree_g1_sonic"
        self.args.device = "cuda:0"
        launch = build_server(self.root, self.args)
        self.assertIn("gr00t.eval.run_gr00t_server", launch.command)
        self.assertIn("--strict", launch.command)
        self.assertTrue(launch.env["PYTHONPATH"].startswith(str(self.args.groot_root)))

    def server_args(self):
        return argparse.Namespace(
            checkpoint=self.checkpoint, groot_root=self.root / "groot", qwen_config=None,
            embodiment="unitree_g1_sonic", device="cuda:0", python=sys.executable,
            host="127.0.0.1", port=5555,
        )

    def test_server_records_checkpoint_files_and_detects_changed_weights(self):
        processor = self.checkpoint / "processor"
        processor.mkdir()
        (self.checkpoint / "processor_config.json").rename(processor / "processor_config.json")
        (processor / "statistics.json").write_text('{"mean": [0]}')
        shard = self.checkpoint / "shards/second.safetensors"
        shard.parent.mkdir()
        shard.write_bytes(b"second shard")
        (self.checkpoint / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"parameter": "shards/second.safetensors"}}))
        args = self.server_args()
        before = build_server(self.root, args).provenance
        files = {entry["path"]: entry for entry in before["checkpoint_files"]}
        expected = {"model.safetensors", "config.json", "model.safetensors.index.json",
                    "processor/processor_config.json", "processor/statistics.json",
                    "shards/second.safetensors"}
        self.assertEqual(set(files), expected)
        for name, entry in files.items():
            content = (self.checkpoint / name).read_bytes()
            self.assertEqual(entry["bytes"], len(content))
            self.assertEqual(entry["sha256"], hashlib.sha256(content).hexdigest())
        (self.checkpoint / "model.safetensors").write_bytes(b"different weights")
        after = build_server(self.root, args).provenance
        changed = {entry["path"]: entry for entry in after["checkpoint_files"]}
        self.assertNotEqual(files["model.safetensors"]["sha256"], changed["model.safetensors"]["sha256"])
        self.assertEqual(files["config.json"], changed["config.json"])

    def run_server_cli(self, *options):
        qwen = self.root / "qwen"
        qwen.mkdir(exist_ok=True)
        (qwen / "config.json").write_text("{}")
        argv = ["start_server.py", "--checkpoint", str(self.checkpoint),
                "--groot-root", str(self.root / "groot"), "--qwen-config", str(qwen), *options]
        with patch("zerohlm_bench.launch.os.chdir"), \
             patch("zerohlm_bench.launch.os.execvpe") as execute, \
             patch.object(sys, "argv", argv):
            result = main("server", self.root)
        return result, execute

    def test_server_cli_saves_unique_records_before_execution(self):
        for _ in range(2):
            result, execute = self.run_server_cli()
            self.assertEqual(result, 0)
            execute.assert_called_once()
        records = list((self.root / "outputs/policy-server").glob("*.json"))
        self.assertEqual(len(records), 2)
        for record in records:
            metadata = json.loads(record.read_text())
            self.assertEqual(metadata["checkpoint"], str(self.checkpoint))
            self.assertTrue(metadata["checkpoint_files"])

    def test_server_cli_explicit_record_never_overwrites(self):
        record = self.root / "records/task2.json"
        self.run_server_cli("--metadata-output", str(record))
        original = record.read_bytes()
        with self.assertRaises(SystemExit) as error:
            self.run_server_cli("--metadata-output", str(record))
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(record.read_bytes(), original)

    def test_server_check_does_not_create_metadata(self):
        record = self.root / "records/check.json"
        for options in (("--check",), ("--check", "--metadata-output", str(record))):
            result, execute = self.run_server_cli(*options)
            self.assertEqual(result, 0)
            execute.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())
        self.assertFalse(record.parent.exists())

    def test_client_does_not_need_checkpoint_weights(self):
        shutil.rmtree(self.checkpoint)
        launch = build_client(self.root, self.args)
        self.assertEqual(launch.provenance["protocol"], str(self.root / "configs/protocol/shared"))

    def test_server_requires_weight_loading_despite_inherited_test_flags(self):
        self.args.groot_root = self.root / "groot"
        self.args.qwen_config = None
        self.args.embodiment = "unitree_g1_sonic"
        self.args.device = "cuda:0"
        for value in ("1", "true", "YES", " on ", "unexpected"):
            with self.subTest(value=value), patch.dict(
                "os.environ", {"GROOT_SKIP_HF_MODEL_WEIGHTS": value, "GROOT_HF_LOCAL_FIRST": "1"}
            ):
                with self.assertRaisesRegex(ValueError, "GROOT_SKIP_HF_MODEL_WEIGHTS"):
                    build_server(self.root, self.args)
        for value in ("", "0", "false", "NO", " off "):
            with self.subTest(value=value), patch.dict(
                "os.environ", {"GROOT_SKIP_HF_MODEL_WEIGHTS": value}
            ):
                self.assertIn("--strict", build_server(self.root, self.args).command)

    def test_server_rejects_missing_weight_shard(self):
        self.args.groot_root = self.root / "groot"
        self.args.qwen_config = None
        self.args.embodiment = "unitree_g1_sonic"
        self.args.device = "cuda:0"
        (self.checkpoint / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"parameter": "model-00002-of-00002.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "Missing required path"):
            build_server(self.root, self.args)

    def test_flags(self):
        self.assertEqual(flag_arguments({"--headless": True, "--disabled": False,
                                        "--count": 2, "--values": [1, 2]}),
                         ["--headless", "--count", "2", "--values", "1", "2"])
        with self.assertRaises(ValueError):
            flag_arguments({"--invalid value": 1})


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "video": {"modality_keys": ["ego_view"], "delta_indices": [0]},
            "state": {"modality_keys": ["left_q", "right_q"], "delta_indices": [0]},
            "action": {"modality_keys": ["sonic_token"], "delta_indices": [0, 1]},
            "language": {"modality_keys": ["task"], "delta_indices": [0]},
        }

    def test_equivalent_wire_representations(self):
        objects = {name: SimpleNamespace(**fields, action_configs=None)
                   for name, fields in self.expected.items()}
        serialized = {name: {"__ModalityConfig__": True, "as_json": fields}
                      for name, fields in self.expected.items()}
        validate_policy_modalities(self.expected, objects)
        validate_policy_modalities(self.expected, serialized)
        validate_policy_modalities(self.expected, dict(reversed(list(self.expected.items()))))

    def test_modality_order_or_history_change_rejected(self):
        for modality, field, value in (("state", "modality_keys", ["right_q", "left_q"]),
                                       ("video", "delta_indices", [-1, 0]),
                                       ("action", "delta_indices", [0])):
            received = json.loads(json.dumps(self.expected))
            received[modality][field] = value
            with self.assertRaisesRegex(ValueError, "differs from the task profile"):
                validate_policy_modalities(self.expected, received)

    def test_missing_and_malformed_modalities_rejected(self):
        missing = dict(self.expected)
        del missing["language"]
        for received in (missing, None, {"video": {"modality_keys": "ego_view", "delta_indices": [0]}}):
            with self.assertRaises(ValueError):
                validate_policy_modalities(self.expected, received)

    def test_mismatched_server_stops_before_rendering(self):
        source = (Path(__file__).resolve().parents[1] / "runtime/scripts/imitation_learning/robomimic/evaluation_runner.py").read_text()
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "run_checkpoint")
        code = compile(ast.Module(body=[function], type_ignores=[]), "run_checkpoint", "exec",
                       flags=__future__.annotations.compiler_flag)
        client = Mock()
        received = json.loads(json.dumps(self.expected))
        received["video"]["delta_indices"] = [-1, 0]
        client.get_modality_config.return_value = received
        renderer = Mock()
        adapter = SimpleNamespace(pending_rows=lambda *args: [1],
                                  load_processor_modality_configs=lambda *args: self.expected,
                                  robot_inference_client=lambda *args: client,
                                  init_gs_renderer=renderer)
        namespace = {"Path": Path, "validate_policy_modalities": validate_policy_modalities,
                     "contextlib": contextlib}
        exec(code, namespace)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"torch": Mock()}):
            args = SimpleNamespace(output_root=Path(directory), model_name="sim", resume=False,
                                   checkpoint_dir=Path(directory), embodiment="unitree_g1_sonic",
                                   gr00t_host="localhost", gr00t_port=5555, gr00t_timeout_ms=1000)
            with self.assertRaisesRegex(ValueError, "differs from the task profile"):
                namespace["run_checkpoint"](args, {}, [1], "hash", 1, None, adapter)
        renderer.assert_not_called()
        client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
