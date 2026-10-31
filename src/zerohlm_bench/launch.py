"""Validate benchmark inputs and construct simulator and policy processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class Launch:
    command: list[str]
    env: dict[str, str]
    cwd: Path
    provenance: dict


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def require(path: Path) -> Path:
    if not path.exists():
        raise ValueError(f"Missing required path: {path}")
    return path


def repository_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a repository-relative path: {value}")
    return root / path


def expand(value, root: Path):
    if isinstance(value, str):
        return value.format(root=str(root))
    if isinstance(value, list):
        return [expand(item, root) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, root) for key, item in value.items()}
    return value


def flag_arguments(values: dict) -> list[str]:
    result = []
    for flag, value in values.items():
        if not re.fullmatch(r"--[a-zA-Z][\w-]*", flag):
            raise ValueError(f"Invalid option: {flag}")
        if value is False:
            continue
        result.append(flag)
        if value is True:
            continue
        parts = value if isinstance(value, list) else [value]
        if not parts or any(type(part) not in (str, int, float) for part in parts):
            raise ValueError(f"Invalid value for {flag}: {value!r}")
        result.extend(str(part) for part in parts)
    return result


def python_executable(value: str) -> str:
    executable = shutil.which(value)
    if executable is None:
        raise ValueError(f"Python executable not found: {value}")
    return executable


def checkpoint_config(checkpoint: Path, embodiment: str) -> dict:
    processor = checkpoint / "processor_config.json"
    # Final-only GR00T exports store the same processor schema in processor/.
    if not processor.is_file():
        processor = checkpoint / "processor" / "processor_config.json"
    payload = json.loads(require(processor).read_text())
    try:
        modalities = payload["processor_kwargs"]["modality_configs"][embodiment]
        for key in ("video", "state", "language", "action"):
            if not modalities[key]["modality_keys"] or not modalities[key]["delta_indices"]:
                raise ValueError(f"Empty {key} modality")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid processor_config.json for {embodiment}") from exc
    return modalities


def validate_external_assets(root: Path, paths: list[str]) -> None:
    if not paths:
        return
    manifest = json.loads(require(root / "configs" / "external_assets.json").read_text())
    entries = {item["path"]: item for item in manifest["files"]}
    for name in paths:
        if name not in entries:
            raise ValueError(f"External asset is not declared in configs/external_assets.json: {name}")
        path = repository_path(root, name)
        entry = entries[name]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Missing or invalid external asset: {name}. "
                             "Run python scripts/download_external_assets.py before evaluation.")


def validate_scene_assets(root: Path, spec: dict) -> None:
    manifest = require(repository_path(root, spec["path"]))
    if digest(manifest) != spec["sha256"]:
        raise ValueError(f"Scene manifest SHA-256 mismatch: {manifest}")
    for entry in json.loads(manifest.read_text())["files"]:
        path = repository_path(root, entry["path"])
        if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Scene asset missing or invalid: {entry['path']}. "
                             "Restore the scene assets with git lfs pull before evaluation.")


def build_client(root: Path, args: argparse.Namespace) -> Launch:
    root = root.resolve()
    config_path = require(root / "configs" / f"{args.task}.json")
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != 1 or config.get("task") != args.task:
        raise ValueError("Task config schema or task name does not match")
    manifest_spec = config["manifest"]
    manifest = require(repository_path(root, manifest_spec["path"]))
    if digest(manifest) != manifest_spec["sha256"]:
        raise ValueError(f"Manifest SHA-256 mismatch: {manifest}")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    total = manifest_spec["rows"]
    if len(rows) != total or not isinstance(total, int) or total <= 0:
        raise ValueError("Manifest row count does not match the task config")
    stop = total if args.row_count is None else args.row_start + args.row_count
    if not 0 <= args.row_start < stop <= total:
        raise ValueError(f"Requested rows [{args.row_start}, {stop}) exceed {total} rows")
    for item in config.get("integrity", []):
        path = require(repository_path(root, item["path"]))
        if digest(path) != item["sha256"]:
            raise ValueError(f"Input SHA-256 mismatch: {path}")
    validate_external_assets(root, config.get("external_assets", []))
    if "scene_assets" in config:
        validate_scene_assets(root, config["scene_assets"])
    resolved = expand(config, root)
    for path in resolved.get("required_paths", []):
        require(Path(path))
    evaluator = require(repository_path(root / "runtime", config["evaluator"]))
    fixed = resolved["args"]
    if args.device is not None:
        fixed["--device"] = args.device
    reserved = {"--checkpoint_dir", "--output_root", "--gr00t_host", "--gr00t_port",
                "--manifest", "--manifest_count", "--manifest_row_start", "--manifest_row_stop",
                "--max_manifest_rows", "--generate_manifest", "--resume"}
    if reserved.intersection(fixed):
        raise ValueError(f"Config contains launch-owned options: {sorted(reserved.intersection(fixed))}")
    if fixed.get("--num_envs") != 1 or fixed.get("--fallback_env_counts") != [1]:
        raise ValueError("The benchmark requires exactly one environment per client")
    if fixed.get("--no-resume") is not True:
        raise ValueError("The benchmark requires --no-resume")
    checkpoint = require(repository_path(root, config["protocol"]))
    modalities = checkpoint_config(checkpoint, fixed["--embodiment"])
    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    dynamic = {"--checkpoint_dir": str(checkpoint), "--output_root": str(output),
               "--gr00t_host": args.host, "--gr00t_port": args.port,
               "--manifest": str(manifest), "--manifest_count": total,
               "--manifest_row_start": args.row_start, "--manifest_row_stop": stop}
    command = [python_executable(args.python), "-u", str(evaluator)]
    command += flag_arguments(fixed) + flag_arguments(dynamic)
    env = os.environ.copy()
    env.update({key: str(value) for key, value in resolved.get("env", {}).items()})
    python_paths = [str(root / "runtime"), str(root / "src")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHONUNBUFFERED"] = "1"
    provenance = {"task": args.task, "config_sha256": digest(config_path),
                  "manifest_sha256": digest(manifest), "rows": [args.row_start, stop],
                  "protocol": str(checkpoint), "processor_sha256": digest(checkpoint / "processor_config.json"),
                  "endpoint": f"tcp://{args.host}:{args.port}", "command": command,
                  "config": resolved, "modalities": modalities}
    return Launch(command, env, root / "runtime", provenance)


def build_server(root: Path, args: argparse.Namespace) -> Launch:
    # GR00T's shape-test mode constructs a model without checkpoint weights.
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS", "").strip().lower() not in {"", "0", "false", "no", "off"}:
        raise ValueError("Unset GROOT_SKIP_HF_MODEL_WEIGHTS: evaluation requires trained model weights.")
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_config(checkpoint, args.embodiment)
    require(checkpoint / "config.json")
    weights = set(checkpoint.glob("*.safetensors")) | set(checkpoint.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError(f"No model weights found in {checkpoint}")
    for index_path in checkpoint.glob("*.index.json"):
        index = json.loads(index_path.read_text())
        for filename in set(index.get("weight_map", {}).values()):
            weights.add(require(repository_path(checkpoint, filename)))
    checkpoint_files = [
        {"path": path.relative_to(checkpoint).as_posix(), "bytes": path.stat().st_size,
         "sha256": digest(path)}
        for path in sorted(weights | set(checkpoint.rglob("*.json"))) if path.is_file()
    ]
    env = os.environ.copy()
    python = python_executable(args.python)
    groot_root = args.groot_root.expanduser().resolve() if args.groot_root else root / "integrations" / "groot"
    require(groot_root / "gr00t" / "eval" / "run_gr00t_server.py")
    if args.groot_root is None:
        manifest = json.loads(require(groot_root / "source_manifest.json").read_text())
        for entry in manifest["files"]:
            if digest(require(groot_root / entry["path"])) != entry["sha256"]:
                raise ValueError(f"Bundled GR00T source checksum mismatch: {entry['path']}")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(groot_root), env.get("PYTHONPATH")]))
    if args.qwen_config is not None:
        qwen_config = require(args.qwen_config.expanduser().resolve())
        require(qwen_config / "config.json")
        env["GR00T_QWEN3_INIT_CONFIG_PATH"] = str(qwen_config)
    command = [python, "-u", "-m", "gr00t.eval.run_gr00t_server",
               "--model-path", str(checkpoint), "--embodiment-tag", args.embodiment,
               "--device", args.device, "--host", args.host, "--port", str(args.port), "--strict"]
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(),
                  "checkpoint": str(checkpoint), "checkpoint_files": checkpoint_files,
                  "groot_root": str(groot_root), "command": command}
    return Launch(command, env, root, provenance)


def cuda_device(value: str) -> str:
    if not re.fullmatch(r"cuda:[0-9]+", value):
        raise argparse.ArgumentTypeError("Device must use cuda:N with a non-negative GPU index")
    return value


def port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535")
    return port


def main(kind: str, root: Path) -> int:
    parser = argparse.ArgumentParser(description=f"Start the benchmark {kind}.")
    parser.add_argument("--host", default="127.0.0.1" if kind == "client" else "0.0.0.0")
    parser.add_argument("--port", type=port_number, default=5555)
    parser.add_argument("--python", default=sys.executable, help="Python executable for the selected runtime")
    parser.add_argument("--check", action="store_true", help="Validate local inputs and print the command without starting a process")
    if kind == "client":
        parser.add_argument("--task", choices=("task1", "task2", "task3"), required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--row-start", type=int, default=0)
        parser.add_argument("--row-count", type=int)
        parser.add_argument("--device", type=cuda_device, help="GPU device for simulation and rendering, such as cuda:0")
    else:
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--groot-root", type=Path)
        parser.add_argument("--metadata-output", type=Path, help="New JSON path for the checkpoint record; defaults to a unique file in outputs/policy-server")
        parser.add_argument("--qwen-config", type=Path, default=root / "assets" / "shared" / "Cosmos-Reason2-2B-qwen3vl-files", help="Local Cosmos-Reason2/Qwen3-VL config and processor files")
        parser.add_argument("--embodiment", default="unitree_g1_sonic")
        parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        if kind == "client":
            args.output = args.output.expanduser().resolve()
        elif args.metadata_output is not None:
            args.metadata_output = args.metadata_output.expanduser().resolve()
            if args.metadata_output.exists():
                raise ValueError(f"Metadata output already exists: {args.metadata_output}")
        launch = build_client(root, args) if kind == "client" else build_server(root, args)
        if args.check:
            print(json.dumps(launch.provenance, indent=2))
            return 0
        if kind == "client":
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "launch.json").write_text(json.dumps(launch.provenance, indent=2) + "\n")
        else:
            metadata = args.metadata_output or root / "outputs" / "policy-server" / f"server-{uuid4().hex}.json"
            metadata.parent.mkdir(parents=True, exist_ok=True)
            with metadata.open("x") as stream:
                stream.write(json.dumps(launch.provenance, indent=2) + "\n")
            print(f"Policy server checkpoint record: {metadata}", flush=True)
        os.chdir(launch.cwd)
        os.execvpe(launch.command[0], launch.command, launch.env)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    return 0
