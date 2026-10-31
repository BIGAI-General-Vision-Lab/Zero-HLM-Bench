#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SIM_PYTHON=${SIM_PYTHON:-python}
case "${1:-}" in
  ""|--check) ;;
  *) printf 'Usage: SIM_PYTHON=python bash scripts/setup_simulator.sh [--check]\n' >&2; exit 2 ;;
esac
"$SIM_PYTHON" - <<'PY'
import importlib.metadata as metadata
import re
import shutil
import subprocess
import sys
from pathlib import Path

if sys.version_info[:2] != (3, 10):
    raise SystemExit("Python 3.10 is required for the validated Isaac Sim environment.")
for name, expected in {"isaacsim": "4.5.0.0", "isaaclab": "0.40.22", "torch": "2.5.1+cu118"}.items():
    actual = metadata.version(name)
    if actual != expected:
        raise SystemExit(f"Expected {name}=={expected}; found {actual}. See docs/installation.md.")
import torch
from torch.utils.cpp_extension import CUDA_HOME
if CUDA_HOME is None:
    raise SystemExit("Install CUDA Toolkit 11.8 and set CUDA_HOME to its installation directory.")
nvcc = Path(CUDA_HOME) / "bin/nvcc"
if not nvcc.is_file():
    raise SystemExit("CUDA_HOME must contain bin/nvcc.")
version = subprocess.check_output([str(nvcc), "--version"], text=True)
match = re.search(r"release (\d+\.\d+)", version)
if match is None or match.group(1) != torch.version.cuda:
    raise SystemExit(f"CUDA compiler must match PyTorch CUDA {torch.version.cuda}.")
for program in ("g++", "ffmpeg", "ffprobe"):
    if shutil.which(program) is None:
        raise SystemExit(f"Required executable is missing from PATH: {program}")
print("Simulator, PyTorch, CUDA compiler, and video tools match the validated prerequisites.")
PY
if [[ ${1:-} == --check ]]; then
  exit 0
fi
"$SIM_PYTHON" -m pip install -r "$RELEASE_ROOT/requirements/simulator.txt"
"$SIM_PYTHON" -m pip install --no-build-isolation --no-deps \
  "$RELEASE_ROOT/third_party/diff-surfel-rasterization" \
  "$RELEASE_ROOT/third_party/simple-knn"
"$SIM_PYTHON" - <<'PY'
import torch
import diff_surfel_rasterization
import simple_knn._C
print("Gaussian rasterization extensions imported successfully.")
PY
