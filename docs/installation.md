# 🛠️ Simulator Installation

Use separate Python environments for the simulation client and [policy server](../integrations/groot/README.md). Run the following steps in the simulation environment.

## Requirements

| Component | Version |
| --- | --- |
| Platform | Linux x86-64 with an NVIDIA GPU |
| Python | 3.10 |
| Isaac Sim | 4.5.0 |
| Isaac Lab | Revision `0b826f7297c617b8f2741c0a14764f68b9a07d0f` |
| PyTorch / torchvision | 2.5.1 / 0.20.1, CUDA 11.8 |
| Build tools | CUDA Toolkit 11.8 (`nvcc`), GCC 11 |
| Video tools | FFmpeg and ffprobe |

Evaluation uses an RTX 4090. See the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/v2.0.0/source/setup/installation/pip_installation.html) for driver requirements and NVIDIA license acceptance.

## 1. Install Isaac Lab

On Ubuntu 22.04, install the graphics libraries required even for headless rendering:

```bash
sudo apt-get install libglu1-mesa libopengl0
```

In a Python 3.10 environment:

```bash
python -m pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 0b826f7297c617b8f2741c0a14764f68b9a07d0f
./isaaclab.sh --install none
```

## 2. Install the Benchmark

Return to the Zero-HLM-Bench repository root. Set `CUDA_HOME` to your CUDA 11.8 toolkit; the PyTorch wheel alone does not provide the compiler.

```bash
python scripts/download_external_assets.py
bash scripts/setup_simulator.sh --check
bash scripts/setup_simulator.sh
```

These commands download the required NVIDIA assets, check prerequisites, and install benchmark dependencies and rendering extensions. Isaac Sim, Isaac Lab, and PyTorch must already be installed.

When building without a visible GPU, specify the target architecture. For RTX 4090:

```bash
TORCH_CUDA_ARCH_LIST=8.9 bash scripts/setup_simulator.sh
```

## 3. Verify and Run

```bash
python -c 'import torch; import diff_surfel_rasterization; import simple_knn._C; print("Rendering extensions ready")'
```

Continue with the [two-episode evaluation](server_client.md#quick-check) to verify the full pipeline. The first simulator startup may take several minutes to build shader caches.

Third-party code and assets retain their own terms, including research and non-commercial restrictions for Gaussian-rendering components. See [Third-Party Notices](../THIRD_PARTY_NOTICES.md).
