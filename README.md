<div align="center">

# 🤖 Zero-HLM-Bench

### A Simulation Benchmark for Humanoid Loco-Manipulation

[**📄 Paper**](https://arxiv.org/abs/2610.35075) · [**🌐 Project Page**](https://young-bit.github.io/zero-hlm/) · [**🤗 Dataset**](https://huggingface.co/datasets/YDY0427/Zero-HLM-Bench)

[![Zero-HLM: paper overview followed by four-policy comparisons across three tasks](docs/media/benchmark-overview.gif)](https://young-bit.github.io/zero-hlm/#benchmark)

*Selected evaluation rollouts from Zero-HLM: four policies, three tasks, and three configurations per task. Gold indicates success; red indicates failure. Playback is condensed for this overview.*

[Video](docs/media/benchmark-overview.mp4) · [Installation](#installation) · [Training Data](#training-data) · [Evaluation](#evaluation) · [Custom Policies](#custom-policies) · [Citation](#citation)

</div>

<a id="overview"></a>

## 🔎 Overview

**Zero-HLM-Bench** evaluates language-conditioned humanoid loco-manipulation in reconstructed environments. It combines scene collision geometry, Gaussian-rendered observations, and SONIC whole-body control on a Unitree G1. Models are compared using fixed initial configurations and shared task-level success criteria.

The benchmark accompanies [Zero-HLM: Generalizable Humanoid Loco-Manipulation without Real Robot Data](https://arxiv.org/abs/2610.35075).

**📦 Current release:** **three tasks**, **100 evaluation configurations per task**, and **800 simulation training demonstrations per task**, with a separate policy server and simulation client.

<a id="tasks"></a>

## 🎯 Tasks

| Task | Instruction | Whole-Body Skills |
| --- | --- | --- |
| **1 · Mobile Pick-and-Place** | Walk to the table, pick up the banana, and place it in the basket. | Locomotion, grasping, and placement |
| **2 · Low-Reach Hat Placement** | Pick up the hat, carry it to the cabinet, and place it on top. | Low reaching, transport, and elevated placement |
| **3 · Leg-Assisted Basket Transport** | Carry the basket, close the cabinet with a foot, and place the basket on the stool. | Bimanual transport, leg-assisted interaction, and placement |

Each task has a fixed 100-configuration manifest. The client verifies its integrity before evaluation and records each rollout's outcome. See the [evaluation protocol](docs/evaluation.md) for sampling ranges, control settings, and success criteria.

<a id="installation"></a>

## 🛠️ Installation

Clone the repository with [Git LFS](https://git-lfs.com/) installed to retrieve the scene assets:

```bash
git clone https://github.com/BIGAI-General-Vision-Lab/Zero-HLM-Bench.git
cd Zero-HLM-Bench
git lfs pull
```

Use separate Python environments for the **simulation client** and **policy server**. The simulator uses Python 3.10, Isaac Sim 4.5, Isaac Lab, and CUDA 11.8; the exact versions and system dependencies are listed in the [installation guide](docs/installation.md).

**Simulation environment** — after installing Isaac Lab and the required CUDA toolchain:

```bash
python scripts/download_external_assets.py
bash scripts/setup_simulator.sh --check
bash scripts/setup_simulator.sh
```

**Policy environment** — after installing the matching PyTorch build:

```bash
python scripts/setup_groot.py --with-dependencies
```

The included adapter supports **GR00T N1.7**. Follow the [policy environment guide](integrations/groot/README.md) for its dependencies and checkpoint layout.

<a id="training-data"></a>

## 📦 Training Data

Download all three datasets from [Hugging Face](https://huggingface.co/datasets/YDY0427/Zero-HLM-Bench):

```bash
python scripts/download_data.py --task all --destination data
```

| Dataset | Demonstrations | Frames | Download |
| --- | ---: | ---: | ---: |
| Task 1 | 800 | 648,370 | 2.79 GB |
| Task 2 | 800 | 750,245 | 3.02 GB |
| Task 3 | 800 | 963,304 | 2.59 GB |
| **Total** | **2,400** | **2,361,919** | **8.40 GB** |

Datasets use **LeRobot v2.1** with 640 × 480 RGB observations, 34-dimensional robot states, and 66-dimensional actions at 50 Hz. Actions comprise a 64-dimensional SONIC latent command and two hand commands.

Use `--task task1`, `task2`, or `task3` to download a single task. Downloads are pinned to an immutable dataset revision and verified with SHA-256 before extraction. Allow approximately **18 GB** for all archives and extracted data. The released 800-episode training subsets are identified separately from the training selections used by existing checkpoints.

<a id="evaluation"></a>

## 📊 Evaluation

The **server runs policy inference**; the **client handles simulation, rendering, control, and scoring**. With a trained Task 2 checkpoint in `checkpoints/task2`, start the following commands from the repository root.

**Terminal 1 · Policy environment**

```bash
python scripts/start_server.py --checkpoint checkpoints/task2 --port 5555
```

**Terminal 2 · Simulation environment**

```bash
python scripts/start_client.py --task task2 --port 5555 --output outputs/task2
```

This evaluates all **100 configurations** for Task 2. For Tasks 1 and 3, select the corresponding `--task` and checkpoint. Use `--host` on the client to connect to a server on another machine, and `--device` to choose the simulation GPU.

For a short two-episode check:

```bash
python scripts/start_client.py \
  --task task2 --port 5555 \
  --row-start 0 --row-count 2 --output outputs/task2-smoke
```

Results contain per-episode success checks, termination reasons, rendered observations, and configuration hashes. Report successful episodes out of the **100 fixed configurations** for each task. See the [server/client guide](docs/server_client.md) for launch options and output details.

<a id="custom-policies"></a>

## 🔌 Custom Policies

To evaluate another model, implement its inference on the **server side** while keeping the simulation client and task configuration unchanged.

| Interface | Content |
| --- | --- |
| **Observations** | Egocentric RGB, 34-dimensional robot state, and task instruction |
| **Actions** | 64-dimensional SONIC latent commands and two hand commands |
| **Transport** | ZeroMQ REQ/REP with MessagePack serialization |
| **Endpoints** | `ping`, `reset`, `get_modality_config`, `get_action` |

The [integration specification](docs/server_client.md#adapting-another-model) defines array shapes, modality keys, history indices, and action chunks. Keep manifests, cameras, control settings, and success criteria fixed across policy comparisons.

<a id="repository-structure"></a>

## 📂 Repository Structure

```text
configs/              Task settings, evaluation manifests, and interface schemas
assets/               Task 1–3 scenes and shared rendering/controller resources
src/zerohlm_bench/     Launch and configuration utilities
runtime/              Simulation, rendering, control, and task success checks
integrations/groot/   GR00T inference adapter
scripts/              Setup, data download, and server/client entry points
docs/                 Installation and evaluation guides
tests/                Configuration, interface, and packaging tests
```

<a id="todo"></a>

## 📋 TODO

- [ ] Release evaluation results for all benchmark models.
- [ ] Release datasets and model evaluation results under Extended Settings.

<a id="citation"></a>

## 📚 Citation

```bibtex
@misc{yang2026zerohlm,
  title         = {Zero-HLM: Generalizable Humanoid Loco-Manipulation
                   without Real Robot Data},
  author        = {Dianyi Yang and Mengya Liu and Zhumei Wang and Yu Liu
                   and Xiaochen Zhang and Yixuan Li and Le Ma and Yuhang Yang
                   and Weiqi Huang and Wei Liang and Baoxiong Jia and Siyuan Huang},
  year          = {2026},
  eprint        = {2610.35075},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.35075}
}
```

<a id="acknowledgments-and-licensing"></a>

## 🤝 Acknowledgments and Licensing

The benchmark builds on Isaac Sim / Isaac Lab, SONIC, Gaussian Splatting, NVIDIA Isaac-GR00T, and LeRobot. Third-party code and assets retain their respective licenses; see [Third-Party Notices](THIRD_PARTY_NOTICES.md) and the accompanying license files.
