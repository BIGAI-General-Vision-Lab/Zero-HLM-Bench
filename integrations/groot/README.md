# 🤖 GR00T N1.7 Policy Server

This integration provides GR00T N1.7 inference for the Unitree G1 with SONIC actions. It is derived from [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T); upstream license notices are retained in the source and [LICENSE](LICENSE).

## Install

Use a separate Python 3.10 environment from the simulator. From the benchmark repository root, install PyTorch for your CUDA runtime, then the integration. For CUDA 11.8:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu118
python scripts/setup_groot.py --with-dependencies
```

Verify the installation:

```bash
python scripts/setup_groot.py --check
```

Flash Attention is optional; the server uses PyTorch SDPA when it is absent.

## Run

Provide a trained checkpoint containing model weights, `config.json`, and `processor_config.json`:

```bash
python scripts/start_server.py --checkpoint checkpoints/task2 --port 5555
```

The launcher supplies the bundled backbone configuration and processor files. Inference weights come from your checkpoint.

Start the simulation client using the [evaluation guide](../../docs/server_client.md). This package supports inference; use an upstream GR00T checkout for training.
