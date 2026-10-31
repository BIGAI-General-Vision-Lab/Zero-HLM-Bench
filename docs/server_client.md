# 🔌 Policy Server and Evaluation Client

The **server runs model inference**; the **client handles simulation, rendering, control, and scoring**. Start one of each, either on the same machine or on two machines connected over TCP.

## Evaluate GR00T N1.7

Prepare the [policy environment](../integrations/groot/README.md) and [simulation environment](installation.md). Run both commands from the repository root.

Download the released Task 2 checkpoint:

```bash
python scripts/download_models.py --task task2
```

Use `--task all` for all three tasks. Each checkpoint includes weights, the shard index, model configuration, and processor configuration, normalization statistics, and embodiment mapping. Task 3 retains these processor files in `processor/`; the launcher supports both layouts.

**Terminal 1 · Policy environment**

```bash
python scripts/start_server.py \
  --checkpoint checkpoints/task2 --host 0.0.0.0 --port 5555
```

**Terminal 2 · Simulation environment**

```bash
python scripts/start_client.py \
  --task task2 --host 127.0.0.1 --port 5555 --output outputs/task2
```

This evaluates all **100 configurations**. For Tasks 1 and 3, change `--task` and use the corresponding checkpoint. The client does not need checkpoint weights.

## Quick Check

Run two configurations before a full evaluation:

```bash
python scripts/start_client.py \
  --task task2 --host 127.0.0.1 --port 5555 \
  --row-start 0 --row-count 2 --output outputs/task2-smoke
```

Use a new or empty output directory for each run. Results include episode outcomes, aggregate metrics, and `launch.json` with the evaluated configuration. Append `--check` to either launcher to check local inputs without starting the server or simulator.

## Checkpoint Records

Each server launch saves a unique JSON record under `outputs/policy-server/` and prints its path before starting inference. The record includes checkpoint-relative filenames, byte counts, and SHA-256 hashes for model weights and checkpoint JSON files, including processor settings and shard indices. Use `--metadata-output records/task2-server.json` to select a new file; existing files are never overwritten. `--check` prints the record without creating files. Hashing large checkpoints adds some startup time.

Keep the server record together with the client's `launch.json` and episode results. The client records the task configuration and observation/action protocol; the server record identifies the model weights. Keep the checkpoint unchanged while the server runs.

## Launch Options

| Option | Use |
| --- | --- |
| Client `--host` | Address of the policy server; use `127.0.0.1` on one machine. |
| Client `--device cuda:N` | Select the simulation GPU for rendering and physics. Keep devices visible when selecting a physical GPU index. |
| Server `CUDA_VISIBLE_DEVICES=N` | Select the GPU available to model inference. |
| `--row-start`, `--row-count` | Select consecutive manifest rows, starting from a zero-based index. |
| `--python` | Use a different Python interpreter. |

Task settings are in [configs](../configs); sampling and scoring are described in the [evaluation protocol](evaluation.md). Keep these settings unchanged across model comparisons.

## Adapting Another Model

Implement model inference on the **server side**. Reuse the existing client and convert the model's output into SONIC and hand commands.

The interface uses **ZeroMQ REQ/REP** with **MessagePack** and **MessagePack-NumPy** array serialization. Requests contain an `endpoint` and, when needed, a `data` mapping.

| Endpoint | Request data | Response |
| --- | --- | --- |
| `ping` | absent | Status mapping |
| `reset` | `{"options": null}` | Status mapping; clear temporal state |
| `get_modality_config` | absent | `unitree_g1_sonic` modality mapping |
| `get_action` | `{"observation": observation, "options": null}` | `[action, info]` |

| Input / Output | Format |
| --- | --- |
| `video.ego_view` | RGB uint8, shape `[1, T, H, W, 3]` |
| `state` | 34-dimensional float32 state, split into modality fields of shape `[1, T, D]` |
| `language` | Task instruction in the task's language modality field |
| `sonic_token` | 64-dimensional action, shape `[1, T, 64]` |
| `left_hand_gripper`, `right_hand_gripper` | One command per hand, each shape `[1, T, 1]` |

Match the modality keys, order, and history indices in the [shared protocol](../configs/protocol/shared/processor_config.json). Return at least **16 action steps for Task 1** and **40 for Tasks 2 and 3**. Return an `error` mapping when inference fails.

The [GR00T integration](../integrations/groot) provides a working server implementation.
