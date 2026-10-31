# Third-Party Notices

Each component retains its own license. A license for one component does not change the terms of another component, model, or asset.

| Component | Location | License file |
| --- | --- | --- |
| Isaac Lab-derived environment and manager code | `runtime/envs`, `runtime/managers`, `runtime/isaaclab_utils` | `licenses/IsaacLab-BSD-3-Clause.txt` |
| NVIDIA GR00T inference integration | `integrations/groot/gr00t` | `integrations/groot/LICENSE` |
| 2D Gaussian Splatting | `runtime/submodule/2D_GS/2d-gaussian-splatting` | Component `LICENSE.md` |
| Differentiable surfel rasterization | `third_party/diff-surfel-rasterization` | Component `LICENSE.md` |
| Simple KNN | `third_party/simple-knn` | Component `LICENSE.md` |
| GLM | `third_party/diff-surfel-rasterization/third_party/glm` | Component `copying.txt` |
| Unitree G1 assets | Robot assets under `assets/task1`, `assets/task2`, `assets/task3` | `assets/shared/licenses/Unitree-G1-BSD-3-Clause.txt` |
| NVIDIA GEAR-SONIC controller and model assets | Controller ONNX files under each task | `assets/shared/licenses/GEAR-SONIC-LICENSE.txt` and `GEAR-SONIC-Asset-Notices.txt` |

The Gaussian-rendering components include research and non-commercial restrictions. Isaac Sim and downloaded policy checkpoints are separate dependencies with their own terms.

Cosmos-Reason2 configuration and tokenizer resources retain the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/), reproduced in `assets/shared/licenses/Cosmos-Reason2-NVIDIA-Open-Model-License.md`. Their upstream source is [NVIDIA Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B).

GEAR-SONIC model assets and Cosmos-Reason2 resources: Licensed by NVIDIA Corporation under the NVIDIA Open Model License. Built on NVIDIA Cosmos.

NVIDIA Grid files are obtained by `scripts/download_external_assets.py` directly from NVIDIA's versioned asset server. They are excluded from source packages and remain subject to the terms supplied by NVIDIA; the Isaac Lab code license does not cover those assets. See NVIDIA's [asset licensing FAQ](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/common/license-faq.html) and [additional software and materials terms](https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/).

The separately downloaded [Zero-HLM-Bench GR00T checkpoints](https://huggingface.co/YDY0427/Zero-HLM-Bench-GR00T) retain the upstream model [NVIDIA LICENSE](https://huggingface.co/YDY0427/Zero-HLM-Bench-GR00T/blob/main/LICENSE), including its non-commercial research and evaluation limitation. The integration code license does not replace the model-weight license.
