# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, List, Optional

from gr00t.data.types import ModalityConfig

from .embodiment_configs import MODALITY_CONFIGS


@dataclass
class SingleDatasetConfig:
    """Configuration for a single dataset in a mixed-training setup.

    A list of these objects can be supplied in ``DataConfig.datasets`` to mix
    multiple datasets at arbitrary ratios.  For convenience the *legacy*
    single-dataset fields still exist; if ``datasets`` is non-empty they take
    precedence.
    """

    
    dataset_paths: List[Any]

    
    embodiment_tag: Optional[str] = None

    
    mix_ratio: float = 1.0

    dataset_type: str = "physical_embodiment"

    
    
    val_dataset_path: Optional[str] = None


@dataclass
class DataConfig:
    """Dataset configuration (supports single or multiple datasets)."""

    
    
    
    datasets: List[SingleDatasetConfig] = field(default_factory=list)

    
    
    
    
    
    
    modality_configs: dict[str, dict[str, ModalityConfig]] = field(
        default_factory=lambda: MODALITY_CONFIGS
    )

    
    download_cache: bool = False
    shard_size: int = 2**10
    episode_sampling_rate: float = 0.1
    num_shards_per_epoch: int = int(1e5)

    
    override_pretraining_statistics: bool = True

    
    mode: str = "single_turn"
    random_chop: float = 0.0
    mock_dataset_mode: bool = False  

    
    shuffle: bool = True
    seed: int = 42
    multiprocessing_context: str = "fork"  
    allow_padding: bool = False

    
    subsample_ratio: float = 1.0

    
    image_crop_size: List[int] = field(default_factory=lambda: [244, 244])
    image_target_size: List[int] = field(default_factory=lambda: [224, 224])
    video_backend: str = "torchcodec"
