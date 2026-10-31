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
from typing import Optional


@dataclass
class TrainingConfig:
    """Training configuration."""

    
    output_dir: str = "./outputs"
    experiment_name: Optional[str] = None

    
    max_steps: int = 30000  
    global_batch_size: int = 1024
    batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1

    
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    warmup_steps: int = 0  
    max_grad_norm: float = 1.0

    
    
    
    optim: str = "adamw_torch_fused"

    start_from_checkpoint: Optional[str] = None
    skip_weight_loading: bool = False  

    
    tf32: bool = True
    fp16: bool = False
    bf16: bool = True
    eval_bf16: bool = True

    
    logging_steps: int = 10
    save_steps: int = 1000
    save_total_limit: int = 5

    
    save_vl_model: bool = False  
    save_only_model: bool = False  

    
    upload_checkpoints: bool = False
    upload_every: int = 1000
    upload_last_n_checkpoints: int = 5
    max_concurrent_uploads: int = 2

    
    eval_strategy: str = "no"  
    eval_steps: int = 500
    eval_set_split_ratio: float = 0.1
    eval_batch_size: int = 2
    save_best_eval_metric_name: str = ""
    save_best_eval_metric_greater_is_better: bool = True

    
    deepspeed_stage: int = 2  
    gradient_checkpointing: bool = False

    
    transformers_trust_remote_code: bool = True
    transformers_local_files_only: bool = False
    transformers_cache_dir: str | None = None
    transformers_access_token: str | None = None  

    
    use_ddp: bool = False
    ddp_bucket_cap_mb: int = 100

    
    num_gpus: int = 1
    dataloader_num_workers: int = 2

    
    remove_unused_columns: bool = False

    
    use_wandb: bool = False
    wandb_project: str = "finetune-gr00t-n1d7"

    
    enable_profiling: bool = False

    
    max_retries: int = 3

    
    assert_loss_less_than: float | None = None

    
    add_rl_callback: bool = False

    
    enable_open_loop_eval: bool = False
    """Enable open-loop evaluation on saved checkpoints."""

    open_loop_eval_traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    open_loop_eval_steps_per_traj: int = 100
    """Number of steps to evaluate per trajectory."""

    open_loop_eval_plot_indices: Optional[list[int]] = None
    """List of action indices to plot. If None, plots all indices."""
