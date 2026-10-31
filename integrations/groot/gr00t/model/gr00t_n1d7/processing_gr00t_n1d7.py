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

from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import random
import re
from typing import Any, Dict
import warnings

import albumentations as A
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import cached_file

from gr00t.configs.data.embodiment_configs import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.utils import parse_modality_configs, to_json_serializable

from .image_augmentations import (
    apply_with_replay,
    build_image_transformations,
    build_image_transformations_albumentations,
)


try:
    from transformers import Qwen3VLProcessor
except ImportError:
    Qwen3VLProcessor = None


warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")

logger = logging.getLogger(__name__)


EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    
    "oxe_droid_relative_eef_relative_joint": 24,
    "xdof_relative_eef_relative_joint": 27,
    "xdof_relative_eef_relative_joint_subtask": 27,
    "real_g1_relative_eef_relative_joints": 25,
    "real_r1_pro_sharpa_relative_eef": 26,
    "real_r1_pro_sharpa_relative_eef_human": 26,
    "real_r1_pro_sharpa_relative_eef_maxinsights": 26,
    "real_r1_pro_sharpa_relative_eef_mecka": 26,
    
    "unitree_g1_full_body_with_waist_height_nav_cmd": 25,
    "unitree_g1_sonic": 11,
    "simpler_env_google": 0,
    "simpler_env_widowx": 1,
    "libero_sim": 2,
    "new_embodiment": 10,
    "robocasa_panda_omron": 10,
    "robocasa_gr1_tabletop": 10,
}


def build_processor(model_name: str, transformers_loading_kwargs: dict) -> Qwen3VLProcessor:
    if Qwen3VLProcessor is None:
        raise ImportError(
            "Qwen3VLProcessor is not available. "
            "Please upgrade transformers: pip install transformers>=4.52.0"
        )
    processor_path = os.environ.get("GR00T_QWEN3_INIT_CONFIG_PATH") or model_name
    return Qwen3VLProcessor.from_pretrained(processor_path, **transformers_loading_kwargs)


class Gr00tN1d7DataCollator:
    def __init__(
        self,
        model_name: str,
        model_type: str = "qwen",
        transformers_loading_kwargs: dict = {},
    ):
        
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        
        self.processor.tokenizer.padding_side = "left"
        self.model_type = model_type
        self.model_name = model_name

    def __call__(self, features: list[Dict[str, Any]]) -> BatchFeature:
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))

        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            if key == "vlm_content":
                
                text_list = []
                image_inputs = []
                for v in values:
                    curr_text_list = [v["text"]]

                    text_list += curr_text_list
                    curr_image_inputs = v["images"]
                    image_inputs += curr_image_inputs

                vlm_inputs = self.processor(
                    text=text_list,
                    images=image_inputs,
                    return_tensors="pt",
                    padding=True,
                )
                for k, v in vlm_inputs.items():
                    batch[k] = v
            elif key in (
                "pixel_values",
                "image_grid_thw",
                "attention_mask",
                "input_ids",
            ):
                raise Exception("Not implemented")
            else:
                
                batch[key] = torch.from_numpy(np.stack(values))
        return BatchFeature(data={"inputs": batch})

    def __str__(self):
        return f"Gr00tN1d7DataCollator(model_name={self.model_name}, model_type={self.model_type})"


class Gr00tN1d7Processor(BaseProcessor):
    data_collator_class = Gr00tN1d7DataCollator

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: (dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None) = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        image_crop_size: list[int] = None,
        image_target_size: list[int] = None,
        shortest_image_edge: int = 256,
        crop_fraction: float = 0.95,
        random_rotation_angle: int | None = None,
        color_jitter_params: dict[str, float] | None = None,
        formalize_language: bool = True,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        model_type: str = "qwen",
        max_state_dim: int = 29,
        max_action_dim: int = 29,
        max_action_horizon: int = 50,
        apply_sincos_state_encoding: bool = False,
        use_albumentations: bool = False,
        extra_augmentation_config: dict | None = None,
        use_relative_action: bool = False,
        embodiment_id_mapping: dict[str, int] | None = None,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
        
        exclude_state: bool = False,
        state_dropout_prob: float = 0.0,
        
        use_mean_std: bool = False,
        
        letter_box_transform: bool = False,
    ):
        self.modality_configs = parse_modality_configs(modality_configs)

        
        self.state_action_processor = StateActionProcessor(
            modality_configs=modality_configs,
            statistics=statistics,
            use_percentiles=use_percentiles,
            clip_outliers=clip_outliers,
            apply_sincos_state_encoding=apply_sincos_state_encoding,
            use_relative_action=use_relative_action,
        )

        
        self.use_percentiles = use_percentiles
        self.use_mean_std = use_mean_std
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action
        self.extra_augmentation_config = extra_augmentation_config

        
        self.exclude_state = exclude_state
        self.state_dropout_prob = state_dropout_prob

        self.letter_box_transform = letter_box_transform

        
        self.formalize_language = formalize_language
        self.model_name = model_name
        self.model_type = model_type

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_action_horizon = max_action_horizon

        
        self.image_crop_size = image_crop_size
        self.image_target_size = image_target_size
        self.random_rotation_angle = random_rotation_angle
        self.color_jitter_params = color_jitter_params
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        
        self.processor.tokenizer.padding_side = "left"
        self.embodiment_id_mapping = embodiment_id_mapping or EMBODIMENT_TAG_TO_PROJECTOR_INDEX
        
        for k, v in EMBODIMENT_TAG_TO_PROJECTOR_INDEX.items():
            if k not in self.embodiment_id_mapping:
                self.embodiment_id_mapping[k] = v
        self.shortest_image_edge = shortest_image_edge
        self.crop_fraction = crop_fraction

        
        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}

        
        self.use_albumentations = use_albumentations
        if use_albumentations:
            self.train_image_transform, self.eval_image_transform = (
                build_image_transformations_albumentations(
                    image_target_size,
                    image_crop_size,
                    random_rotation_angle,
                    color_jitter_params,
                    shortest_image_edge,
                    crop_fraction,
                    extra_augmentation_config=self.extra_augmentation_config,
                )
            )
        else:
            self.train_image_transform, self.eval_image_transform = build_image_transformations(
                image_target_size,
                image_crop_size,
                random_rotation_angle,
                color_jitter_params,
            )
        self._collator = self.data_collator_class(
            model_name=model_name,
            model_type=model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )
        self.train()

    @property
    def collator(self):
        return self._collator

    def train(self):
        super().train()
        self.state_action_processor.train()

    def eval(self):
        super().eval()
        self.state_action_processor.eval()

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """Set dataset statistics for normalization."""
        for key in statistics:
            if key not in self.statistics or override:
                if override:
                    logger.info("Overriding statistics for embodiment %r", key)
                self.statistics[key] = deepcopy(statistics[key])
            else:
                
                
                
                
                
                
                logger.warning(
                    "Statistics for embodiment %r already present; new stats "
                    "DISCARDED (override=False). If the new data differs from "
                    "the existing distribution this will cause silent "
                    "normalization mismatch — pass override=True (or "
                    "override_pretraining_statistics=True at the dataset level) "
                    "to use the merged stats instead.",
                    key,
                )

        self.state_action_processor.set_statistics(statistics, override=override)

        
        self.action_dim = {}
        for embodiment_tag in self.state_action_processor.statistics:
            self.action_dim[embodiment_tag] = self.state_action_processor.get_action_dim(
                embodiment_tag
            )

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ):
        """Undo action normalization and convert relative actions to absolute."""
        
        out_dict = {}
        start_idx = 0
        joint_groups = self.modality_configs[embodiment_tag.value]["action"].modality_keys
        action_horizon = len(self.modality_configs[embodiment_tag.value]["action"].delta_indices)
        for key in joint_groups:
            joint_dim = self.state_action_processor.norm_params[embodiment_tag.value]["action"][
                key
            ]["dim"].item()
            out_dict[key] = action[..., :action_horizon, start_idx : start_idx + joint_dim]
            start_idx += joint_dim

        
        return self.state_action_processor.unapply_action(
            out_dict, embodiment_tag.value, state=state
        )

    def unapply(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
        prev_action: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Undo action normalization and convert relative→absolute.

        Args:
            action: Normalized action array of shape (..., action_horizon, action_dim)
            embodiment_tag: Embodiment tag
            state: State observations with "state." prefixed keys (for relative actions)
            prev_action: Unused (kept for API compatibility)

        Returns:
            Dict mapping "action.<key>" to unnormalized (absolute) action arrays.
        """
        out_dict = {}
        start_idx = 0
        joint_groups = self.modality_configs[embodiment_tag.value]["action"].modality_keys
        action_horizon = len(self.modality_configs[embodiment_tag.value]["action"].delta_indices)
        for key in joint_groups:
            joint_dim = self.state_action_processor.norm_params[embodiment_tag.value]["action"][
                key
            ]["dim"].item()
            out_dict[key] = action[..., :action_horizon, start_idx : start_idx + joint_dim]
            start_idx += joint_dim

        
        stripped_state = None
        if state is not None:
            stripped_state = {k.replace("state.", ""): v for k, v in state.items()}

        result = self.state_action_processor.unapply_action(
            out_dict, embodiment_tag.value, state=stripped_state
        )
        return {f"action.{key}": value for key, value in result.items()}

    def process_observation(self, observation: dict[str, Any], embodiment_tag: EmbodimentTag):
        """Process batched observation tensors for inference.

        Args:
            observation: Dict with keys like "video.<view>", "state.<key>", "<language_key>"
                Video values expected as numpy arrays of shape (B, T, H, W, C).
            embodiment_tag: Embodiment tag identifying the robot configuration.

        Returns:
            BatchFeature with tokenized VLM inputs, state, embodiment_id, and action_mask.
        """
        modality_config = self.modality_configs[embodiment_tag.value]
        transformed_observation = {}

        
        state_keys = modality_config["state"].modality_keys
        state_data = {key: observation[f"state.{key}"] for key in state_keys}
        exclude_state = self.exclude_state or getattr(
            modality_config["state"], "exclude_state", False
        )
        if exclude_state:
            normalized_states = torch.cat(
                [torch.from_numpy(np.zeros_like(state_data[key])) for key in state_keys], dim=-1
            )
        else:
            norm_state_dict = self.state_action_processor.apply_state(
                state=state_data, embodiment_tag=embodiment_tag.value
            )
            normalized_states = torch.cat(
                [torch.from_numpy(norm_state_dict[key]) for key in state_keys], dim=-1
            )

        assert normalized_states.shape[1] <= self.max_state_dim, (
            f"State dimension {normalized_states.shape[1]} exceeds max_state_dim {self.max_state_dim}"
        )
        padding_shape = (
            *normalized_states.shape[:-1],
            self.max_state_dim - normalized_states.shape[-1],
        )
        normalized_states = torch.cat([normalized_states, torch.zeros(padding_shape)], dim=-1)
        transformed_observation["state"] = normalized_states

        
        image_keys = modality_config["video"].modality_keys
        images_dict = {view: torch.from_numpy(observation[f"video.{view}"]) for view in image_keys}
        images = torch.stack(
            [images_dict[view] for view in image_keys], dim=2
        )  
        assert images.ndim == 6
        B, T, V, img_H, img_W, img_C = images.shape

        if self.use_albumentations:
            images_flat = images.reshape(B * T * V, img_H, img_W, img_C)
            pil_images = [Image.fromarray(img.numpy()) for img in images_flat]
            transformed_pil, _ = apply_with_replay(self.eval_image_transform, pil_images)
            transformed_stacked = torch.stack(transformed_pil)  
            _, img_C_new, img_H_new, img_W_new = transformed_stacked.shape
            transformed_images = transformed_stacked.reshape(
                B, T * V, img_C_new, img_H_new, img_W_new
            ).numpy()
        else:
            
            images_perm = images.permute(0, 1, 2, 5, 3, 4).reshape(B, T * V, img_C, img_H, img_W)
            transformed_images = self.eval_image_transform(images_perm).numpy()

        language_key = modality_config["language"].modality_keys[0]
        language = [
            re.sub(r"[^\w\s]", "", lang.lower()) if self.formalize_language else lang
            for lang in observation[language_key]
        ]

        texts, all_images = [], []
        for i in range(B):
            vlm_inputs = self._apply_vlm_processing(transformed_images[i], language[i])
            vc = vlm_inputs["vlm_content"]
            texts.append(vc["text"])
            all_images.extend(vc["images"])
        tokenized = self.processor(text=texts, images=all_images, return_tensors="pt", padding=True)
        for k, v in tokenized.items():
            transformed_observation[k] = v

        embodiment_id = (
            torch.ones(B, dtype=torch.int32) * self.embodiment_id_mapping[embodiment_tag.value]
        )
        transformed_observation["embodiment_id"] = embodiment_id

        
        action_config = modality_config["action"]
        action_horizon = len(action_config.delta_indices)
        assert action_horizon <= self.max_action_horizon, (
            f"Action horizon {action_horizon} (from delta_indices) exceeds"
            f" max_action_horizon {self.max_action_horizon}. Increase model config"
            f" action_horizon to >= {action_horizon}."
        )
        action_mask = torch.zeros((B, self.max_action_horizon), dtype=torch.float32)
        if action_horizon > 0:
            action_mask[:, :action_horizon] = 1.0
        transformed_observation["action_mask"] = action_mask

        return BatchFeature(transformed_observation)

    def _apply_vlm_processing(self, images: np.ndarray, language: str) -> BatchFeature:
        """
        Args:
            batch:
                video: [T, C, H, W]
        Returns: vlm_content format for collation
        """
        
        pil_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in images]

        
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in pil_images],
                    {"type": "text", "text": language},
                ],
            }
        ]

        
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        
        return {
            "vlm_content": {
                "text": text,
                "images": pil_images,
                "conversation": conversation,
            }
        }

    def __call__(
        self,
        messages: list[dict[str, Any]],
    ):
        assert len(messages) == 1
        content = messages[0]["content"]
        embodiment_tag = content.embodiment
        action_data = content.actions
        state_data = content.states

        
        norm_state_dict, normalized_actions = self.state_action_processor.apply(
            state=state_data,
            action=action_data,
            embodiment_tag=embodiment_tag.value,
        )

        if normalized_actions:
            
            action_keys = self.modality_configs[embodiment_tag.value]["action"].modality_keys
            normalized_actions = torch.cat(
                [torch.from_numpy(normalized_actions[key]) for key in action_keys],
                dim=-1,
            )  
            action_dim = normalized_actions.shape[1]
            
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        normalized_actions.shape[0],
                        self.max_action_dim - normalized_actions.shape[1],
                    ),
                ],
                dim=-1,
            )  
            
            action_horizon = normalized_actions.shape[0]
            assert action_horizon <= self.max_action_horizon, (
                f"Action sequence length {action_horizon} exceeds max_action_horizon"
                f" {self.max_action_horizon}. Increase model config action_horizon to"
                f" >= {action_horizon}."
            )
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        self.max_action_horizon - normalized_actions.shape[0],
                        self.max_action_dim,
                    ),
                ],
                dim=0,
            )  
            
            action_mask = torch.ones_like(normalized_actions)
            action_mask[action_horizon:] = 0
            action_mask[:, action_dim:] = 0
        else:
            assert not self.training, "Action is required in training mode"
            normalized_actions = None
            action_mask = None

        
        state_keys = self.modality_configs[embodiment_tag.value]["state"].modality_keys
        exclude_state = self.exclude_state or getattr(
            self.modality_configs[embodiment_tag.value]["state"], "exclude_state", False
        )
        if exclude_state or (
            self.state_dropout_prob > 0
            and random.random() < self.state_dropout_prob
            and self.training
        ):
            normalized_states = torch.cat(
                [torch.from_numpy(np.zeros_like(state_data[key])) for key in state_keys], dim=-1
            )
        else:
            normalized_states = torch.cat(
                [torch.from_numpy(norm_state_dict[key]) for key in state_keys], dim=-1
            )
        normalized_states = torch.cat(
            [
                normalized_states,
                torch.zeros(
                    normalized_states.shape[0],
                    self.max_state_dim - normalized_states.shape[1],
                ),
            ],
            dim=-1,
        )

        
        if self.training:
            image_transform = self.train_image_transform
        else:
            image_transform = self.eval_image_transform
        image_keys = self.modality_configs[embodiment_tag.value]["video"].modality_keys

        if self.formalize_language:
            language = content.text.lower()
            language = re.sub(r"[^\w\s]", "", language)
        else:
            language = content.text

        vlm_inputs = self._get_vlm_inputs(
            image_keys=image_keys,
            images=content.images,
            masks=content.masks,
            image_transform=image_transform,
            language=language,
        )

        transformed_inputs = {
            "state": normalized_states.to(torch.get_default_dtype()),
        }
        if normalized_actions is not None:
            transformed_inputs["action"] = normalized_actions.to(torch.get_default_dtype())
        
        transformed_inputs.update(vlm_inputs)
        if action_mask is not None:
            transformed_inputs["action_mask"] = action_mask
        transformed_inputs["embodiment_id"] = self.embodiment_id_mapping[embodiment_tag.value]
        return transformed_inputs

    def _get_vlm_inputs(
        self,
        image_keys: list[str],
        images: list[Image.Image],
        masks: dict[str, list[np.ndarray]] | None,
        image_transform: transforms.Compose | A.Compose,
        language: str,
    ):
        temporal_stacked_images = {}

        if self.use_albumentations:
            
            replay = None
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
                if masks is not None:
                    assert view in masks, f"{view} not in masks"
                view_masks = masks.get(view) if masks else None
                view_images = images[view]

                
                transformed_images, replay = apply_with_replay(
                    image_transform, view_images, view_masks, replay
                )
                temporal_stacked_images[view] = torch.stack(transformed_images)  
        else:
            if masks is not None:
                raise ValueError(
                    "Mask transforms require albumentations. Set use_albumentations_transforms=True."
                )
            
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
                temporal_stacked_images[view] = torch.stack(
                    [image_transform(img) for img in images[view]]
                )  

        for k, v in temporal_stacked_images.items():
            assert isinstance(k, str), f"{k} is not a string"
            assert isinstance(v, torch.Tensor), f"{v} is not a torch tensor"
            assert v.ndim == 4, f"{v} is not a 4D tensor"
            assert v.dtype == torch.uint8, f"{v} is not a uint8 tensor"
            assert v.shape[1] == 3, f"{v} is not a 3 channel tensor"

        stacked_images = (
            torch.stack([temporal_stacked_images[view] for view in image_keys], dim=1)
            .flatten(0, 1)
            .numpy()
        )  

        vlm_inputs = self._apply_vlm_processing(stacked_images, language)
        return vlm_inputs

    def save_pretrained(self, save_directory: str | Path) -> list[Path]:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        main_config_file = save_directory / "processor_config.json"
        statistics_file = save_directory / "statistics.json"
        embodiment_id_file = save_directory / "embodiment_id.json"

        config = {
            "processor_class": self.__class__.__name__,
            "processor_kwargs": {
                "modality_configs": to_json_serializable(self.modality_configs),
                
                "image_crop_size": self.image_crop_size,
                "image_target_size": self.image_target_size,
                "use_albumentations": self.use_albumentations,
                "random_rotation_angle": self.random_rotation_angle,
                "color_jitter_params": self.color_jitter_params,
                "shortest_image_edge": self.shortest_image_edge,
                "crop_fraction": self.crop_fraction,
                "letter_box_transform": self.letter_box_transform,
                
                "model_name": self.model_name,
                "model_type": self.model_type,
                "formalize_language": self.formalize_language,
                
                "max_state_dim": self.max_state_dim,
                "max_action_dim": self.max_action_dim,
                "max_action_horizon": self.max_action_horizon,
                
                "use_percentiles": self.use_percentiles,
                "use_mean_std": self.use_mean_std,
                "clip_outliers": self.clip_outliers,
                "apply_sincos_state_encoding": self.apply_sincos_state_encoding,
                "use_relative_action": self.use_relative_action,
                
                "exclude_state": self.exclude_state,
                "state_dropout_prob": self.state_dropout_prob,
            },
        }
        with open(main_config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        with open(statistics_file, "w") as f:
            json.dump(
                to_json_serializable(self.state_action_processor.statistics),
                f,
                indent=2,
            )
        
        with open(embodiment_id_file, "w") as f:
            json.dump(self.embodiment_id_mapping, f, indent=2)
        return [main_config_file, statistics_file, embodiment_id_file]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs):
        transformers_loading_kwargs = kwargs.pop(
            "transformers_loading_kwargs", {"trust_remote_code": True}
        )
        hub_keys = (
            "_commit_hash",
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "subfolder",
            "token",
        )
        hub_kwargs = {key: kwargs.pop(key) for key in hub_keys if key in kwargs}
        use_auth_token = kwargs.pop("use_auth_token", None)
        if "token" not in hub_kwargs and use_auth_token is not None:
            hub_kwargs["token"] = use_auth_token
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        config_file = pretrained_model_name_or_path / "processor_config.json"
        statistics_file = pretrained_model_name_or_path / "statistics.json"
        embodiment_id_file = pretrained_model_name_or_path / "embodiment_id.json"
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if not is_local:
            config_file = Path(
                cached_file(pretrained_model_name_or_path, "processor_config.json", **hub_kwargs)
            )
            statistics_file = Path(
                cached_file(pretrained_model_name_or_path, "statistics.json", **hub_kwargs)
            )
            embodiment_id_file = Path(
                cached_file(pretrained_model_name_or_path, "embodiment_id.json", **hub_kwargs)
            )

        with open(config_file, "r") as f:
            config = json.load(f)
        with open(statistics_file, "r") as f:
            statistics = json.load(f)
        if embodiment_id_file.exists():
            with open(embodiment_id_file, "r") as f:
                embodiment_id_mapping = json.load(f)
        else:
            embodiment_id_mapping = None
        processor_kwargs = config["processor_kwargs"]
        processor_kwargs["statistics"] = statistics
        processor_kwargs["embodiment_id_mapping"] = embodiment_id_mapping

        
        
        
        processor_kwargs.setdefault("model_name", "nvidia/Cosmos-Reason2-2B")
        processor_kwargs.setdefault("model_type", "qwen")
        processor_kwargs.setdefault("clip_outliers", True)

        
        if kwargs:
            
            modality_configs = kwargs.pop("modality_configs", {})
            for embodiment_tag, modality_config in modality_configs.items():
                processor_kwargs["modality_configs"][embodiment_tag] = modality_config
            override_keys = [
                "random_rotation_angle",
                "color_jitter_params",
                "use_relative_action",
                "exclude_state",
                "state_dropout_prob",
                "use_mean_std",
                "model_name",
                "model_type",
                "max_action_horizon",
                "max_state_dim",
                "max_action_dim",
            ]
            for key in override_keys:
                if key in kwargs:
                    override = kwargs.pop(key)
                    if override is not None:
                        processor_kwargs[key] = override
        return cls(**processor_kwargs, transformers_loading_kwargs=transformers_loading_kwargs)


AutoProcessor.register("Gr00tN1d7", Gr00tN1d7Processor)
