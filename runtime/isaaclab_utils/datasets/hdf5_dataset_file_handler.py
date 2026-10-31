# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import h5py
import json
import numpy as np
import os
import torch
from collections.abc import Iterable

from .dataset_file_handler_base import DatasetFileHandlerBase
from .episode_data import EpisodeData
import imageio.v2 as imageio
import cv2
from pathlib import Path
import tqdm

class HDF5DatasetFileHandler(DatasetFileHandlerBase):
    """HDF5 dataset file handler for storing and loading episode data."""

    def __init__(self):
        """Initializes the HDF5 dataset file handler."""
        self._hdf5_file_stream = None
        self._hdf5_data_group = None
        self._demo_count = 0
        self._env_args = {}

    def open(self, file_path: str, mode: str = "r"):
        """Open an existing dataset file."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        self._hdf5_file_stream = h5py.File(file_path, mode)
        self._hdf5_data_group = self._hdf5_file_stream["data"]
        self._demo_count = len(self._hdf5_data_group)

    def create(self, file_path: str, env_name: str = None):
        """Create a new dataset file."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"
        dir_path = os.path.dirname(file_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        self._hdf5_file_stream = h5py.File(file_path, "w")

        # set up a data group in the file
        self._hdf5_data_group = self._hdf5_file_stream.create_group("data")
        self._hdf5_data_group.attrs["total"] = 0
        self._demo_count = 0

        # set environment arguments
        # the environment type (we use gym environment type) is set to be compatible with robomimic
        # Ref: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/envs/env_base.py#L15
        env_name = env_name if env_name is not None else ""
        self.add_env_args({"env_name": env_name, "type": 2})

    def __del__(self):
        """Destructor for the file handler."""
        self.close()

    """
    Properties
    """

    def add_env_args(self, env_args: dict):
        """Add environment arguments to the dataset."""
        self._raise_if_not_initialized()
        self._env_args.update(env_args)
        self._hdf5_data_group.attrs["env_args"] = json.dumps(self._env_args)

    def set_env_name(self, env_name: str):
        """Set the environment name."""
        self._raise_if_not_initialized()
        self.add_env_args({"env_name": env_name})

    def get_env_name(self) -> str | None:
        """Get the environment name."""
        self._raise_if_not_initialized()
        env_args = json.loads(self._hdf5_data_group.attrs["env_args"])
        if "env_name" in env_args:
            return env_args["env_name"]
        return None

    def get_episode_names(self) -> Iterable[str]:
        """Get the names of the episodes in the file."""
        self._raise_if_not_initialized()
        return self._hdf5_data_group.keys()

    def get_num_episodes(self) -> int:
        """Get number of episodes in the file."""
        return self._demo_count

    @property
    def demo_count(self) -> int:
        """The number of demos collected so far."""
        return self._demo_count

    """
    Operations.
    """

    def load_episode(self, episode_name: str, device: str) -> EpisodeData | None:
        """Load episode data from the file."""
        self._raise_if_not_initialized()
        if episode_name not in self._hdf5_data_group:
            return None
        episode = EpisodeData()
        h5_episode_group = self._hdf5_data_group[episode_name]

        def load_dataset_helper(group):
            """Helper method to load dataset that contains recursive dict objects."""
            data = {}
            for key in group:
                if isinstance(group[key], h5py.Group):
                    data[key] = load_dataset_helper(group[key])
                else:
                    # Converting group[key] to numpy array greatly improves the performance
                    # when converting to torch tensor
                    data[key] = torch.tensor(np.array(group[key]), device=device)
            return data

        episode.data = load_dataset_helper(h5_episode_group)

        if "seed" in h5_episode_group.attrs:
            episode.seed = h5_episode_group.attrs["seed"]

        if "success" in h5_episode_group.attrs:
            episode.success = h5_episode_group.attrs["success"]

        episode.env_id = self.get_env_name()

        return episode

    def write_episode(self, episode: EpisodeData, output_dir: str = None):
        """Add an episode to the dataset.

        Args:
            episode: The episode data to add.
        """
        self._raise_if_not_initialized()
        if episode.is_empty():
            return

        #revised by wzm
        video_base_path = Path("./datasets/render_video") if output_dir is None else Path(output_dir)
        rgb_dir = video_base_path / "rgb"
        depth_dir = video_base_path / "depth"
        semantic_dir = video_base_path / "semantic"

        for dir_path in [rgb_dir, depth_dir, semantic_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        demo_id = f"demo_{self._demo_count}"
        if "images" in episode.data:
            images_data = episode.data["images"]
            try:
                if "rgb" in images_data:
                    rgb_frames = images_data["rgb"]
                    print(f"RGB frames type: {type(rgb_frames)}, shape: {rgb_frames.shape}, dtype: {rgb_frames.dtype}")
                    rgb_path = rgb_dir / f"{demo_id}.mp4"
                    self._save_video(rgb_path, rgb_frames)

                if "depth" in images_data:
                    depth_frames = images_data["depth"]
                    # depth_path = depth_dir / f"{demo_id}.mp4"
                    depth_path = depth_dir / f"{demo_id}.h5"
                    # self._save_depth_video(depth_path, depth_frames)
                    self._save_depth_h5(depth_path, depth_frames)

                if "semantic" in images_data:
                    semantic_frames = images_data["semantic"]
                    semantic_path = semantic_dir / f"{demo_id}.mp4"
                    self._save_semantic_video(semantic_path, semantic_frames)

                episode.data["images"] = {"info": images_data["info"]}

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error saving video for {demo_id}: {str(e)}")

        # create episode group based on demo count
        h5_episode_group = self._hdf5_data_group.create_group(f"demo_{self._demo_count}")

        # store number of steps taken
        if "actions" in episode.data:
            h5_episode_group.attrs["num_samples"] = len(episode.data["actions"])
        else:
            h5_episode_group.attrs["num_samples"] = 0

        if episode.seed is not None:
            h5_episode_group.attrs["seed"] = episode.seed

        if episode.success is not None:
            h5_episode_group.attrs["success"] = episode.success

        def create_dataset_helper(group, key, value):
            """Helper method to create dataset that contains recursive dict objects."""
            if isinstance(value, dict):
                key_group = group.create_group(key)
                for sub_key, sub_value in value.items():
                    create_dataset_helper(key_group, sub_key, sub_value)
            else:
                group.create_dataset(key, data=value.cpu().numpy(), compression="gzip")

        for key, value in episode.data.items():
            create_dataset_helper(h5_episode_group, key, value)

        # increment total step counts
        self._hdf5_data_group.attrs["total"] += h5_episode_group.attrs["num_samples"]

        # increment total demo counts
        self._demo_count += 1

    def flush(self):
        """Flush the episode data to disk."""
        self._raise_if_not_initialized()

        self._hdf5_file_stream.flush()

    def close(self):
        """Close the dataset file handler."""
        if self._hdf5_file_stream is not None:
            self._hdf5_file_stream.close()
            self._hdf5_file_stream = None

    def _raise_if_not_initialized(self):
        """Raise an error if the dataset file handler is not initialized."""
        if self._hdf5_file_stream is None:
            raise RuntimeError("HDF5 dataset file stream is not initialized")

    def _save_video(self, path: Path, frames, fps=30):

        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        elif not isinstance(frames, np.ndarray):
            frames = np.array(frames)

        if len(frames.shape) != 4:
            raise ValueError(f"Expected frames shape [T, H, W, C], got {frames.shape}")

        T, H, W, C = frames.shape
        if C != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {C} channels")

        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)

        writer = imageio.get_writer(str(path), fps=fps, codec='libx264', quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    def _save_depth_video(self, path: Path, depth_frames):
        if isinstance(depth_frames, torch.Tensor):
            depth_frames = depth_frames.cpu().numpy()

        if len(depth_frames) == 0:
            return

        if depth_frames.ndim == 4 and depth_frames.shape[3] == 1:
            depth_frames = depth_frames.squeeze(axis=-1)
        height, width = depth_frames.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height), isColor=True)

        valid_depths = depth_frames[np.isfinite(depth_frames) & (depth_frames > 0)]

        if len(valid_depths) > 0:
            depth_min = np.min(valid_depths)
            depth_max = np.max(valid_depths)
        else:
            depth_min, depth_max = 0.0, 1.0 

        if abs(depth_max - depth_min) < 1e-6:
            depth_max = depth_min + 1.0

        
        for depth in depth_frames:
            depth_normalized = np.zeros((height, width), dtype=np.uint8)
            valid_mask = np.isfinite(depth) & (depth > 0)

            if depth_max > depth_min:
                depth_normalized[valid_mask] = np.clip(
                    255 * (depth[valid_mask] - depth_min) / (depth_max - depth_min),
                    0, 255
                ).astype(np.uint8)

            depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            depth_color[~valid_mask] = 0

            out.write(depth_color)

        out.release()

    # modified by ydy
    def _save_depth_h5(self, path: Path, depth_frames):
        """Save batched depth frames (N,H,W,1) to HDF5 with PNG compression."""
        if isinstance(depth_frames, torch.Tensor):
            depth_frames = depth_frames.cpu().numpy()

        depth_frames = np.squeeze(depth_frames, axis=-1)  # [N,H,W,1] -> [N,H,W]
        processed = np.where(np.isfinite(depth_frames), depth_frames, 0)
        scaled = (processed * 1000).clip(0, 65535).astype(np.uint16)

        with h5py.File(path, 'w') as h5:
            for frame_idx in range(scaled.shape[0]):
                _, encoded = cv2.imencode('.png', scaled[frame_idx])
                h5.create_dataset(f"frame_{frame_idx:04d}", data=encoded)


    def _save_semantic_video(self, path: Path, semantic_frames):
        """
        Save semantic frames as a lossless video using FFV1 codec.

        Args:
            path: Output file path (will auto-correct to .avi extension)
            semantic_frames: Input frames as [N,H,W] or [N,H,W,C] tensor/array 
                            (supports single-channel or RGB formats)

        Raises:
            RuntimeError: If video writer cannot be initialized
            ValueError: If input data is invalid
        """
        # Convert PyTorch tensor to numpy if needed
        if isinstance(semantic_frames, torch.Tensor):
            semantic_frames = semantic_frames.cpu().numpy()

        # Validate input dimensions
        if len(semantic_frames) == 0:
            raise ValueError("Empty frame sequence provided")
        if semantic_frames.ndim not in [3, 4]:
            raise ValueError(f"Invalid input shape: {semantic_frames.shape}")

        # Ensure AVI container for FFV1 codec
        output_path = path.with_suffix('.avi') if path.suffix != '.avi' else path
        height, width = semantic_frames.shape[1:3]

        # Initialize video writer
        try:
            fourcc = cv2.VideoWriter_fourcc(*'FFV1')
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                30.0,  # FPS
                (width, height),
                isColor=True  # Force color output
            )

            if not writer.isOpened():
                raise RuntimeError("Failed to initialize video writer. Check codec support.")

            # Process and write each frame
            for frame in semantic_frames:
                # Handle single-channel frames
                if frame.ndim == 3 and frame.shape[2] == 1:
                    frame = frame.squeeze()

                # Normalize and convert to uint8
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)

                # Apply colormap to single-channel frames
                if frame.ndim == 2:
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                # Remove alpha channel if present
                elif frame.shape[2] == 4:
                    frame = frame[..., :3]

                # Convert to BGR and write frame
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        except Exception as e:
            # Cleanup on failure
            if 'writer' in locals():
                writer.release()
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"Video saving failed: {str(e)}")

        finally:
            if 'writer' in locals():
                writer.release()