# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-place specific event helpers."""

from __future__ import annotations

import random

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from envs import ManagerBasedEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


_POSE_RANGE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def _sample_pose_offsets(
    count: int,
    pose_range: dict[str, tuple[float, float]],
    *,
    device: str,
) -> torch.Tensor:
    """Sample XYZ and RPY offsets using IsaacLab's uniform sampler."""

    bounds = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in _POSE_RANGE_KEYS],
        device=device,
        dtype=torch.float32,
    )
    return math_utils.sample_uniform(bounds[:, 0], bounds[:, 1], (count, 6), device=device)


def _world_aabb_from_local_bounds(
    positions: torch.Tensor,
    orientations: torch.Tensor,
    local_bounds_min: tuple[float, float, float],
    local_bounds_max: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project an oriented local bounding box into a conservative world AABB."""

    local_min = positions.new_tensor(local_bounds_min)
    local_max = positions.new_tensor(local_bounds_max)
    local_center = 0.5 * (local_min + local_max)
    local_half_extent = 0.5 * (local_max - local_min)
    rotation = math_utils.matrix_from_quat(orientations)
    world_center = positions + torch.matmul(rotation, local_center)
    world_half_extent = torch.matmul(rotation.abs(), local_half_extent)
    return world_center, world_half_extent


def _world_aabbs_overlap(
    first_positions: torch.Tensor,
    first_orientations: torch.Tensor,
    first_local_bounds_min: tuple[float, float, float],
    first_local_bounds_max: tuple[float, float, float],
    second_positions: torch.Tensor,
    second_orientations: torch.Tensor,
    second_local_bounds_min: tuple[float, float, float],
    second_local_bounds_max: tuple[float, float, float],
    clearance: float,
) -> torch.Tensor:
    """Return a per-sample conservative overlap mask for two oriented assets."""

    first_center, first_half_extent = _world_aabb_from_local_bounds(
        first_positions,
        first_orientations,
        first_local_bounds_min,
        first_local_bounds_max,
    )
    second_center, second_half_extent = _world_aabb_from_local_bounds(
        second_positions,
        second_orientations,
        second_local_bounds_min,
        second_local_bounds_max,
    )
    separation = (first_center - second_center).abs()
    overlap_limit = first_half_extent + second_half_extent + float(clearance)
    return torch.all(separation <= overlap_limit, dim=-1)


def reset_rigid_object_pair_without_overlap(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    object_pose_range: dict[str, tuple[float, float]],
    placement_pose_range: dict[str, tuple[float, float]],
    object_local_bounds_min: tuple[float, float, float],
    object_local_bounds_max: tuple[float, float, float],
    placement_local_bounds_min: tuple[float, float, float],
    placement_local_bounds_max: tuple[float, float, float],
    clearance: float = 0.015,
    max_sampling_attempts: int = 64,
    tag_success: bool = False,
    now_attempts: int = 0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    placement_cfg: SceneEntityCfg = SceneEntityCfg("placement"),
) -> None:
    """Reset an object and placement together while rejecting colliding poses.

    Collision rejection uses conservative world-space AABBs derived from each
    asset's local USD bounds. If random sampling is exhausted, the known-safe
    default pair is used instead; an unsafe default raises before either pose is
    written to simulation.

    ``tag_success`` and ``now_attempts`` are accepted for compatibility with
    this project's custom EventManager reset protocol.
    """

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return
    if max_sampling_attempts < 1:
        raise ValueError("max_sampling_attempts must be at least 1")

    object_asset: RigidObject = env.scene[object_cfg.name]
    placement_asset: RigidObject = env.scene[placement_cfg.name]
    object_root_state = object_asset.data.default_root_state[env_ids].clone()
    placement_root_state = placement_asset.data.default_root_state[env_ids].clone()
    env_origins = env.scene.env_origins[env_ids]

    object_pose = object_root_state[:, :7].clone()
    placement_pose = placement_root_state[:, :7].clone()
    object_pose[:, :3] += env_origins
    placement_pose[:, :3] += env_origins

    pending = torch.arange(env_ids.numel(), device=env.device, dtype=torch.long)
    for _ in range(max_sampling_attempts):
        if pending.numel() == 0:
            break

        object_offsets = _sample_pose_offsets(
            pending.numel(), object_pose_range, device=object_asset.device
        ).to(dtype=object_pose.dtype)
        placement_offsets = _sample_pose_offsets(
            pending.numel(), placement_pose_range, device=placement_asset.device
        ).to(dtype=placement_pose.dtype)

        object_positions = object_root_state[pending, :3] + env_origins[pending] + object_offsets[:, :3]
        placement_positions = (
            placement_root_state[pending, :3] + env_origins[pending] + placement_offsets[:, :3]
        )
        object_delta = math_utils.quat_from_euler_xyz(
            object_offsets[:, 3], object_offsets[:, 4], object_offsets[:, 5]
        )
        placement_delta = math_utils.quat_from_euler_xyz(
            placement_offsets[:, 3], placement_offsets[:, 4], placement_offsets[:, 5]
        )
        object_orientations = math_utils.quat_mul(object_root_state[pending, 3:7], object_delta)
        placement_orientations = math_utils.quat_mul(
            placement_root_state[pending, 3:7], placement_delta
        )

        overlap = _world_aabbs_overlap(
            object_positions,
            object_orientations,
            object_local_bounds_min,
            object_local_bounds_max,
            placement_positions,
            placement_orientations,
            placement_local_bounds_min,
            placement_local_bounds_max,
            clearance,
        )
        accepted = ~overlap
        accepted_ids = pending[accepted]
        object_pose[accepted_ids, :3] = object_positions[accepted]
        object_pose[accepted_ids, 3:7] = object_orientations[accepted]
        placement_pose[accepted_ids, :3] = placement_positions[accepted]
        placement_pose[accepted_ids, 3:7] = placement_orientations[accepted]
        pending = pending[overlap]

    if pending.numel() > 0:
        defaults_overlap = _world_aabbs_overlap(
            object_pose[pending, :3],
            object_pose[pending, 3:7],
            object_local_bounds_min,
            object_local_bounds_max,
            placement_pose[pending, :3],
            placement_pose[pending, 3:7],
            placement_local_bounds_min,
            placement_local_bounds_max,
            clearance,
        )
        if torch.any(defaults_overlap):
            failed_env_ids = env_ids[pending[defaults_overlap]].tolist()
            raise RuntimeError(
                "Object/placement reset sampling failed and default poses are not collision-safe "
                f"for envs {failed_env_ids}."
            )

    object_asset.write_root_pose_to_sim(object_pose, env_ids=env_ids)
    object_asset.write_root_velocity_to_sim(object_root_state[:, 7:13], env_ids=env_ids)
    placement_asset.write_root_pose_to_sim(placement_pose, env_ids=env_ids)
    placement_asset.write_root_velocity_to_sim(placement_root_state[:, 7:13], env_ids=env_ids)


def _collect_tag_positions(tag_root_prim_path: str) -> list[tuple[str, list[float]]]:
    """Collect world-space tag positions under the given root prim path."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    table_paths = sim_utils.find_matching_prim_paths(tag_root_prim_path)
    positions = []
    for table_path in table_paths:
        table_prim = stage.GetPrimAtPath(table_path)
        if not table_prim:
            continue
        for prim in Usd.PrimRange(table_prim):
            if not prim:
                continue
            if prim.GetPath().pathString == table_path:
                # Skip the container prim itself; tags are its children.
                continue
            xf = UsdGeom.Xformable(prim)
            if not xf:
                continue
            world_tf: Gf.Matrix4d = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            trans = world_tf.ExtractTranslation()
            positions.append((prim.GetPath().pathString, [float(trans[0]), float(trans[1]), float(trans[2])]))
    return positions


def _tags_for_env(tag_positions: list[tuple[str, list[float]]], env_id: int) -> list[tuple[str, list[float]]]:
    env_key = f"/World/envs/env_{env_id}/"
    return [(path, pos) for path, pos in tag_positions if env_key in path]


def reset_root_state_from_tags(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    tag_root_prim_path: str = "/World/envs/env_.*/floor2/Xform_small_table",
    per_tag_quota: int | None = None,
    tag_selection: str = "round_robin",
    reset_counts_when_exhausted: bool = True,
    tag_success: object | None = None,
    now_attempts: object | None = None,
    z_offset: float = 0.05,
    log_tag_resets: bool = False,
):
    """Reset the asset root state using tag positions as XY anchors.

    The tag positions are sampled from the prims under ``tag_root_prim_path`` for each environment.
    Pose ranges are applied as offsets around the sampled tag position, and velocities are sampled
    from the provided ranges, similar to ``reset_root_state_uniform``.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    tag_positions = _collect_tag_positions(tag_root_prim_path)
    base_positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    positions = base_positions.clone()

    state = getattr(env, "_tag_reset_state", None)
    if state is None:
        state = {"counts": {}, "cursor": {}}
        setattr(env, "_tag_reset_state", state)

    for i, env_id in enumerate(env_ids.tolist()):
        tags = _tags_for_env(tag_positions, env_id)
        if not tags:
            continue
        tag_index_map = {path: idx for idx, (path, _) in enumerate(tags)}
        chosen_path = None
        if per_tag_quota is None or per_tag_quota <= 0:
            chosen_path, pos = random.choice(tags)
        else:
            counts = state["counts"].setdefault(env_id, {})
            cursor = state["cursor"].setdefault(env_id, 0)
            remaining = [(path, pos) for path, pos in tags if counts.get(path, 0) < per_tag_quota]
            if not remaining and reset_counts_when_exhausted:
                counts.clear()
                remaining = tags
            if not remaining:
                chosen_path, pos = random.choice(tags)
            elif tag_selection == "round_robin":
                start = cursor % len(remaining)
                chosen_path, pos = remaining[start]
                state["cursor"][env_id] = cursor + 1
            else:
                chosen_path, pos = random.choice(remaining)
            if chosen_path is not None:
                counts[chosen_path] = counts.get(chosen_path, 0) + 1

        positions[i] = torch.tensor([pos[0], pos[1], pos[2] + z_offset], device=asset.device)
        if log_tag_resets and chosen_path is not None:
            tag_idx = tag_index_map.get(chosen_path, -1)
            used = state["counts"].get(env_id, {}).get(chosen_path, 0)
            if per_tag_quota is not None and per_tag_quota > 0:
                print(f"[tag-reset] env={env_id} tag={tag_idx} count={used}/{per_tag_quota} path={chosen_path}")
            else:
                print(f"[tag-reset] env={env_id} tag={tag_idx} count={used} path={chosen_path}")

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = positions + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
