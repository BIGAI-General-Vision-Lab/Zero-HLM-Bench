# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from .final_task_walk_pick_place_render_env_cfg import FinalWalkPickPlaceRenderEnvCfg
from .final_task_2_pick_up_hat_render_env_cfg import FinalTask2PickUpHatRenderEnvCfg
from .final_task_3_carry_basket_kick_door_render_env_cfg import FinalTask3CarryBasketKickDoorRenderEnvCfg

for task_id, config in (
    ("Isaac-Final-Walk-PickPlace-Render-Abs-v0", FinalWalkPickPlaceRenderEnvCfg),
    ("Isaac-Final-Task2-PickUpHat-Render-Abs-v0", FinalTask2PickUpHatRenderEnvCfg),
    ("Isaac-Final-Task3-CarryBasketKickDoor-Render-Abs-v0", FinalTask3CarryBasketKickDoorRenderEnvCfg),
):
    gym.register(id=task_id, entry_point="envs:ManagerBasedRLEnv",
                 kwargs={"env_cfg_entry_point": config}, disable_env_checker=True)
