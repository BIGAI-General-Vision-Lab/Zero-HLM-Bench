from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from .final_task_2_pick_up_hat_env_cfg import FinalTask2PickUpHatEnvCfg, FinalTask2PickUpHatSceneCfg


FINAL_TASK_2_PICK_UP_HAT_RENDER_TASK_ID = "Isaac-Final-Task2-PickUpHat-Render-Abs-v0"


_HEAD_CAMERA_INTRINSIC_MATRIX = [
    387.1702575683594,
    0.0,
    324.482177734375,
    0.0,
    386.6981506347656,
    243.91378784179688,
    0.0,
    0.0,
    1.0,
]


_HEAD_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/d455_link/d455_Color_link/head_camera",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.05593399809616198, 0.007245818534049027, 0.005172408873459918),
        rot=(0.9999008332055785, -0.0031024160714459344, 0.008536490842546162, -0.010762299633661657),
        convention="world",
    ),
    data_types=["rgb", "semantic_segmentation"],
    update_period=0.05,
    depth_clipping_behavior="max",
    colorize_semantic_segmentation=True,
    semantic_segmentation_mapping={
        "class:robot": (255, 36, 66, 255),
        "class:target_placement": (255, 184, 48, 255),
        "class:target_object": (55, 255, 139, 255),
        "class:UNLABELLED": (128, 128, 128, 255),
        "class:BACKGROUND": (12, 12, 12, 255),
    },
    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=_HEAD_CAMERA_INTRINSIC_MATRIX,
        width=640,
        height=480,
        focal_length=1.93,
        focus_distance=0.5,
        clipping_range=(0.01, 100.0),
        lock_camera=False,
    ),
    width=640,
    height=480,
    update_latest_camera_pose=True,
)


@configclass
class FinalTask2PickUpHatRenderSceneCfg(FinalTask2PickUpHatSceneCfg):
    head_camera: TiledCameraCfg = _HEAD_CAMERA_CFG
    lazy_sensor_update = False


@configclass
class FinalTask2PickUpHatRenderEnvCfg(FinalTask2PickUpHatEnvCfg):
    scene: FinalTask2PickUpHatRenderSceneCfg = FinalTask2PickUpHatRenderSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    def __post_init__(self):
        super().__post_init__()
