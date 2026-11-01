"""CPU-only contracts for Gaussian scenes with Isaac mesh foreground.

Execute the renderer methods with recorded-like arrays and a rasterizer stub,
so these contracts do not require IsaacLab, PyTorch or a CUDA extension.
"""
import ast
import contextlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime/scripts/imitation_learning/robomimic"


def load_nodes(path, names, scope):
    nodes = [node for node in ast.parse(path.read_text()).body
             if getattr(node, "name", None) in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope


class ImageTensor:
    """Rasterizer output implementing only tensor-to-image operations."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def clamp(self, low, high):
        return ImageTensor(np.clip(self.value, low, high))

    def permute(self, *axes):
        return ImageTensor(self.value.transpose(axes))

    def squeeze(self):
        return ImageTensor(self.value.squeeze())


@pytest.mark.parametrize("color_match", [False, True])
def test_mesh_robot_and_objects_overlay_scene_without_depth_occlusion(monkeypatch, color_match):
    monkeypatch.delenv("HUMANOIDVLA_GS_DEBUG_DIR", raising=False)
    colors = np.array([[255, 36, 66], [255, 184, 48], [55, 255, 139]])
    scope = load_nodes(RUNTIME / "wbc_play_groot_sonic.py", {
        "Online2DGSRenderer", "semantic_foreground_mask", "apply_training_rgb_color_match"},
        {"np": np, "os": os, "Path": Path, "FOREGROUND_SEMANTIC_RGB": colors,
         "TRAINING_RGB_SCALE": np.array([0.902, 0.824, 0.831]),
         "TRAINING_RGB_OFFSET": np.array([11.6, 15.9, 9.0]),
         "camera_pose_for_2dgs": lambda *args: (np.zeros(3), [0, 0, 0, 1])})
    renderer = scope["Online2DGSRenderer"].__new__(scope["Online2DGSRenderer"])
    renderer.foreground_mode = "isaac_mesh"
    renderer.semantic_tolerance = 12
    renderer.apply_training_color_match = color_match
    renderer._debug_dumped = False
    renderer.zbuf = SimpleNamespace(MiniCam=lambda **kwargs: object(), pose_to_matrix=lambda *args: np.eye(4))
    background = np.full((2, 4, 3), [20, 40, 60], dtype=np.uint8)
    renderer.background_rgb_transform = lambda rgb: rgb + 5
    renderer.renderer = SimpleNamespace(
        intrinsics={"width": 4, "height": 2, "fx": 10, "fy": 10},
        render=lambda camera: {"render": ImageTensor(background.transpose(2, 0, 1) / 255),
                               "depth_m": ImageTensor(np.full((2, 4), 0.1)),
                               "valid_mask": ImageTensor([[True]*4, [False]*4])})
    rgb = np.full((2, 4, 3), [190, 110, 70], dtype=np.uint8)
    semantic = np.zeros_like(rgb)
    semantic[:, :3] = colors
    actual, depth = renderer.render_with_gs_bg(
        pos_=np.zeros(3), quat_w_world=np.array([1, 0, 0, 0]), isaac_render_rgb=rgb,
        isaac_render_depth=np.full((2, 4), 100.0), isaac_render_semantic_images=semantic)
    expected = background + 5
    expected[:, :3] = rgb[:, :3]
    if color_match:
        expected = scope["apply_training_rgb_color_match"](expected)
    np.testing.assert_array_equal(actual, expected)
    # Keep the established background-depth return contract.
    np.testing.assert_array_equal(depth[0], 100)
    np.testing.assert_array_equal(depth[1], 65535)


def test_camera_pose_is_local_to_selected_environment(monkeypatch):
    monkeypatch.delenv("HUMANOIDVLA_GS_DEBUG_DIR", raising=False)
    scope = load_nodes(RUNTIME / "wbc_play_groot_sonic.py", {"gs_render_from_env"},
                       {"np": np, "os": os, "Path": Path})
    class Batch:
        def __init__(self, value):
            self.value = value
        def __getitem__(self, index):
            return ImageTensor(self.value[index])
    calls = []
    camera = SimpleNamespace(data=SimpleNamespace(
        output={name: Batch([np.zeros((2, 2, 3))]*2) for name in
                ("rgb", "distance_to_image_plane", "semantic_segmentation")},
        pos_w=Batch([[0, 0, 0], [12, 24, 6]]), quat_w_world=Batch([[1, 0, 0, 0]]*2)))
    renderer = SimpleNamespace(foreground_mode="isaac_mesh", render_with_gs_bg=lambda **kwargs: calls.append(kwargs))
    scope["gs_render_from_env"](renderer, camera, env_id=1, env_origin_w=np.array([10, 20, 0]),
                                 camera_pose_override=(np.array([13, 25, 7]), np.array([0, 0, 0, 1])))
    np.testing.assert_array_equal(calls[0]["pos_"], [3, 5, 7])
    np.testing.assert_array_equal(calls[0]["quat_w_world"], [0, 0, 0, 1])


def test_task3_hinge_updates_preserve_static_scene_and_mesh_rendering():
    scope = load_nodes(RUNTIME / "evaluate_task3.py", {"Task3DynamicDoorRenderer"},
                       {"np": np, "shared": SimpleNamespace(SceneSE2=SimpleNamespace(identity=lambda: None))})
    cls = scope["Task3DynamicDoorRenderer"]
    renderer = cls.__new__(cls)
    xyz = np.array([[1., 2., 3.], [4., 5., 6.], [8., 9., 10.]])
    renderer._scene_xyz0 = SimpleNamespace(device="cpu", dtype=np.float64)
    # Addition of the stored canonical scene coordinates is exercised below.
    class ScenePoints:
        device = "cpu"
        dtype = np.float64
        def __add__(self, translation):
            return xyz[:2] + translation
    class Destination:
        def __init__(self):
            self.values = np.zeros_like(xyz)
        def __getitem__(self, key):
            return SimpleNamespace(copy_=lambda value: self.values.__setitem__(key, value))
    renderer._scene_xyz0 = ScenePoints()
    destination = Destination()
    transforms = []
    renderer.dynamic = SimpleNamespace(scene_count=2, model=SimpleNamespace(_xyz=destination),
                                       set_relative_world_transform=transforms.append)
    renderer._last_scene_key = None
    renderer._last_hinge = None
    renderer._se2_matrix = lambda xy, yaw: np.eye(4)
    hinges = []
    renderer.hinge_cache = SimpleNamespace(transform=lambda angle: (hinges.append(angle) or np.eye(4), None))
    frames = []
    renderer.base = SimpleNamespace(zbuf=SimpleNamespace(torch=SimpleNamespace(
        no_grad=contextlib.nullcontext, as_tensor=lambda value, **kwargs: np.array(value))),
        render_env=lambda **kwargs: frames.append(kwargs) or np.zeros((2, 2, 3), dtype=np.uint8))
    renderer.color_matcher = None
    positions = np.array([[0.2]])
    door = SimpleNamespace(find_joints=lambda name: ([0], None), data=SimpleNamespace(joint_pos=positions))
    env = SimpleNamespace(_task3_eval_scene_se2={0: SimpleNamespace(translation_xy=(10, 20), delta_yaw=0)},
                          scene={"scene": door})
    for angle in (0.2, 0.2, 0.6):
        positions[0, 0] = angle
        renderer.render_env(env=env, env_id=0, robot="mesh robot", sim_joint_names=[])
    assert hinges == [0.2, 0.6]
    assert len(transforms) == 2
    assert len(frames) == 3
    np.testing.assert_array_equal(destination.values[:2], xyz[:2] + [10, 20, 0])
    np.testing.assert_array_equal(destination.values[2], [0, 0, 0])


def test_only_mesh_foreground_cli_is_supported():
    import argparse

    parser = argparse.ArgumentParser()
    tree = ast.parse((RUNTIME / "evaluation_runtime.py").read_text())
    foreground_arg = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                          and node.args and isinstance(node.args[0], ast.Constant)
                          and node.args[0].value == "--gs_foreground_mode")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=foreground_arg)], type_ignores=[])),
                 "foreground_argument", "exec", flags=0), {"parser": parser})
    assert parser.parse_args([]).gs_foreground_mode == "isaac_mesh"
    assert parser.parse_args(["--gs_foreground_mode", "isaac_mesh"]).gs_foreground_mode == "isaac_mesh"
    with pytest.raises(SystemExit):
        parser.parse_args(["--gs_foreground_mode", "robot_gs"])
    for filename in ("wbc_play_groot_sonic.py", "evaluation_runtime.py"):
        text = (RUNTIME / filename).read_text()
        for flag in ("--robot_gs_parts_dir", "--robot_gs_reference_ply", "--robot_link_json", "--robot_urdf"):
            assert flag not in text


def test_scene_adapter_import_does_not_load_robot_reconstruction(monkeypatch):
    import builtins
    import importlib.util
    import sys
    from types import ModuleType

    torch = ModuleType("torch")
    torch.no_grad = lambda: lambda function: function
    torch.nn = ModuleType("torch.nn")
    torch.nn.functional = ModuleType("torch.nn.functional")
    base = ModuleType("z_buffer_merger")
    base.GSBackgroundRenderer = type("BaseRenderer", (), {})
    base.MiniCam = object
    base.parse_pose_file = lambda *args: args
    for name, module in (("torch", torch), ("torch.nn", torch.nn),
                         ("torch.nn.functional", torch.nn.functional), ("z_buffer_merger", base)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    original_import = builtins.__import__
    forbidden = {"gs_3d_adapter", "select_robot_points_open3d", "open3d"}
    def checked_import(name, *args, **kwargs):
        assert name.split(".")[0] not in forbidden
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    path = ROOT / "runtime/submodule/2D_GS/Real2Sim/z_buffer_merger_with_robot.py"
    spec = importlib.util.spec_from_file_location("scene_adapter_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert issubclass(module.GSBackgroundRenderer, base.GSBackgroundRenderer)
    assert module.MiniCam is base.MiniCam
    assert module.parse_pose_file is base.parse_pose_file
    assert not hasattr(module.GSBackgroundRenderer, "build_robot_model")
    assert not hasattr(module, "RobotArmGSAssembler")
    import inspect
    assert "gs_model" not in inspect.signature(module.GSBackgroundRenderer.render).parameters


def test_renderer_signatures_match_all_initializer_call_sites():
    sources = {name: ast.parse((RUNTIME / name).read_text()) for name in
               ("wbc_play_groot_sonic.py", "gs_3d_adapter.py", "evaluate_task3.py")}
    signatures = {}
    for tree in sources.values():
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in {"Online2DGSRenderer", "ThreeDGSRenderer"}:
                constructor = next(item for item in node.body if getattr(item, "name", None) == "__init__")
                signatures[node.name] = {arg.arg for arg in constructor.args.args + constructor.args.kwonlyargs}
    for tree in sources.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in signatures:
                assert {keyword.arg for keyword in node.keywords if keyword.arg} <= signatures[node.func.id]
    task3_class = next(node for node in sources["evaluate_task3.py"].body
                       if isinstance(node, ast.ClassDef) and node.name == "Task3DynamicDoorRenderer")
    common = next(node.value for node in ast.walk(task3_class) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "common" for target in node.targets))
    assert {keyword.arg for keyword in common.keywords} <= signatures["Online2DGSRenderer"]
    for path in (RUNTIME / "wbc_play_groot_sonic.py", RUNTIME / "gs_3d_adapter.py"):
        source = path.read_text()
        for obsolete in ("RobotArmGSAssembler", "robot_attrs", "texture_robot", "DEFAULT_ROBOT"):
            assert obsolete not in source


@pytest.mark.parametrize("backend,inference", [("2dgs", False), ("2dgs", True), ("3dgs", False)])
def test_scene_adapter_renders_current_scene_model_with_metric_depth(backend, inference):
    class Tensor(np.ndarray):
        def clone(self):
            return self.copy()
        def float(self):
            return self.astype(np.float32)
    def tensor(value):
        return np.asarray(value).view(Tensor)
    torch = SimpleNamespace(
        zeros=lambda *args, **kwargs: np.zeros(3), float32=np.float32,
        isfinite=np.isfinite,
        nan_to_num=lambda value, **kwargs: tensor(np.nan_to_num(value, **kwargs)))
    path = ROOT / "runtime/submodule/2D_GS/Real2Sim/z_buffer_merger_with_robot.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "GSBackgroundRenderer")
    render = next(node for node in cls.body if getattr(node, "name", None) == "render")
    render.decorator_list = []
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[render], type_ignores=[]), str(path), "exec"), scope)
    model = object()  # Task3 replaces renderer.gs with its combined scene/door model.
    calls = []
    def rasterize(camera, gs, pipe, background, **kwargs):
        calls.append((gs, kwargs))
        return {"surf_depth": tensor([2., np.nan, 4., 5.]),
                "rend_alpha": tensor([1., 1., 0., 1.]),
                "depth": tensor([2., np.nan, 4., 200.])}
    renderer = SimpleNamespace(gs=model, backend_name=backend, pipe=SimpleNamespace(),
                               inference_only=inference, _supports_inference_only=True,
                               render_fn=rasterize)
    result = scope["render"](renderer, SimpleNamespace(znear=0.01, zfar=100.))
    np.testing.assert_array_equal(result["depth_m"],
                                  [2., 100., 100., 5.] if backend == "2dgs" else [2., 100., 4., 100.])
    np.testing.assert_array_equal(result["valid_mask"],
                                  [1., 1., 0., 1.] if backend == "2dgs" else [1., 0., 1., 0.])
    assert calls == [(model, {"inference_only": True} if inference else {})]
