from __future__ import annotations
import numpy as np

TASK3_TASK_NAME_TOKEN = "Final-Task3-CarryBasketKickDoor"
TASK3_WAIST_STIFFNESS_DELTA = 10.0


def resolve_robot_names(robot) -> tuple[list[str], list[str]]:
    joint_names = getattr(robot.data, "joint_names", None)
    body_names = getattr(robot.data, "body_names", None)
    if joint_names is None:
        joint_names = getattr(robot, "joint_names", None)
    if body_names is None:
        body_names = getattr(robot, "body_names", None)
    if joint_names is None or body_names is None:
        raise RuntimeError("Failed to resolve robot joint/body names.")
    return list(joint_names), list(body_names)

def build_name_indices(names: list[str], required_names: list[str], tag: str) -> list[int]:
    name_to_idx = {name: i for i, name in enumerate(names)}
    resolved: list[int] = []
    missing: list[str] = []
    for req in required_names:
        if req in name_to_idx:
            resolved.append(name_to_idx[req])
            continue
        candidates = [i for i, name in enumerate(names) if name.endswith(f"_{req}")]
        if len(candidates) == 1:
            resolved.append(candidates[0])
            continue
        missing.append(req)
    if missing:
        raise RuntimeError(f"Missing {tag} names in env: {missing}")
    return resolved

def resolve_env_action_order(env, sim_joint_names: list[str], fallback_joint_names: list[str]) -> tuple[list[str], list[int], int]:
    action_terms = list(env.action_manager._terms.values())
    if not action_terms:
        raise RuntimeError("No action terms found in env.action_manager.")
    action_term = action_terms[0]
    action_dim = int(action_term.action_dim)
    term_joint_names = getattr(action_term, "_joint_names", None)
    action_joint_names = list(term_joint_names) if term_joint_names is not None else list(fallback_joint_names)
    if len(action_joint_names) != action_dim:
        raise RuntimeError(f"Action joint name count {len(action_joint_names)} != action_dim {action_dim}")
    action_joint_sim_ids = build_name_indices(sim_joint_names, action_joint_names, "env action joint")
    return action_joint_names, action_joint_sim_ids, action_dim

def torch_to_np(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

def disable_time_out(env_cfg) -> None:
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is not None and hasattr(terminations, "time_out"):
        setattr(terminations, "time_out", None)

def apply_sonic_actuator_profile(
    robot_cfg,
    *,
    preserve_aux_actuators: bool = True,
    waist_stiffness_scale: float | None = None,
    waist_damping_scale: float | None = None,
) -> None:
    from isaaclab.actuators import ImplicitActuatorCfg

    if waist_stiffness_scale is None:
        waist_stiffness_scale = 1.0
    if waist_stiffness_scale <= 0.0:
        raise ValueError(f"waist_stiffness_scale must be > 0, got {waist_stiffness_scale}")
    if waist_damping_scale is None:
        waist_damping_scale = 1.0
    if waist_damping_scale <= 0.0:
        raise ValueError(f"waist_damping_scale must be > 0, got {waist_damping_scale}")

    armature_5020 = 0.003609725
    armature_7520_14 = 0.010177520
    armature_7520_22 = 0.025101925
    armature_4010 = 0.00425
    natural_freq = 10.0 * 2.0 * np.pi
    damping_ratio = 2.0

    def stiffness(armature: float) -> float:
        return armature * natural_freq * natural_freq

    def damping(armature: float) -> float:
        return 2.0 * damping_ratio * armature * natural_freq

    stiffness_5020 = stiffness(armature_5020)
    stiffness_7520_14 = stiffness(armature_7520_14)
    stiffness_7520_22 = stiffness(armature_7520_22)
    stiffness_4010 = stiffness(armature_4010)
    waist_stiffness_5020 = waist_stiffness_scale * 2.0 * stiffness_5020
    waist_yaw_stiffness = waist_stiffness_scale * stiffness_7520_14
    damping_5020 = damping(armature_5020)
    damping_7520_14 = damping(armature_7520_14)
    damping_7520_22 = damping(armature_7520_22)
    damping_4010 = damping(armature_4010)

    existing_actuators = dict(getattr(robot_cfg, "actuators", {}) or {})
    preserved = (
        {name: cfg for name, cfg in existing_actuators.items() if name in ("hand", "head")}
        if preserve_aux_actuators
        else {}
    )
    robot_cfg.soft_joint_pos_limit_factor = 0.9
    robot_cfg.actuators = {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 139.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 20.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": stiffness_7520_22,
                ".*_hip_roll_joint": stiffness_7520_22,
                ".*_hip_yaw_joint": stiffness_7520_14,
                ".*_knee_joint": stiffness_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": damping_7520_22,
                ".*_hip_roll_joint": damping_7520_22,
                ".*_hip_yaw_joint": damping_7520_14,
                ".*_knee_joint": damping_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": armature_7520_22,
                ".*_hip_roll_joint": armature_7520_22,
                ".*_hip_yaw_joint": armature_7520_14,
                ".*_knee_joint": armature_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * stiffness_5020,
            damping=2.0 * damping_5020,
            armature=2.0 * armature_5020,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=waist_stiffness_5020,
            damping=waist_damping_scale * 2.0 * damping_5020,
            armature=2.0 * armature_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=waist_yaw_stiffness,
            damping=damping_7520_14,
            armature=armature_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": stiffness_5020,
                ".*_shoulder_roll_joint": stiffness_5020,
                ".*_shoulder_yaw_joint": stiffness_5020,
                ".*_elbow_joint": stiffness_5020,
                ".*_wrist_roll_joint": stiffness_5020,
                ".*_wrist_pitch_joint": stiffness_4010,
                ".*_wrist_yaw_joint": stiffness_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": damping_5020,
                ".*_shoulder_roll_joint": damping_5020,
                ".*_shoulder_yaw_joint": damping_5020,
                ".*_elbow_joint": damping_5020,
                ".*_wrist_roll_joint": damping_5020,
                ".*_wrist_pitch_joint": damping_4010,
                ".*_wrist_yaw_joint": damping_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": armature_5020,
                ".*_shoulder_roll_joint": armature_5020,
                ".*_shoulder_yaw_joint": armature_5020,
                ".*_elbow_joint": armature_5020,
                ".*_wrist_roll_joint": armature_5020,
                ".*_wrist_pitch_joint": armature_4010,
                ".*_wrist_yaw_joint": armature_4010,
            },
        ),
    }
    robot_cfg.actuators.update(preserved)

def apply_task3_replay_waist_override(robot_cfg, task_name: str) -> dict[str, float]:
    """Match Task3 collection's +10 waist stiffness override."""

    if TASK3_TASK_NAME_TOKEN not in str(task_name):
        return {}
    applied: dict[str, float] = {}
    for actuator_name in ("waist", "waist_yaw"):
        actuator_cfg = robot_cfg.actuators.get(actuator_name)
        if actuator_cfg is None:
            raise KeyError(f"Task3 SONIC actuator profile is missing {actuator_name!r}")
        stiffness = getattr(actuator_cfg, "stiffness", None)
        if not isinstance(stiffness, (int, float, np.floating)):
            raise TypeError(
                f"Task3 {actuator_name} stiffness must be scalar after applying the SONIC profile, got {stiffness!r}"
            )
        actuator_cfg.stiffness = float(stiffness) + TASK3_WAIST_STIFFNESS_DELTA
        applied[actuator_name] = float(actuator_cfg.stiffness)
    return applied
