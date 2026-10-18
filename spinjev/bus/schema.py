"""The signal contract between layers.

Two rules decide whether a quantity belongs here:

1. Anything a real SO-ARM100 cannot measure is not an observation. It goes in `Truth`, which
   feeds rewards and logs only. Joint angles, joint rates and actuator torque come from servos;
   the wrist pose comes from forward kinematics; the object pose comes from a pose estimator.
2. The cerebellum eats a fixed-length vector. `OBS_LAYOUT` is that vector's definition -- adding
   a field means retraining, so it is frozen for 阶段 1 onwards.
"""

from dataclasses import dataclass, field

import numpy as np

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
NJNT = len(ARM_JOINTS)


@dataclass
class Proprio:
    """Everything the arm can sense about itself. Measurable on the real servos."""

    t: float = 0.0
    q: np.ndarray = field(default_factory=lambda: np.zeros(NJNT))
    qvel: np.ndarray = field(default_factory=lambda: np.zeros(NJNT))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(NJNT))
    ee_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ee_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0]))
    ee_close: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    grasp_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def grip(self) -> float:
        return float(self.q[-1])

    @property
    def saturated(self) -> np.ndarray:
        """Servos at their torque ceiling are not tracking the spine's setpoint."""
        return np.abs(self.tau) >= 0.95 * TAU_LIMIT


@dataclass
class Percept:
    """Object pose as a detector would report it -- noisy, and sometimes absent."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    valid: bool = True
    confidence: float = 1.0


@dataclass
class Truth:
    """Simulation-only. Never an input to any learned component."""

    obj_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    obj_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    obj_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ncontact: int = 0


# STS3215 stall torque at 7.4 V, the actuator forcerange in assets/so_arm100/so_arm100.xml.
TAU_LIMIT = 1.86

# Fixed-length input vector for the cerebellum. Sections are concatenated in this order.
OBS_LAYOUT = (
    ("q", NJNT),
    ("qvel", NJNT),
    ("tau", NJNT),
    ("ee_pos", 3),
    ("ee_axis", 3),
    ("obj_pos", 3),
    ("obj_conf", 1),
    ("joint_margin", NJNT),
    ("grip_slip", 1),
)
OBS_DIM = sum(dim for _, dim in OBS_LAYOUT)


def obs_vector(parts: dict) -> np.ndarray:
    """Render the named sections into the flat vector the student eats."""
    return np.concatenate([np.asarray(parts[name], dtype=np.float64).reshape(dim) for name, dim in OBS_LAYOUT])
