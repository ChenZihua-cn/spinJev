"""Feature compressor: the last two seconds of proprioception plus the current percept, rendered
into the fixed-length vector of `OBS_LAYOUT`.

Everything here is something the arm can get without the simulator: joint angles and the two
derived sections come from `Proprio` (encoders, currents, forward kinematics) and `Percept` (the
pose estimator). No `Truth` argument exists on any of these functions -- that is the iron rule of
PLAN 2.7, enforced by signature.

Two of the nine sections are *window* quantities and are defined here.

`joint_margin` is the distance to the nearer hard stop of each joint in units of `MARGIN_REF`,
saturating at 1. A servo knows its own travel from calibration, so this is observable; the
commanded setpoint that would sharpen "about to run out of travel" is not in `Proprio`, so the
signal is purely positional.

`grip_slip` is how fast the object estimate is moving away from the hand, in units of
`SLIP_LIMIT`. Both the hand's own motion and the detector's noise are the enemies here: the
grasp-point term cancels the first (the wrist drags the object estimate along on every tick,
which is not a slip), and the regression over `SLIP_TICKS` ticks beats the second -- a 20 ms
difference of a 10 mm-noise pose is ~0.5 m/s of pure noise, so a one-tick estimator would report
slip constantly. Positive means the grip is losing ground.
"""

from collections import deque

import numpy as np

from spinjev.bus.ringbuffer import RingBuffer
from spinjev.bus.schema import Percept, Proprio, obs_vector

MARGIN_REF = 0.5  # rad; a scaling choice for the vector's dynamic range, not a servo spec
SLIP_TICKS = 20  # 0.4 s at 50 Hz: long enough that 10 mm pose noise does not read as motion
SLIP_LIMIT = 0.10  # m/s of object-vs-hand motion that saturates the index; a scaling choice


class Compressor:
    """`compressor(ring, proprio, percept)` -> vector of length `OBS_DIM`.

    Push the frame into the ring first, then compress it: the slip fit looks the hand's pose up by
    timestamp, so a tick that never reached the window simply contributes nothing to the fit.
    """

    def __init__(self, qlo, qhi, slip_ticks=SLIP_TICKS, slip_limit=SLIP_LIMIT, margin_ref=MARGIN_REF):
        self.qlo = np.asarray(qlo, dtype=np.float64)
        self.qhi = np.asarray(qhi, dtype=np.float64)
        self.slip_limit = float(slip_limit)
        self.margin_ref = float(margin_ref)
        self._objs = deque(maxlen=int(slip_ticks))
        self._slip = 0.0

    def reset(self):
        """Forget the object track; call it at an episode boundary."""
        self._objs.clear()
        self._slip = 0.0

    @property
    def slip_rate(self):
        """Raw slip estimate in m/s, before the vector's scaling."""
        return self._slip

    def joint_margin(self, proprio: Proprio) -> np.ndarray:
        margin = np.minimum(proprio.q - self.qlo, self.qhi - proprio.q)
        return np.clip(margin / self.margin_ref, 0.0, 1.0)

    def grip_slip(self, ring: RingBuffer, proprio: Proprio, percept: Percept) -> float:
        if percept.valid:
            self._objs.append((float(proprio.t), np.asarray(percept.pos, dtype=np.float64)))
        hands = {float(f.t): f.grasp_pos for f in ring.frames()}
        pts = [(t, float(np.linalg.norm(pos - hands[t]))) for t, pos in self._objs if t in hands]
        if len(pts) >= 3:
            x = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            x = x - x[0]
            denom = len(x) * float(x @ x) - float(x.sum()) ** 2
            if denom > 1e-12:
                self._slip = (len(x) * float(x @ y) - float(x.sum()) * float(y.sum())) / denom
        return float(np.clip(self._slip / self.slip_limit, -1.0, 1.0))

    def __call__(self, ring: RingBuffer, proprio: Proprio, percept: Percept) -> np.ndarray:
        parts = {
            "q": proprio.q,
            "qvel": proprio.qvel,
            "tau": proprio.tau,
            "ee_pos": proprio.ee_pos,
            "ee_axis": proprio.ee_axis,
            "obj_pos": np.asarray(percept.pos, dtype=np.float64),
            # a lost detection reads zero confidence, so a stale pose cannot masquerade as a vague one
            "obj_conf": float(percept.confidence) if percept.valid else 0.0,
            "joint_margin": self.joint_margin(proprio),
            "grip_slip": self.grip_slip(ring, proprio, percept),
        }
        return obs_vector(parts)
