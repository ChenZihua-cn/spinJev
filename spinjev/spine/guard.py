"""Hard guards.

Deterministic, evaluated at every spine tick, and outside every learned component's reach:
when a guard fires the arm freezes at its current setpoint. Escalation does not go through here
-- escalate is a cerebellum output, a guard trip is the world refusing to cooperate.

Every input is something a real servo can report: joint angle, torque, and "the arm is touching
the table but nothing asked it to".
"""

from dataclasses import dataclass, field

import numpy as np

from spinjev.bus.schema import TAU_LIMIT

# Reach envelope measured on this asset: the jaw corner sweeps at most 0.43 m of radius at
# table height. These fences sit outside that, so they catch a solver or controller going wrong
# rather than trimming an ordinary reach.
RADIUS_FENCE = 0.55
HEIGHT_FENCE = 0.70
# Pressing the table at more than a few newtons is a crash; brushing it is not -- a 3 cm block on
# a flat table can only be gripped with the fingertips sweeping at table level.
TABLE_FORCE_LIMIT = 3.0
# A gripper closed onto a block is *supposed* to saturate, so its torque is not a straining-arm
# signal. The 5 arm joints are.
SATURATION_TICKS = 25
FATAL = ("table_hit", "out_of_workspace", "arm_straining")


@dataclass
class GuardState:
    trips: tuple = ()
    fatal: bool = False
    limit_margin: np.ndarray = field(default_factory=lambda: np.zeros(6))
    saturated: bool = False
    touching: bool = False

    def __bool__(self):
        return bool(self.trips)


class Guard:
    def __init__(self, qlo, qhi, tau_limit=TAU_LIMIT):
        self.qlo = np.asarray(qlo, dtype=np.float64)
        self.qhi = np.asarray(qhi, dtype=np.float64)
        self.tau_limit = float(tau_limit)
        self._sat = 0

    def reset(self):
        self._sat = 0

    def check(self, q, tau, ee_pos, table_force=0.0) -> GuardState:
        trips = []
        if table_force > TABLE_FORCE_LIMIT:
            trips.append("table_hit")
        radius = float(np.hypot(ee_pos[0], ee_pos[1]))
        if radius > RADIUS_FENCE or ee_pos[2] > HEIGHT_FENCE or ee_pos[2] < -0.05:
            trips.append("out_of_workspace")
        margin = np.minimum(q - self.qlo, self.qhi - q)
        straining = np.abs(tau[:-1]) >= 0.95 * self.tau_limit
        self._sat = self._sat + 1 if straining.any() else 0
        if self._sat > SATURATION_TICKS:
            trips.append("arm_straining")
        return GuardState(
            trips=tuple(trips),
            fatal=any(t in FATAL for t in trips),
            limit_margin=margin,
            saturated=bool(straining.any()),
            touching=table_force > 0.0,
        )
