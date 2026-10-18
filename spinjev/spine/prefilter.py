"""Physical pre-filter for the `Choice` candidate set.

It answers one question per primitive: *could the body execute this right now* -- reachability,
joint limits, the table. "Should I close now" is a goal question and does not live here.

APPROACH is legal when *any* supplied waypoint is executable, so "the block itself is out of
reach" is carried by the per-waypoint reasons (`approach:grasp`, `approach:hover`, ...), not by the
primitive set. That is the distinction an escalate decision has to read.

Call it when the goal set changes (a new object estimate, a new phase), not on every tick: each
waypoint costs an IK solve. `gripper_range` is the mechanical travel, not the pre-grasp opening --
"can I open further" is a question about the stop.
"""

from dataclasses import dataclass, field

import numpy as np

from spinjev.bus.schema import TAU_LIMIT
from spinjev.spine.primitives import REACHED_TOL, Primitive

GRASP_RADIUS = 0.05  # a block this close to the grasp point is between the jaws
IK_TOL = REACHED_TOL  # legal iff the spine can actually arrive, at the spine's own tolerance
AXIS_DOWN = -0.90    # cosine between the approach axis and straight down
LIMIT_KEEP = 0.01    # joint travel a solution must leave itself
TORQUE_HEADROOM = 0.90  # hold at most this fraction of the servo ceiling, or the waypoint is refused


@dataclass
class Legality:
    legal: frozenset = field(default_factory=frozenset)
    reasons: dict = field(default_factory=dict)
    waypoints: dict = field(default_factory=dict)

    def __contains__(self, prim):
        return prim in self.legal


class Prefilter:
    def __init__(self, kin, guard, min_wrist_z, gripper_range, torque_of=None, tau_limit=TAU_LIMIT):
        self.kin = kin
        self.guard = guard
        self.min_wrist_z = float(min_wrist_z)
        self.lo, self.hi = gripper_range
        # `torque_of(q)` -> per-joint holding torque. Supplied by the backend: on the sim it is
        # gravity compensation, on the arm it is the measured holding current.
        self.torque_of = torque_of
        self.tau_limit = float(tau_limit)

    def held(self, proprio, percept):
        """Observable grasp flag: the *estimated* block centre sits at the grasp point. On the
        real arm it is the same test, fed by the vision pose estimate instead of the sim. The
        grasp point comes from the backend, because where a block has to sit to be caught between
        two asymmetric jaws is a property of the gripper, not of the controller."""
        if not percept.valid:
            return False
        return bool(np.linalg.norm(np.asarray(percept.pos) - proprio.grasp_pos) < GRASP_RADIUS)

    def waypoint(self, goal, q):
        """(joint target, reason) for a wrist goal. Two seeds -- the live pose, then the posture
        reference -- enough to step out of a folded elbow without a full search."""
        best = None
        n = self.kin.lo.size
        for seed in (q[:n], self.kin.qref):
            sol, err, axis, _ = self.kin.solve(goal, np.clip(seed, self.kin.lo, self.kin.hi), iters=80)
            if best is None or err < best[1]:
                best = (sol, err, axis)
            if err < IK_TOL and axis[2] < AXIS_DOWN:
                break
        sol, err, axis = best
        margin = np.minimum(sol - self.kin.lo, self.kin.hi - sol)
        if err >= IK_TOL:
            return None, "ik"
        if axis[2] >= AXIS_DOWN:
            return None, "orientation"
        if margin.min() < LIMIT_KEEP:
            return None, "limits"
        if goal[2] < self.min_wrist_z:
            return None, "table"
        if self.torque_of is not None and np.max(np.abs(self.torque_of(sol))) > TORQUE_HEADROOM * self.tau_limit:
            return None, "torque"
        return sol, "ok"

    def legal(self, proprio, percept, goals):
        """`goals` maps a waypoint name to a world wrist position. Returns the candidate set the
        `Choice` head may draw from, plus why each rejected primitive was rejected."""
        reasons, targets = {}, {}
        q5 = proprio.q[: self.kin.addr.size]
        for name, pos in goals.items():
            sol, why = self.waypoint(np.asarray(pos, dtype=np.float64), q5)
            if why == "ok":
                targets[name] = sol
            else:
                reasons[f"approach:{name}"] = why
        reasons["approach"] = "ok" if targets else "unreachable"
        reasons["close"] = "ok" if proprio.grip > self.lo + 0.05 else "already_shut"
        reasons["release"] = "ok" if proprio.grip < self.hi - 0.05 else "already_open"
        state = self.guard.check(proprio.q, proprio.tau, proprio.ee_pos, 0.0)
        reasons["lift"] = "ok" if (self.held(proprio, percept) and proprio.ee_pos[2] >= self.min_wrist_z
                                   and not state.saturated) else "nothing_gripped"

        tests = {
            Primitive.APPROACH: bool(targets),
            Primitive.CLOSE: reasons["close"] == "ok",
            Primitive.LIFT: reasons["lift"] == "ok",
            Primitive.RELEASE: reasons["release"] == "ok",
        }
        return Legality(frozenset(p for p, ok in tests.items() if ok), reasons, targets)
