"""Primitive controllers.

A primitive is (a deterministic controller, a goal source, a speed profile), *not* a motor
command: the position actuators in the MJCF close the 500 Hz loop, and this file only decides
what setpoint they are aiming at.

Naming is the semantic vocabulary the cerebellum predicts over; the implementation is shared.
Of the seven names in PLAN 2.3, five collapse onto two controllers:

    接近 / 移送 / 收回  ->  APPROACH, differing only in which goal the caller supplies
    对准                ->  APPROACH with a waypoint above the object
    闭合 / 释放          ->  the gripper controller
    提升                ->  APPROACH with the xy frozen

escalate is deliberately absent: it is the act-or-escalate head's output, not a controller.
"""

import time
from enum import Enum

import numpy as np

from spinjev.spine.guard import Guard

PHYS_DT = 0.002  # model.opt.timestep, asserted in test/test_spine.py
DECIM = 10       # spine ticks every 20 ms -> 50 Hz decisions, 500 Hz servo
SLEW = 2.5       # rad/s cap on setpoint motion; a comfort limit, not a servo spec
REACHED_TOL = 0.01  # arrival tolerance, in metres of wrist position
GRIP_TOL = 0.03
GRIP_STALL_TORQUE = 0.85  # fraction of the servo ceiling that counts as "pushing against something"
GRIP_STALL_VEL = 0.05     # rad/s: below this the jaw has stopped
GRIP_STALL_TICKS = 5      # ... and it has stayed stopped for this many ticks
CLOSE_Q = -0.22  # measured: the low end of the gripper range is jaws-shut
# Pre-grasp opening. Not the mechanical end (2.0) and not the tightest fit: the channel between
# the jaws at the pinch depth is ~36 mm for a 30 mm block, so 6 mm of slack, and the 10 mm IK
# residual eats most of that. Opening wider buys the straddle back.
OPEN_Q = 1.8
TIMEOUT_S = 4.0


class Primitive(str, Enum):
    APPROACH = "approach"
    CLOSE = "close"
    LIFT = "lift"
    RELEASE = "release"


class Spine:
    """Turns one `Choice` + goal into joint setpoints and drives the arm there.

    The backend supplies the kinematics (`sim.kin`) and its own geometry (`sim.tip_offset`), so
    nothing here knows it is pointed at MuJoCo rather than servos.
    """

    def __init__(self, sim, guard=None, decim=DECIM, slew=SLEW, timeout_s=TIMEOUT_S):
        self.sim = sim
        self.guard = guard or Guard(sim.qlo, sim.qhi)
        self.decim = int(decim)
        self.dt = self.decim * PHYS_DT
        self.max_ticks = int(round(timeout_s / self.dt))
        self.slew = float(slew) * self.dt
        self.prim = None
        self.goal = None
        self.q_target = sim.data.qpos[sim.arm_qadr].copy()
        self.ticks = 0
        self._stall = 0
        self.status = "idle"
        self.guard_state = None
        self.solve_s = 0.0

    # -- command interface ---------------------------------------------------------
    def command(self, prim, goal=None):
        """Arm a primitive. APPROACH/LIFT take a world wrist position; LIFT uses only its z.

        The gripper setpoint is latched: a grasp that stopped because the jaws met the block has
        to keep being driven shut through everything that follows, or the clamp force disappears
        the moment the arm starts to move.
        """
        grip = self.q_target[5]
        self.prim = Primitive(prim)
        self.q_target = self.sim.data.qpos[self.sim.arm_qadr].copy()
        self.q_target[5] = grip
        self.guard.reset()
        self.ticks = 0
        self._stall = 0
        self.status = "running"
        if self.prim is Primitive.LIFT:
            xy = self.sim.kin.ee()[0][:2]
            self.goal = np.array([xy[0], xy[1], float(goal[2])])
        elif self.prim is Primitive.APPROACH:
            self.goal = np.asarray(goal, dtype=np.float64)
        else:
            self.goal = None
        return self

    def hold(self):
        """Freeze where the arm is: the guard's fail-safe, and what a finished primitive leaves.
        The gripper keeps holding its last command rather than its last measured angle, so a grip
        does not loosen while the arm waits."""
        grip = self.q_target[5]
        self.prim = None
        self.goal = None
        self.q_target = self.sim.data.qpos[self.sim.arm_qadr].copy()
        self.q_target[5] = grip

    def finished(self):
        return self.status != "running"

    # -- one 50 Hz tick ------------------------------------------------------------
    def tick(self):
        q = self.sim.data.qpos[self.sim.arm_qadr].copy()
        tau = self.sim.data.actuator_force[self.sim.ujnt].copy()
        pos = self.sim.kin.ee()[0]
        state = self.guard.check(q, tau, pos, self.sim.contact_force_with_table())
        self.guard_state = state
        if state.fatal:
            self.hold()
            self.status = "trip:" + ",".join(state.trips)
            return state
        if self.prim is None:
            self.sim.act(q)
            return state

        t0 = time.perf_counter()
        stalled = False
        if self.prim in (Primitive.APPROACH, Primitive.LIFT):
            n = self.sim.kin.lo.size
            sol, _, _, _ = self.sim.kin.solve(self.goal, self.q_target[:n], iters=25)
            target = np.concatenate([sol, self.q_target[n:5], [self.q_target[5]]])
            err = float(np.linalg.norm(pos - self.goal))
            tol = REACHED_TOL
        else:
            shut = CLOSE_Q if self.prim is Primitive.CLOSE else OPEN_Q
            target = np.array([*self.q_target[:5], shut])
            err = abs(q[5] - shut)
            tol = GRIP_TOL
            # Something between the jaws: the servo is at its torque ceiling, the jaw is not moving,
            # and it is not anywhere near the shut end. That is what a grip looks like from inside
            # the arm -- the only grasp signal a real servo can give.
            stalled = (
                self.prim is Primitive.CLOSE
                and err > tol
                and abs(tau[5]) > GRIP_STALL_TORQUE * float(self.sim.tau_limit[5])
                and abs(self.sim.data.qvel[self.sim.arm_dofadr[5]]) < GRIP_STALL_VEL
            )
        self.solve_s = time.perf_counter() - t0
        self._stall = self._stall + 1 if stalled else 0

        delta = np.clip(target - q, -self.slew, self.slew)
        self.q_target = np.clip(q + delta, self.sim.qlo, self.sim.qhi)
        self.sim.act(self.q_target)
        self.ticks += 1

        if err < tol:
            self.status = "reached"
            self.prim = None
        elif self._stall >= GRIP_STALL_TICKS:
            self.status = "blocked"
            self.prim = None
        elif self.ticks >= self.max_ticks:
            self.status = "timeout"
            self.prim = None
        return state

    def advance(self, ticks=1):
        """Run `ticks` decision periods, stopping early if the primitive completes or trips."""
        for _ in range(int(ticks)):
            self.tick()
            self.sim.step(self.decim)
            if self.prim is None:
                break
        return self
