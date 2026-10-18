"""Desktop pick-and-place: pick the block, carry it to the drop zone, release it.

Two facts drive this layer:

* 80% of episodes put the block inside the arm's measured reach, 20% put it *genuinely* outside
  it (or hidden behind the occluder). That 20% is what gives `escalate` a ground truth, so the
  sampler enforces the ratio instead of leaving it to chance.
* What "outside the reach" means is set by the measured gripper-down envelope of this asset
  (test/test_spine.py), not by the 0.43 m jaw-corner sweep in PLAN: the wrist itself only gets
  to r = 0.20..0.35 m and z <= 0.20 m. Reachable placements stay inside that with margin,
  unreachable ones are past the far fence or folded too close in.
"""

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from spinjev.backends.sim.mujoco_sim import TABLE_SCENE, Sim
from spinjev.bus.schema import Percept
from spinjev.spine.prefilter import Prefilter
from spinjev.spine.primitives import OPEN_Q, Spine

TABLE_HALF = (0.40, 0.50)  # tabletop box half-sizes, from table_scene.xml
TABLE_CENTRE_Y = -0.35
# Envelope measured with the jaws pointing down (test/test_spine.py scans it): the wrist reaches
# r 0.20..0.35 m straight ahead, 0.20..0.30 m at +-45 deg, and no higher than z = 0.20 m. Closer
# than 0.20 m the arm cannot fold without lifting the jaws. Everything here sits inside that.
R_REACH = 0.32  # outer radius of an ordinary grasp
R_NEAR = 0.24  # inside this the elbow has no room
THETA_MAX = math.radians(20)
R_AWAY = (0.42, 0.55)  # "beyond the arm" placements
R_TOO_CLOSE = (0.10, 0.17)  # "the arm cannot fold in that far" placements -- also escalate
DROP_ZONE = np.array([0.125, -0.216])  # 0.25 m out at +30 deg: the loaded arm shrinks its own envelope
DROP_RADIUS = 0.07
MIN_OBJECT_GAP = 0.12  # keep the block out of the drop zone so the transfer is a real step
HOLD_CLEAR = 0.02  # "held" means the block centre rose this much above where it rests
HOLD_TICKS = 25  # ... and it stays there for 0.5 s at 50 Hz
SETTLE_TICKS = 30


@dataclass
class Reward:
    hold: float = 1.0
    drop: float = -1.0
    guard_trip: float = -1.0
    escalate_when_right: float = 1.0
    escalate_when_wrong: float = -0.5


@dataclass
class TaskSpec:
    """Grasp geometry, measured on this asset.

    The basin is narrow and it is not our imagination: the pinch surfaces sit 60-100 mm in front
    of the wrist, the jaws touch each other at ~40 mm, and the channel centre moves as the jaws
    close -- so one fixed lateral offset cannot be right for both the open descent and the closed
    grip. depth 0.085-0.095 x lateral -0.035..-0.025 is what grips today. Consequence for 阶段 1c:
    expert trajectories do not all succeed, and that is why every decision sample carries the
    trajectory's 0/1 outcome instead of trusting the expert's intent.
    """
    # Both grasp offsets are measured, not chosen: test/test_spine.py sweeps the descent against
    # the lateral jaw-channel offset and reads off the pair that actually grips a block standing on
    # the table. GRASP_DEPTH is how far down the wrist sits below that point, GRASP_LATERAL how far
    # the channel sits off the approach axis (the gripper is asymmetric -- see mujoco_sim).
    grasp_depth: float = 0.090
    grasp_lateral: float = -0.030
    min_wrist_z: float = 0.115
    pregrasp_clear: float = 0.06  # hover height above the block before descending
    lift_height: float = 0.15  # carry height: clears the block, inside the measured z <= 0.20 ceiling
    home: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.26, 0.16]))
    max_ticks: int = 900  # 18 s at 50 Hz
    friction_jitter: float = 0.25
    reward: Reward = field(default_factory=Reward)


@dataclass
class Episode:
    index: int = 0
    object_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = 0.0
    unreachable: bool = False
    occluded: bool = False

    @property
    def escalate_gt(self) -> bool:
        return self.unreachable or self.occluded


@dataclass
class Outcome:
    done: bool = False
    success: bool = False
    escalated: bool = False
    released: bool = False
    dropped: bool = False
    tripped: tuple = ()
    held_ticks: int = 0
    ticks: int = 0
    reward: float = 0.0
    reason: str = ""


class PickPlace:
    """One Sim, the randomiser, and the success / escalate bookkeeping."""

    def __init__(self, sim=None, perceptor=None, spec=TaskSpec(), seed=0):
        self.sim = sim or Sim(xml=TABLE_SCENE)
        self.spine = Spine(self.sim)
        self.prefilter = Prefilter(
            self.sim.kin, self.spine.guard, spec.min_wrist_z,
            (self.sim.qlo[-1], self.sim.qhi[-1]),
            # the solver returns the 5 reach joints; the gripper is not part of a waypoint
            torque_of=lambda q: self.sim.static_torque(
                np.concatenate([q, [self.sim.data.qpos[self.sim.gadr]]])
            )[:5],
        )
        self.perceptor = perceptor
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.ep = Episode()
        self.out = Outcome()
        self._prev_z = self.block_z
        self._settled = 0
        self._was_shut = False
        self.gid = lambda n: mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, n)

    @property
    def hold_z(self) -> float:
        """Block centre high enough to call the grip held: it has to clear its own rest height, not a
        fixed number that ignores whether the block is standing or lying down."""
        return self.sim.block_half_z + HOLD_CLEAR

    @property
    def block_z(self) -> float:
        """Rest height of the target block = its half-height, read from the model, never assumed."""
        return self.sim.block_half_z

    # -- episode lifecycle ---------------------------------------------------------
    def reset(self, unreachable=None, occluded=None, xy=None):
        """A fifth of every episode is unwinnable -- 10% out of physical reach, 10% hidden
        behind the occluder. Both carry the same label (escalate), and neither is left to chance.
        """
        if unreachable is None and occluded is None:
            u = self.rng.random()
            unreachable, occluded = u < 0.10, 0.10 <= u < 0.20
        unreachable = bool(unreachable)
        occluded = bool(occluded) and not unreachable
        xy = np.asarray(xy, dtype=np.float64) if xy is not None else self._sample_xy(unreachable)
        self.sim.reset(xy)
        self._jitter_friction()
        if occluded:
            self.sim.set_occluder([xy[0], xy[1], 0.30])
        self.spine.hold()
        self.spine.status = "idle"
        self.spine.guard.reset()
        self.ep = Episode(
            index=self.ep.index + 1, object_xy=xy, radius=float(np.linalg.norm(xy)),
            unreachable=unreachable, occluded=occluded,
        )
        self.out = Outcome()
        self._prev_z = self.block_z
        self._settled = 0
        self._was_shut = False
        return self.ep

    def _sample_xy(self, unreachable):
        for _ in range(500):
            if unreachable:
                # Half out of reach because it is far, half because it is too close. A policy
                # that only learns "far = escalate" has learned a proxy, not the workspace.
                band = R_AWAY if self.rng.random() < 0.5 else R_TOO_CLOSE
                r = self.rng.uniform(*band)
            else:
                r = self.rng.uniform(R_NEAR, R_REACH)
            th = self.rng.uniform(-THETA_MAX, THETA_MAX)
            xy = np.array([r * math.sin(th), -r * math.cos(th)])
            on_table = abs(xy[0]) < TABLE_HALF[0] - 0.05 and abs(xy[1] - TABLE_CENTRE_Y) < TABLE_HALF[1] - 0.05
            if on_table and float(np.linalg.norm(xy - DROP_ZONE)) > MIN_OBJECT_GAP:
                return xy
        raise RuntimeError("no legal object placement in 500 draws")

    def _jitter_friction(self):
        m = self.sim.model
        scale = float(self.rng.uniform(1 - self.spec.friction_jitter, 1 + self.spec.friction_jitter))
        for name in ("target", "tabletop"):
            g = self.gid(name)
            if g >= 0:
                m.geom_friction[g, 0] = 1.5 * scale

    # -- state readout -------------------------------------------------------------
    def sense(self):
        return self.sim.sense()

    def truth(self):
        return self.sim.truth()

    def percept(self) -> Percept:
        truth = self.truth()
        if self.perceptor is None:
            return Percept(pos=truth.obj_pos.copy())
        return self.perceptor(truth, occluded=self.ep.occluded)

    def wrist_goal(self, block_xy, block_z, heading, lift=0.0, depth=None, lateral=None):
        """Where the wrist has to be for the *grasp point* to land on the block (+ `lift`).

        `heading` is the waypoint's bearing, measured the way `_sample_xy` places blocks:
        xy = (r sin h, -r cos h). test/test_spine.py `closing_direction` measures that the jaws'
        closing direction for such a waypoint is exactly (cos h, sin h) -- the gripper is not
        symmetric about the approach axis, so the wrist has to stand off to the side by
        `lateral` or a finger pushes the block over instead of catching it.
        """
        depth = self.spec.grasp_depth if depth is None else depth
        lateral = self.spec.grasp_lateral if lateral is None else lateral
        cx, cy = math.cos(heading), math.sin(heading)
        return np.array([block_xy[0] - lateral * cx, block_xy[1] - lateral * cy, block_z + lift + depth])

    def heading_of(self, xy):
        return float(math.atan2(float(xy[0]), -float(xy[1])))

    def goals(self, percept=None):
        """Waypoints the pre-filter should test, all as wrist positions."""
        p = percept if percept is not None else self.percept()
        th = self.heading_of(p.pos[:2])
        return {
            "hover": self.wrist_goal(p.pos[:2], p.pos[2], th, lift=self.spec.pregrasp_clear),
            "grasp": self.wrist_goal(p.pos[:2], p.pos[2], th),
            "drop": np.array([DROP_ZONE[0], DROP_ZONE[1], self.spec.lift_height]),
            "home": np.asarray(self.spec.home, dtype=np.float64),
        }

    def candidates(self, percept=None):
        p = percept if percept is not None else self.percept()
        return self.prefilter.legal(self.sense(), p, self.goals(p))

    # -- driving -------------------------------------------------------------------
    def drive(self, prim, goal=None, max_ticks=None):
        """Run one primitive to completion, updating the bookkeeping on every decision tick.
        This is the loop experts and evaluation share: nothing advances the arm except through
        here, so no caller can forget to count a tick."""
        self.spine.command(prim, goal)
        budget = self.spine.max_ticks if max_ticks is None else int(max_ticks)
        while self.spine.status == "running" and not self.out.done and budget > 0:
            self.spine.advance(1)
            self.update()
            budget -= 1
        return self.spine.status

    def wait(self, ticks):
        """Hold the current setpoints while the world settles."""
        for _ in range(int(ticks)):
            self.spine.tick()
            self.sim.step(self.spine.decim)
            self.update()
            if self.out.done:
                break
        return self.out

    # -- bookkeeping: call once per decision tick ----------------------------------
    def update(self) -> Outcome:
        out, spec = self.out, self.spec
        out.ticks += 1
        obj = self.truth().obj_pos
        grip = float(self.sim.data.qpos[self.sim.gadr])
        lifted = obj[2] > self.hold_z
        out.tripped = self.spine.guard_state.trips if self.spine.guard_state is not None else ()

        if lifted and not out.released:
            out.held_ticks += 1
            if out.held_ticks == HOLD_TICKS:
                out.reward += spec.reward.hold
        if out.held_ticks >= HOLD_TICKS and not out.released and obj[2] < self.hold_z and self._prev_z > self.hold_z:
            out.dropped = True
            out.done = True
            out.reward += spec.reward.drop
            out.reason = "dropped"
        if grip > OPEN_Q - 0.2:
            # Only counts as a release if the jaws had actually shut first: the arm parks open, and
            # "open" at the start of an episode is not an event.
            if self._was_shut:
                out.released = True
        # Sticky: "the jaws shut at some point" is a memory, not a level. Recomputing it from the
        # live angle made the release unreadable -- the jaws open at the spine's 2.5 rad/s slew,
        # i.e. 0.05 rad per decision tick, so no tick can be both `> OPEN_Q - 0.2` while the *previous*
        # tick was `< OPEN_Q - 0.5`; the window the old test needed is 0.3 rad wide and is never
        # crossed in one tick. `released` therefore stayed False and no pick could ever be scored.
        self._was_shut = self._was_shut or grip < OPEN_Q - 0.5  # a held block stalls the jaws well short of CLOSE_Q
        if out.released and not out.done:
            near = float(np.linalg.norm(obj[:2] - DROP_ZONE)) < DROP_RADIUS
            self._settled = self._settled + 1 if (near and obj[2] < 0.05) else 0
            if self._settled > SETTLE_TICKS:
                out.success = True
                out.done = True
                out.reason = "placed"
        if out.tripped and not out.done:
            out.done = True
            out.reward += spec.reward.guard_trip
            out.reason = "guard:" + ",".join(out.tripped)
        if out.ticks >= spec.max_ticks and not out.done:
            out.done = True
            out.reason = "timeout"
        self._prev_z = float(obj[2])
        return out

    def finish(self, escalated=False, reason="") -> Outcome:
        out, spec = self.out, self.spec
        out.done = True
        out.escalated = bool(escalated)
        if escalated:
            out.reward += spec.reward.escalate_when_right if self.ep.escalate_gt else spec.reward.escalate_when_wrong
        if not out.reason:
            out.reason = reason or ("escalated" if escalated else "timeout")
        return out
