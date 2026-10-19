"""Shared machinery for the scripted experts: distributions, the candidate-set mask, and the
act-or-escalate head.

The two rules this file exists to enforce:

1. **Never propose an illegal primitive.** `Choice` is masked to `Legality.legal` before the
   softmax, so an out-of-candidate action is not representable (PLAN 2.3).
2. **Never read `Truth`.** A decision point carries `Proprio`, `Percept`, the pre-filter's output,
   the waypoints derived from the *estimated* object pose, and the arm's own observable history.
   Outcome flags derived from the simulated block (`success`, `dropped`, `held_ticks`) are not
   handed to the expert; they belong to the record, after the decision.

The recorded distribution is the expert's own: task-preference logits smoothed by a temperature,
plus a logistic act-or-escalate head over observable features. It is a teacher signal for BC
(PLAN 2.8), not a calibrated posterior -- calibration is stage 3's job and comes from the student.
"""

from dataclasses import dataclass, field

import numpy as np

from spinjev.bus.schema import Percept, Proprio
from spinjev.spine.prefilter import Legality
from spinjev.spine.primitives import Primitive

#: Fixed order of the `Choice` vocabulary. The spine collapses the seven names of PLAN 2.3 onto
#: these four controllers, so this is the whole action space the head can emit.
PRIMITIVES = (Primitive.APPROACH, Primitive.CLOSE, Primitive.LIFT, Primitive.RELEASE)
PORDER = tuple(p.value for p in PRIMITIVES)
#: Waypoint names the task layer hands the pre-filter (`PickPlace.goals`).
WAYPOINTS = ("home", "hover", "grasp", "drop")
#: `Prefilter.waypoint` rejections that mean "the body cannot get there". This is the observable
#: half of the escalate ground truth, since APPROACH stays legal while *any* waypoint is legal.
BLOCK = ("ik", "orientation", "limits", "torque", "table")
SCORE_BINS = 5
#: `Score` is an ordinal 1-5 progress bin (PLAN 2.3).
SCORE_MEANING = {1: "stuck", 2: "reaching", 3: "grasping", 4: "carrying", 5: "delivered"}
def softmax(values, temperature=1.0):
    """Numerically stable, with the expert's own temperature in front of it."""
    x = np.asarray(values, dtype=np.float64) / max(float(temperature), 1e-6)
    if x.size == 0:
        return x
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def sigmoid(x):
    return 1.0 / (1.0 + float(np.exp(-float(np.clip(x, -40.0, 40.0)))))


@dataclass
class Probabilities:
    """Everything PLAN 五.3 asks for: the full distribution, not just the argmax."""

    escalate: float = 0.0
    primitives: dict = field(default_factory=dict)  # name -> prob; names outside the set are 0.0
    logits: dict = field(default_factory=dict)      # name -> pre-mask logit
    score: list = field(default_factory=list)       # softmax over SCORE_BINS ordinal bins
    temperature: float = 0.3
    score_temperature: float = 0.9
    features: dict = field(default_factory=dict)    # escalate-head inputs, for calibration audits

    def as_record(self):
        return {
            "escalate_prob": round(float(self.escalate), 6),
            "primitive_probs": {k: round(float(v), 6) for k, v in self.primitives.items()},
            "primitive_logits": {k: round(float(v), 4) for k, v in self.logits.items()},
            "score_probs": [round(float(v), 6) for v in self.score],
            "temperature": round(float(self.temperature), 4),
            "score_temperature": round(float(self.score_temperature), 4),
            "escalate_features": {k: round(float(v), 3) for k, v in self.features.items()},
        }


@dataclass
class DecisionPoint:
    """What an expert is allowed to see at one decision event."""

    proprio: Proprio
    percept: Percept
    legal: Legality
    goals: dict
    vector: np.ndarray = None      # the rendered observation, carried so a record needs one pass
    tick: int = 0
    trips: tuple = ()
    events: tuple = ()
    released: bool = False

    @property
    def seen(self) -> bool:
        return bool(self.percept.valid)

    @property
    def gripped(self) -> bool:
        """The pre-filter's own observable grasp gate: something is between the jaws."""
        return Primitive.LIFT in self.legal.legal

    def blocked(self, waypoint) -> bool:
        """True when the *object* waypoint specifically cannot be executed."""
        return self.legal.reasons.get(f"approach:{waypoint}", "ok") in BLOCK

    def goal(self, waypoint):
        return self.goals.get(waypoint)


@dataclass
class Action:
    """The emitted `Choice` + its goal source, or an escalate, or a hold."""

    primitive: Primitive = None
    waypoint: str = ""
    goal: np.ndarray = None
    escalate: bool = False
    hold: bool = False
    score: int = 2
    note: str = ""

    def as_record(self):
        return {
            "primitive": None if self.primitive is None else self.primitive.value,
            "waypoint": self.waypoint or None,
            "goal": None if self.goal is None else [round(float(v), 5) for v in np.asarray(self.goal)],
            "escalate": bool(self.escalate),
            "hold": bool(self.hold),
            "score": int(self.score),
            "note": self.note or None,
        }


@dataclass
class Decision:
    action: Action
    probs: Probabilities
    candidates: tuple = ()


class Expert:
    """Base class: masking, distributions, and the escalate head's plumbing.

    Subclasses own the phase machine (`wanted`, `report`) and may reweight `ESCALATE_W` or lower
    `tau` to be bolder or more nervous than their siblings.
    """

    name = "expert"
    #: logit weight per observable feature of `escalate_features`
    ESCALATE_W = {
        "unseen": 5.0,             # the detector does not have the object at all
        "object_unreachable": 5.0,  # the grasp waypoint fails on physics
        "plan_blocked": 5.0,        # the phase the plan needs right now has no legal waypoint
        "grip_failed": 3.0,         # closed twice onto nothing
        "stalled": 2.0,             # some phase has timed out twice
        "drop_unreachable": 4.0,    # holding the block but cannot deliver it
        "tripped": 4.0,             # the guard just said no
        "unreliable": 3.0,          # the detector is confident about something it cannot see
    }
    #: below this detector confidence the pose estimate is not worth acting on
    conf_min = 0.60
    tau = 0.5            # routing threshold: report when P(escalate) > tau
    temperature = 0.80   # sharpness of the `Choice` head; 0.8 keeps the runner-up at ~7-10%
    score_temperature = 0.9
    start = "home"       # phase the machine starts in (PickMachine documents the two options)
    drop_radius = 0.07   # DROP_ZONE radius of the task layer, as a distance on the table
    #: escalate-head intercept. A healthy state has every feature at 0, so this *is* P(escalate)
    #: when nothing is wrong: -0.7 -> 0.33, which is under every expert's tau. A base of 0 would
    #: sit exactly on the default threshold.
    bias = -0.7
    stall_after = 2      # timeouts on one phase before `stalled` pushes on the head
    grip_after = 2       # closed-onto-nothing attempts before `grip_failed` does
    PRIOR = {"approach": 0.30, "close": 0.10, "lift": 0.05, "release": 0.00}  # standing task preference
    BOOST = 1.70         # added to the primitive this phase wants

    def __init__(self, seed=0, **overrides):
        self.seed = int(seed)
        self.ep = None
        self.rng = np.random.default_rng([self.seed, 0])
        for key, val in overrides.items():
            if not hasattr(self, key):
                raise AttributeError(f"{type(self).__name__} has no knob {key!r}")
            setattr(self, key, val)
        self.reset(self.ep)

    # -- lifecycle -----------------------------------------------------------------
    def reset(self, ep, episode_index=0):
        """New episode: rewind the machine. `ep` is the task layer's `Episode`."""
        self.ep = ep
        self.rng = np.random.default_rng([self.seed, int(episode_index), 7919])
        self.phase = self.start
        self.events = []
        self.retries = {}
        self.plan_blocked = False
        self.tau_ep = float(self.tau)
        return self

    def book(self) -> dict:
        """Expert-visible memory, stored with the sample. This is the "recent abnormal events"
        channel of PLAN 2.7 -- the agent's own history, not a truth channel."""
        return {
            "phase": self.phase,
            "events": list(self.events),
            "retries": dict(self.retries),
            "plan_blocked": bool(self.plan_blocked),
            "tau": round(float(self.tau_ep), 4),
        }

    def observe(self, event: str):
        self.events.append(event)

    def bump(self, key: str) -> int:
        self.retries[key] = self.retries.get(key, 0) + 1
        return self.retries[key]

    # -- the candidate set and the `Choice` head --------------------------------------
    def candidates(self, dp: DecisionPoint) -> list:
        return [p for p in PRIMITIVES if p in dp.legal.legal]

    def wanted(self, dp: DecisionPoint) -> Action:
        """What this expert would do if everything were legal. Subclasses own the phase machine."""
        raise NotImplementedError

    def primitive_probs(self, dp: DecisionPoint, want: Action) -> tuple:
        """Logits -> softmax over the *legal* subset; illegal names are recorded as 0.0."""
        logits = dict(self.PRIOR)
        if want.primitive is not None:
            logits[want.primitive.value] += self.BOOST
        allowed = [p.value for p in self.candidates(dp)]
        probs = dict.fromkeys(PORDER, 0.0)
        if allowed:
            probs.update(zip(allowed, (float(v) for v in softmax([logits[n] for n in allowed], self.temperature))))
        return logits, probs

    # -- the act-or-escalate head -----------------------------------------------------
    def escalate_features(self, dp) -> dict:
        """Observable inputs only. `object_unreachable` is the far / folded-in unwinnable branch,
        `unseen` the occluded one; the rest are "my own plan stopped working" signals."""
        gripped = dp.gripped
        return {
            "unseen": 0.0 if dp.seen else 1.0,
            "object_unreachable": 1.0 if (dp.seen and not gripped and dp.blocked("grasp")) else 0.0,
            # The plan's own "I have nowhere to go" signal. `object_unreachable` is gated on not
            # being gripped (it must not fire mid-carry), which lets a spurious grasp flag hide an
            # unwinnable placement; this one is gated on nothing but the pre-filter's verdict.
            "plan_blocked": 1.0 if self.plan_blocked else 0.0,
            "grip_failed": 1.0 if self.retries.get("grip", 0) >= self.grip_after and not gripped else 0.0,
            "stalled": 1.0 if max(self.retries.values(), default=0) >= self.stall_after else 0.0,
            "drop_unreachable": 1.0 if (gripped and dp.blocked("drop")) else 0.0,
            "tripped": 1.0 if dp.trips else 0.0,
            # The confidence channel of PLAN 2.5 is what makes the *occluded* branch observable at
            # all: the simulator keeps reporting a pose there, just a bad one (sigma 100 mm against
            # a 10 mm task tolerance), so `valid` alone misses most of the branch.
            "unreliable": 1.0 if (dp.seen and not gripped
                                  and float(dp.percept.confidence) < self.conf_min) else 0.0,
        }

    def escalate_prob(self, dp) -> tuple:
        """P(escalate). A delivered episode is pinned to 0: reporting after the block is down in the
        target zone is not a decision. `no candidates` is pinned to 1: there is nothing else to do.

        "Delivered" is the *estimate* sitting in the drop zone after the jaws opened -- observable,
        and it must not be replaced by the task layer's `released` flag alone: the flag only says the
        jaws shut at some point and then opened, which also happens when the gripper closes on air,
        and pinning on that made a grinding episode unable to report ever again (measured in the
        first pilot: one episode ran to the event budget with P(escalate) pinned at 0.00).
        """
        feats = self.escalate_features(dp)
        if self.delivered(dp):
            feats["delivered"] = 1.0
            return 0.0, feats
        feats["delivered"] = 0.0
        if not self.candidates(dp):
            feats["no_candidates"] = 1.0
            return 1.0, feats
        feats["no_candidates"] = 0.0
        total = float(self.bias)
        total += sum(w * feats[name] for name, w in self.ESCALATE_W.items() if name in feats)
        return sigmoid(total), feats

    def delivered(self, dp) -> bool:
        """Jaws opened, and the estimated block now sits inside the drop radius of the drop zone."""
        drop = dp.goal("drop")
        return bool(
            dp.released and drop is not None and dp.seen
            and np.linalg.norm(np.asarray(dp.percept.pos)[:2] - np.asarray(drop)[:2]) < self.drop_radius
        )

    def score_probs(self, peak: int) -> list:
        bins = np.arange(1, SCORE_BINS + 1, dtype=np.float64)
        return [float(v) for v in softmax(-np.abs(bins - peak), self.score_temperature)]

    # -- the one call the collector makes --------------------------------------------
    def decide(self, dp: DecisionPoint) -> Decision:
        want = self.wanted(dp)
        logits, probs = self.primitive_probs(dp, want)
        p_esc, feats = self.escalate_prob(dp)
        action = want
        if p_esc > self.tau_ep:
            action = Action(escalate=True, score=want.score, note=f"tau {p_esc:.3f}>{self.tau_ep:.2f}")
        elif want.primitive is not None and want.primitive not in dp.legal.legal:
            # A wanted primitive the pre-filter rejected is never emitted: fall back instead.
            action = self.on_illegal(dp, want)
        return Decision(
            action=action,
            probs=Probabilities(
                escalate=p_esc, primitives=probs, logits=logits, score=self.score_probs(want.score),
                temperature=float(self.temperature), score_temperature=float(self.score_temperature),
                features=feats,
            ),
            candidates=tuple(p.value for p in self.candidates(dp)),
        )

    def on_illegal(self, dp: DecisionPoint, want: Action) -> Action:
        return Action(hold=True, score=want.score, note=f"illegal:{want.primitive.value}")

    # -- executor feedback -----------------------------------------------------------
    def report(self, status: str, waypoint: str = ""):
        """Called by the collector after one `drive()` returns, with the spine's status."""
        raise NotImplementedError
