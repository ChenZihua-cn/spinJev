"""From one decision point to one record, rendered through the bus.

Two integrations matter here, and neither is mine to duplicate:

* `bus.compressor.Compressor` renders the fixed-length observation vector, over the 2 s window in
  `bus.ringbuffer`. A decision sample therefore carries *exactly* the vector the student is trained
  and evaluated on; the alternative was a second, subtly different renderer inside the sampling
  path, which is how a dataset ends up with features the policy never sees.
* `bus.eventlog.Decision` is the log row schema. Every record is built through its `to_line()`
  validator (a distribution over the candidates, the chosen name inside it, outcome in {0,1,None})
  and then carries the stage-1c extras on top: which expert chose, the full `Score` distribution,
  the pre-filter's per-waypoint reasons, the commanded waypoint and goal, the escalate head's
  features, and the episode's branch and outcome.

The iron rule of PLAN 2.7 stays structural: `snapshot` reads `env.observe()` (proprio, percept,
vector) and nothing else, and the only place `Truth` is read is `truth_record`, which writes the
record's privileged `truth` field.
"""

import json

import numpy as np

from spinjev.bus.compressor import Compressor
from spinjev.bus.eventlog import SCHEMA_VERSION
from spinjev.bus.eventlog import Decision as LogDecision
from spinjev.bus.ringbuffer import RingBuffer
from spinjev.bus.schema import OBS_DIM, OBS_LAYOUT, Truth
from spinjev.experts.base import DecisionPoint
from spinjev.spine.primitives import Primitive
from spinjev.tasks.pick_place import PickPlace

#: Tag on every record: the bus schema version plus the stage-1c extras.
DATASET = "spinjev-expert-decisions"
ROUND = 5
#: Decision ticks a `hold` action costs the clock (200 ms at 50 Hz).
HOLD_TICKS = 10
SECTION_NAMES = tuple(name for name, _ in OBS_LAYOUT)
SECTION_DIMS = {name: dim for name, dim in OBS_LAYOUT}


def _vec(x, digits=ROUND):
    return [round(float(v), digits) for v in np.asarray(x, dtype=np.float64).reshape(-1)]


def truth_record(truth: Truth) -> dict:
    """The privileged block. Rewards and audits read it; no feature pipeline may."""
    return {
        "obj_pos": _vec(truth.obj_pos, 4),
        "obj_vel": _vec(truth.obj_vel, 4),
        "obj_quat": _vec(truth.obj_quat, 4),
        "ncontact": int(truth.ncontact),
    }


def ep_record(ep, out) -> dict:
    """Branch flags and the physics' verdict -- the labels, kept apart from the observation."""
    return {
        "object_xy": [round(float(v), 5) for v in ep.object_xy],
        "radius": round(float(ep.radius), 4),
        "unreachable": bool(ep.unreachable),
        "occluded": bool(ep.occluded),
        "escalate_gt": bool(ep.escalate_gt),
        "success": bool(out.success),
        "escalated": bool(out.escalated),
        "released": bool(out.released),
        "dropped": bool(out.dropped),
        "tripped": list(out.tripped),
        "held_ticks": int(out.held_ticks),
        "ticks": int(out.ticks),
        "reward": round(float(out.reward), 4),
        "reason": out.reason or "unfinished",
        "escalate_correct": bool(out.escalated) == bool(ep.escalate_gt),
    }


class RecordingEnv(PickPlace):
    """`PickPlace` wired to the bus: a 2 s proprioceptive window plus the compressor's vector.

    `update()` is the task layer's once-per-decision-tick hook, and every path that moves the arm
    (`drive`, `wait`) goes through it, so pushing the frame here puts *every* tick in the ring, not
    only the ones a sample happened to be taken on. Nothing about the bookkeeping changes: `super()`
    does all of it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ring = RingBuffer()
        self.compressor = Compressor(self.spine.guard.qlo, self.spine.guard.qhi)

    def reset(self, *args, **kwargs):
        ep = super().reset(*args, **kwargs)
        self.ring.clear()
        self.compressor.reset()
        if self.perceptor is not None and hasattr(self.perceptor, "reset"):
            self.perceptor.reset()          # the tracker forgets the object at an episode boundary
        return ep

    def update(self):
        self.ring.push(self.sense())
        return super().update()

    def observe(self):
        """(proprio, percept, observation vector) for this tick, with the frame already in the window."""
        proprio = self.sense()
        percept = self.percept()
        return proprio, percept, self.compressor(self.ring, proprio, percept)


def snapshot(env, expert) -> DecisionPoint:
    """The decision point: observable state, then the waypoints derived from it.

    `goals` and the pre-filter are called here instead of through `PickPlace.candidates()` so the
    waypoints handed to the expert are the same ones the candidate set was judged on -- one IK pass
    per decision, and no chance of the two disagreeing.
    """
    proprio, percept, vector = env.observe()
    goals = env.goals(percept)
    legal = env.prefilter.legal(proprio, percept, goals)
    return DecisionPoint(
        proprio=proprio, percept=percept, legal=legal, goals=goals, vector=vector,
        tick=int(env.out.ticks), trips=tuple(env.out.tripped), events=tuple(expert.events),
        released=bool(env.out.released),
    )


def section_of(vector, name) -> np.ndarray:
    """Slice one named section out of a rendered observation vector."""
    lo = sum(SECTION_DIMS[n] for n in SECTION_NAMES[:SECTION_NAMES.index(name)])
    return np.asarray(vector, dtype=np.float64)[lo:lo + SECTION_DIMS[name]]


def decision_record(*, shard, seed, episode_index, decision, expert, event, terminal, dp,
                    choice, book) -> dict:
    """One sample: the bus log row, validated, plus the stage-1c extras."""
    vector = np.asarray(dp.vector, dtype=np.float64)
    assert vector.shape == (OBS_DIM,), f"obs vector {vector.shape} != OBS_DIM {OBS_DIM}"
    assert np.all(np.isfinite(vector)), "non-finite observation"
    action, probs = choice.action, choice.probs
    cands = tuple(sorted(choice.candidates))
    chosen = "" if action.primitive is None else action.primitive.value
    if cands:
        # The bus validator: a distribution over the candidates, and the chosen name among them.
        rec = json.loads(LogDecision(
            episode_id=int(episode_index), tick=int(dp.tick), obs=vector, candidates=cands,
            chosen=chosen, probs={k: float(v) for k, v in probs.primitives.items() if v > 0.0},
            escalate=bool(action.escalate), escalate_prob=float(probs.escalate),
            outcome=None, truth=Truth(), t=float(dp.proprio.t),
        ).to_line())
    else:
        # Nothing is legal, so there is no candidate distribution to validate and reporting is the
        # only sane action. Written directly; the head does force P(escalate) = 1 here.
        rec = {
            "schema": SCHEMA_VERSION, "episode_id": int(episode_index), "tick": int(dp.tick),
            "t": round(float(dp.proprio.t), 3), "obs": _vec(vector), "candidates": [],
            "chosen": "", "probs": {}, "escalate": True,
            "esc_prob": round(float(probs.escalate), 4), "outcome": None, "truth": None,
        }
    rec.update({
        "dataset": DATASET,
        "key": f"s{shard}-e{episode_index}-d{decision}",
        "shard": int(shard),
        "seed": int(seed),
        "episode_index": int(episode_index),
        "decision": int(decision),
        "expert": expert,
        "event": event,             # switch | hold | escalate | guard_trip | terminal | budget
        "terminal": bool(terminal),
        "obs_dim": int(OBS_DIM),
        "candidate_size": len(cands),
        "prefilter_reasons": {k: str(v) for k, v in sorted(dp.legal.reasons.items())},
        "percept": {"valid": bool(dp.percept.valid), "confidence": round(float(dp.percept.confidence), 4)},
        "book": book,               # the expert's own phase / event memory
        "action": action.as_record(),
        "probs_primitive": {k: round(float(v), 6) for k, v in probs.primitives.items()},
        "probs_primitive_logits": {k: round(float(v), 4) for k, v in probs.logits.items()},
        "probs_score": [round(float(v), 6) for v in probs.score],
        "escalate_features": {k: round(float(v), 3) for k, v in probs.features.items()},
        "temperatures": {"choice": round(float(probs.temperature), 4),
                         "score": round(float(probs.score_temperature), 4)},
        "episode_reward": None,     # trajectory return, filled in when the episode flushes
        "event_reward": None,       # the PLAN 2.6 term that landed on this event
        "ep": None,                 # branch + outcome, filled in when the episode flushes
        "truth": None,              # filled in when the episode flushes: PRIVILEGED
    })
    return rec


def selfcheck(seed=7) -> dict:
    """Prove the two properties the dataset is judged on, without a simulator.

    1. the object channel of the vector is the *estimate*, never the truth -- and it is the bus
       compressor that renders it, so the sampling path and the student agree by construction;
    2. a primitive outside the candidate set has probability exactly 0, and the head never emits one.
    """
    from spinjev.bus.schema import Proprio
    from spinjev.perception import Perceptor
    from spinjev.spine.prefilter import Legality
    from spinjev.experts.reach import ReachExpert

    true_pos = np.array([0.000, -0.280, 0.030])
    proprio = Proprio(
        t=0.020, q=np.array([0.0, 0.6, -0.8, 0.0, 0.0, 1.8]), qvel=np.zeros(6), tau=np.zeros(6),
        ee_pos=np.array([0.0, -0.260, 0.160]), grasp_pos=np.array([0.0, -0.350, 0.070]),
    )
    truth = Truth(obj_pos=true_pos.copy(), obj_vel=np.zeros(3), ncontact=1)

    perceptor = Perceptor(seed=seed)                 # the shipped model: sigma_clear = 10 mm
    noisy = perceptor(truth, occluded=False)
    ring = RingBuffer()
    ring.push(proprio)
    vector = Compressor(np.full(6, -2.0), np.full(6, 2.0))(ring, proprio, noisy)
    got = section_of(vector, "obj_pos")
    if np.allclose(got, true_pos):
        raise AssertionError("the object channel carries Truth, not the estimate")
    if not np.allclose(np.asarray(noisy.pos, dtype=np.float64), got):
        raise AssertionError("the object channel is neither the estimate nor its rounding")
    if not np.allclose(section_of(vector, "obj_conf"), [noisy.confidence if noisy.valid else 0.0]):
        raise AssertionError("the confidence channel does not carry the detector's own number")

    legal = Legality(
        legal=frozenset({Primitive.APPROACH, Primitive.CLOSE}),
        reasons={"approach": "ok", "close": "ok", "release": "already_open",
                 "lift": "nothing_gripped"},
        waypoints={"home": np.zeros(5), "hover": np.zeros(5), "grasp": np.zeros(5)},
    )
    goals = {"home": np.array([0.0, -0.26, 0.16]), "hover": true_pos + np.array([0.0, 0.0, 0.12]),
             "grasp": true_pos + np.array([0.0, 0.0, 0.09]), "drop": np.array([0.12, -0.22, 0.15])}
    dp = DecisionPoint(proprio=proprio, percept=noisy, legal=legal, goals=goals, vector=vector, tick=1)
    choice = ReachExpert(seed=0).decide(dp)
    probs = choice.probs.primitives
    if probs["lift"] != 0.0 or probs["release"] != 0.0:
        raise AssertionError(f"illegal primitives must have probability 0.0, got {probs}")
    if abs(sum(probs.values()) - 1.0) > 1e-9:
        raise AssertionError(f"primitive probabilities do not sum to 1: {probs}")
    if choice.action.escalate or choice.action.primitive not in legal.legal:
        raise AssertionError(f"a healthy state with legal candidates must act legally: {choice.action}")
    rec = decision_record(shard=0, seed=0, episode_index=0, decision=0, expert="reach",
                          event="switch", terminal=False, dp=dp, choice=choice, book={})
    if rec["probs_primitive"] != {k: round(float(v), 6) for k, v in probs.items()} or rec["obs"] != _vec(vector):
        raise AssertionError("the record lost part of the distribution or the observation")
    return {
        "obs_dim": int(OBS_DIM),
        "sections": list(SECTION_NAMES),
        "renderer": "spinjev.bus.compressor.Compressor",
        "perceptor": type(perceptor).__module__ + "." + type(perceptor).__name__,
        "obj_channel_is_estimate": True,
        "estimate_minus_truth_mm": round(float(np.linalg.norm(np.asarray(noisy.pos) - true_pos) * 1000), 2),
        "illegal_primitives_zero": True,
        "probs_sum": round(float(sum(probs.values())), 9),
        "recorded_distribution": {k: round(float(v), 6) for k, v in probs.items()},
        "escalate_prob": round(float(choice.probs.escalate), 6),
        "score_probs": [round(float(v), 6) for v in choice.probs.score],
        "record_fields": len(rec),
    }
