"""Scripted experts (PLAN 阶段 1c).

An expert is a *policy*, not a script: at every decision point it is handed
(`Proprio`, `Percept`, `Legality`, episode bookkeeping) and answers with either a primitive
chosen from the candidate set or an escalate. It never touches the simulator and never reads
`Truth` -- the only privileged thing it sees is what `Prefilter` already derived from the
observation.

Every answer carries the full distribution (PLAN 五.3): the act-or-escalate probability, the
softmax over the *legal* primitives, and a `Score` distribution. The expert's scores are its own
task-preference logits, deliberately smoothed, so the recorded distribution is a real teacher
signal and not a one-hot with the argmax copied twice.

Three experts, three failure profiles:

    reach     -- home, hover, descend, close, lift, carry, release, retreat. Escalates as soon as
                 the observation says the object cannot be gripped.
    direct    -- skips the hover and descends straight onto the grasp goal. Riskier tolerance for
                 a missing detection, so it also produces "did not escalate in an unwinnable
                 state" samples.
    escalate  -- retreats and reports early. On unwinnable placements it reports on the first
                 look; on winnable ones it reports whenever its routing threshold is crossed,
                 which is what puts the wrong-escalate cases (PLAN 2.6, -0.5) in the dataset.
"""

from spinjev.experts.base import (
    BLOCK,
    PORDER,
    PRIMITIVES,
    SCORE_BINS,
    WAYPOINTS,
    Action,
    Decision,
    DecisionPoint,
    Expert,
    Probabilities,
)
from spinjev.experts.direct import DirectGraspExpert
from spinjev.experts.escalate import EscalateExpert
from spinjev.experts.reach import ReachExpert

EXPERTS = {
    ReachExpert.name: ReachExpert,
    DirectGraspExpert.name: DirectGraspExpert,
    EscalateExpert.name: EscalateExpert,
}

__all__ = [
    "EXPERTS", "Action", "Decision", "DecisionPoint", "Expert", "Probabilities",
    "PRIMITIVES", "PORDER", "SCORE_BINS", "WAYPOINTS", "BLOCK",
    "ReachExpert", "DirectGraspExpert", "EscalateExpert",
]
