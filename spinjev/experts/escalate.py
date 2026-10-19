"""The escalate / retreat expert: reports first, and sometimes reports when it should not.

On an unwinnable placement this is the expert that gives the cleanest positive label: it goes to the
rest pose, looks at the pre-filter once, and reports without ever touching the block. Because it
starts at `home` and only continues when its head is satisfied, the escalate event lands on the very
first or second decision of the episode.

On a *winnable* placement it still reports occasionally -- that is the point. PLAN 2.6 prices a
wrong escalate at -0.5 precisely so that "report" cannot become a reflex, and BC needs those
negative examples in the data or the student never sees a report that cost anything. The mechanism
is not a coin flip on the outcome: the per-episode routing threshold is drawn from a range (τ is a
config knob, not a constant), and this expert's τ distribution sits low enough that a clean state
with P(escalate)=0.18 still trips it sometimes. Which episodes it trips is unknown to the expert
and known to the physics, so the label comes from the branch flag, not from its own intent.
"""

import numpy as np

from spinjev.experts.machine import PickMachine


class EscalateExpert(PickMachine):
    name = "escalate"
    description = "report early; the retreat to the rest pose is what it does when it gets that far"
    #: Opens above the object like the reach expert. Starting at the rest pose instead was tried:
    #: 5 of 30 episodes then died in a fatal table trip during that very first transit (a joint-space
    #: path dip, which is a 阶段 1a property, not this expert's), which would have made the trips --
    #: not the head -- this expert's distinguishing feature. It still emits `retreat`: the phase is
    #: the last step of the plan, and reaching it at all is what "not reporting" looks like.
    start = "hover"
    #: impatient about everything: one stall is enough, and a guard trip more than enough
    max_retry = 1
    stall_after = 1
    grip_after = 1
    tau = 0.35
    ESCALATE_W = {
        "unseen": 6.0,
        "object_unreachable": 6.0,
        "plan_blocked": 6.0,
        "grip_failed": 2.0,
        "stalled": 1.5,
        "drop_unreachable": 5.0,
        "tripped": 2.0,
        "unreliable": 4.0,
    }
    #: Per-episode escalate-head intercept, and the whole reason this expert is in the set. A clean
    #: state has every feature at 0, so P(escalate) = sigmoid(bias) and the expert reports at once
    #: iff sigmoid(bias) > tau. With tau = 0.35 and this range that is P = (−0.2 − logit(0.35)) /
    #: (−0.2 − (−2.5)) = 0.419 / 2.3 = 18% of episodes: a deliberate 18% wrong-escalate rate on
    #: winnable placements (PLAN 2.6 prices it at -0.5), and 0% of them by luck -- the draw is
    #: derived from (seed, episode), so it is reproducible.
    bias_range = (-2.5, -0.2)

    def reset(self, ep, episode_index=0):
        super().reset(ep, episode_index)
        self.bias = float(self.rng.uniform(*self.bias_range))
        return self
