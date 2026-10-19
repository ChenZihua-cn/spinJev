"""Stand-in for `spinjev/perception` (owned elsewhere) with the signature the task layer already
defines: `PickPlace.percept()` calls `self.perceptor(truth, occluded=self.ep.occluded)`.

Why this file exists at all: with `perceptor=None` the task layer hands back the *simulated* block
pose, so (a) the observation's object channel is privileged state, which is the one thing PLAN 2.7
forbids, and (b) the occluded branch of PLAN 2.4 has no observable consequence at all -- the block
is where it always was, the pre-filter says the grasp is legal, and "escalate" would be correct
only in a way nothing can see. This stub makes the 20% branch decidable from the observation, which
is the minimum the escalate labels need to mean anything.

It implements PLAN 2.5's stage-1 option, "not detected + a detection confidence", and nothing more:

    clear      pos = truth + N(0, sigma) per axis (sigma=0 -> the identity path), valid, conf 1.0
    occluded   valid=False, conf `occluded_confidence`, pos = the last estimate it actually had,
               or the workspace prior on the first look -- a tracker that lost the object reports
               where it last saw it, not where the object is.

Replace it by pointing `--perceptor` at the real module once it lands; nothing here is imported by
the experts, so the swap is one line in the collector.

One measured number belongs in this file, because it decides what `sigma` may ever be: the scripted
experts collapse under perception noise. 10 winnable episodes per cell, reach expert, mujoco 3.16,
`--obj-sigma` swept on the object channel only:

    sigma = 0      -> 6 of 10 placed   (the value the shipped dataset uses)
    sigma = 0.005  -> 3 of 10 placed, 6 reports
    sigma = 0.010  -> 1 of 10 placed, 9 reports
    sigma = 0.020  -> 0 of 10 placed, 10 reports

A 5 mm error on the *estimate* already halves the pick rate, because the estimate is what the grasp
waypoint is built from and the grip basin is ~10 mm wide. So the noise model is not a detail: until
the spine closes onto a *searched* pose instead of a single believed point, `sigma` has to stay at
or below a few millimetres, or the experts need a "look again / refine" phase.
"""

import numpy as np

from spinjev.bus.schema import Percept, Truth

#: Where a detector that has never seen the object would put it: the middle of the table the arm
#: works over. It is a prior, not a measurement, and `valid=False` says so.
PRIOR_XYZ = np.array([0.0, -0.27, 0.03])


class StubPerceptor:
    """Callable perceptor: `(truth, occluded=False) -> Percept`."""

    def __init__(self, seed=0, sigma=0.0, occluded_confidence=0.25, prior=PRIOR_XYZ):
        self.sigma = float(sigma)
        self.occluded_confidence = float(occluded_confidence)
        self.prior = np.asarray(prior, dtype=np.float64)
        self.rng = np.random.default_rng([int(seed), 104729])
        self.last = None
        self.calls = 0

    def reset(self):
        """New episode: the tracker has not seen anything yet. The collector calls this."""
        self.last = None
        self.calls = 0
        return self

    def __call__(self, truth: Truth, occluded: bool = False) -> Percept:
        self.calls += 1
        if occluded:
            pos = self.last if self.last is not None else self.prior.copy()
            return Percept(pos=np.asarray(pos, dtype=np.float64).copy(), valid=False,
                           confidence=self.occluded_confidence)
        pos = np.asarray(truth.obj_pos, dtype=np.float64)
        if self.sigma > 0.0:
            pos = pos + self.rng.normal(0.0, self.sigma, 3)
            pos[2] = max(0.0, float(pos[2]))
        self.last = pos.copy()
        return Percept(pos=pos.copy(), valid=True, confidence=1.0)
