"""The stage-1 perception simulator (PLAN 2.5, first bullet).

A pose estimator, faked: true position plus zero-mean Gaussian error, with the error ~10x bigger
when the target is hidden, plus a detection dropout. The noise levels are stated design choices
chosen to straddle the task's own tolerances -- a 10 mm sigma is half the 20 mm hover error the
spine already tolerates (REACHED_TOL), and 100 mm is several block widths -- not measurements of
any camera.

`truth` is read *only* to synthesize what the pixels would have shown; nothing here is a channel
into the observation beyond the pose and the detector's own confidence.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from spinjev.bus.schema import Percept, Truth

SIGMA_CLEAR = 0.010  # m, sigma per axis with the target in view
SIGMA_OCCLUDED = 0.100  # m, ~an order of magnitude worse (PLAN 2.5)
MISS_RATE = 0.45  # fraction of occluded ticks where the detector reports nothing at all
CONF_CLEAR = 0.90  # detector self-reported confidence when the target is visible
CONF_OCCLUDED = 0.30  # ... and when it is guessing at a partly hidden target
CONF_MISSING = 0.05  # ... and when it reports no detection
CONF_JITTER = 0.05
CONF_FLOOR = 0.02


@dataclass
class Perceptor:
    """`perceptor(truth, occluded)` -> `Percept`, the shape `PickPlace(perceptor=...)` expects.

    Draw order is fixed (error first, then the dropout roll, then the confidence), so the same
    seed always gives the same stream. `reset()` re-seeds it and forgets the tracked pose, which
    is what an episode boundary wants.
    """

    sigma_clear: float = SIGMA_CLEAR
    sigma_occluded: float = SIGMA_OCCLUDED
    miss_rate: float = MISS_RATE
    conf_clear: float = CONF_CLEAR
    conf_occluded: float = CONF_OCCLUDED
    conf_missing: float = CONF_MISSING
    conf_jitter: float = CONF_JITTER
    seed: int = 0
    rng: Optional[np.random.Generator] = field(default=None, repr=False)

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)
        self._last = np.zeros(3)

    def reset(self, seed: Optional[int] = None) -> "Perceptor":
        """Re-seed for a reproducible episode and drop the stale track."""
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._last = np.zeros(3)
        return self

    def sigma(self, occluded: bool) -> float:
        """The noise level the next call will use -- evaluation needs it to score a pose error
        against the sigma that produced it instead of against a fixed pixel budget."""
        return float(self.sigma_occluded if occluded else self.sigma_clear)

    def __call__(self, truth: Truth, occluded: bool = False, rng=None) -> Percept:
        # `occluded` stands in for blocked pixels: it sets the noise level only, the flag itself
        # never reaches the student -- that is the whole point of PLAN 2.5.
        rng = self.rng if rng is None else rng
        sigma = self.sigma(occluded)
        pos = np.asarray(truth.obj_pos, dtype=np.float64) + sigma * rng.standard_normal(3)
        if occluded and float(rng.random()) < self.miss_rate:
            # a miss keeps the last detection: no new pixels, no new number -- zeros would read as motion
            return Percept(pos=self._last.copy(), valid=False, confidence=self.conf_missing)
        base = self.conf_occluded if occluded else self.conf_clear
        conf = float(np.clip(base + self.conf_jitter * rng.standard_normal(), CONF_FLOOR, 1.0))
        self._last = pos.copy()
        return Percept(pos=pos, valid=True, confidence=conf)
