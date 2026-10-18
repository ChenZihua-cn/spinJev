"""Event-sourced decision log: one JSON line per decision tick, append-only.

This is the file the data-collection pipeline writes millions of lines of, so three things it
carries are not optional:

* the **whole** `probs` distribution, not the argmax -- PLAN 5.3. A record that only kept the
  winning primitive cannot be re-used for calibration, and an expert's one-hot and a model's soft
  distribution must both fit into the same field.
* `outcome`, the trajectory's 0/1 result, back-filled once the episode ends. Labels come from the
  physics, never from what the writer intended: the scripted experts fail some of the time. A
  collector holds an episode's rows until the result is known and then writes them all, so `None`
  stays legal -- a run that dies mid-episode still leaves a file that parses.
* `truth`, the privileged state, written because it is *not* an input anywhere. Keeping it beside
  the observation is what makes a reward recomputation or a leaked-feature audit possible later.

`schema` is a version integer stamped on every line, because a GB-scale file cannot be rewritten
when the record changes.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

import numpy as np

from spinjev.bus.schema import Truth

SCHEMA_VERSION = 1
ROUND = 5  # 10 um / 0.03 deg: well inside every tolerance the spine uses, and half the file size


def _vec(x, digits=ROUND):
    return [round(float(v), digits) for v in np.asarray(x, dtype=np.float64).reshape(-1)]


@dataclass
class Decision:
    """One row of the log."""

    episode_id: int
    tick: int
    obs: np.ndarray
    candidates: tuple = ()
    chosen: str = ""
    probs: Dict[str, float] = field(default_factory=dict)
    escalate: bool = False
    escalate_prob: Optional[float] = None
    outcome: Optional[int] = None
    truth: Truth = field(default_factory=Truth)
    t: float = 0.0

    def to_line(self) -> str:
        if self.outcome is not None and int(self.outcome) not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {self.outcome}")
        total = sum(float(p) for p in self.probs.values())
        if not self.probs or abs(total - 1.0) > 1e-3:
            raise ValueError(f"probs must be a distribution over the candidates, sums to {total}")
        if not self.escalate and self.chosen not in self.probs:
            raise ValueError(f"chosen {self.chosen!r} has no probability and did not escalate")
        rec = {
            "schema": SCHEMA_VERSION,
            "episode_id": int(self.episode_id),
            "tick": int(self.tick),
            "t": round(float(self.t), 3),
            "obs": _vec(self.obs),
            "candidates": [str(c) for c in self.candidates],
            "chosen": str(self.chosen),
            "probs": {str(k): round(float(v), 4) for k, v in self.probs.items()},
            "escalate": bool(self.escalate),
            "outcome": None if self.outcome is None else int(self.outcome),
            "truth": {
                "obj_pos": _vec(self.truth.obj_pos, 4),
                "obj_vel": _vec(self.truth.obj_vel, 4),
                "obj_quat": _vec(self.truth.obj_quat, 4),
                "ncontact": int(self.truth.ncontact),
            },
        }
        if self.escalate_prob is not None:
            rec["esc_prob"] = round(float(self.escalate_prob), 4)
        return json.dumps(rec, separators=(",", ":"))


class EventLog:
    """Append-only JSONL writer. One open file per collection worker; array jobs never share one."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, rec: Decision) -> Decision:
        self._f.write(rec.to_line() + "\n")
        return rec

    def flush(self):
        """A run that dies mid-job should still leave readable lines, so durability is opt-in
        per checkpoint rather than per line."""
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self):
        if not self._f.closed:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def read(path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
