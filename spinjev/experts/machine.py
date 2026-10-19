"""The pick sequence the scripted experts run, as one shared phase machine.

`reach`, `direct` and `escalate` are this machine with a different phase list, a different retry
budget and different escalate weights. Keeping it in one place is what stops the three experts
from drifting apart in the parts that must agree: they may only command a primitive that is in the
candidate set, and only a *waypoint* the pre-filter verified.

Phase transitions, from the statuses `Spine` reports (`reached` / `blocked` / `timeout` / `trip:*`):

    close -> lift      the jaws stalled against something: the only grip signal a servo gives
    close -> reopen    the jaws ran fully shut, so nothing was between them; try again
    optional -> skip   a home / hover / retreat waypoint the pre-filter refused is stepped over
    required -> retry  a timeout re-runs the phase, after which the `stalled` feature hands the
                       decision to the escalate head instead of grinding further
"""

import numpy as np

from spinjev.experts.base import Action, Expert
from spinjev.spine.primitives import Primitive

OK = ("reached", "blocked", "held")


class PickMachine(Expert):
    """Shared phase machine; the three experts only differ in the knobs below."""

    #: phase -> (primitive, waypoint, Score bin). "" waypoint = no Cartesian goal (the gripper
    #: heads); "lift_z" = the carry height read off the verified `drop` waypoint, because `Spine`
    #: takes only the z of a LIFT goal.
    PHASES = (
        ("home", Primitive.APPROACH, "home", 2),
        ("hover", Primitive.APPROACH, "hover", 2),
        ("descend", Primitive.APPROACH, "grasp", 3),
        ("close", Primitive.CLOSE, "", 3),
        ("reopen", Primitive.RELEASE, "", 3),
        ("lift", Primitive.LIFT, "lift_z", 4),
        ("carry", Primitive.APPROACH, "drop", 4),
        ("release", Primitive.RELEASE, "", 4),
        ("retreat", Primitive.APPROACH, "home", 5),
        ("settle", None, "", 5),
    )
    #: phases the plan survives without; the others are the task itself
    OPTIONAL = ("home", "hover", "retreat", "settle")
    EXCLUDE = ()
    #: phase the machine starts in. "home" makes the first action an approach to the rest pose;
    #: "hover" starts above the object and leaves the rest pose to the retreat after the release.
    #: The simulator already parks the arm there -- FK at HOME_Q puts the wrist at
    #: (0, -0.306, 0.169), measured, which is *not* TaskSpec.home -- so an opening approach is a
    #: transit through the band the blocks are sampled in, and it tips them: after one home+hover
    #: pair the lowest block centre seen was 12 mm (lying on its side) versus 29 mm upright.
    start = "home"
    max_retry = 2      # attempts on one phase before it is given up on / handed to the head
    stall_after = 2    # ... and the point where `stalled` starts pushing on the escalate head
    grip_after = 2     # ditto for `grip_failed`
    hold_ticks = 10    # what one `hold` action costs the clock

    # -- plan ------------------------------------------------------------------------
    def table(self):
        return [row for row in self.PHASES if row[0] not in self.EXCLUDE]

    def spec(self, phase):
        for name, prim, waypoint, score in self.PHASES:
            if name == phase:
                return prim, waypoint, score
        raise KeyError(f"unknown phase {phase!r}")

    def next_phase(self, phase):
        names = [row[0] for row in self.table()]
        i = names.index(phase)
        return names[min(i + 1, len(names) - 1)]

    def verified(self, dp, waypoint) -> bool:
        return waypoint in dp.legal.waypoints

    # -- what this expert wants right now --------------------------------------------
    def wanted(self, dp) -> Action:
        notes = []
        self.plan_blocked = False
        for _ in range(len(self.PHASES)):
            prim, waypoint, score = self.spec(self.phase)
            skippable = self.phase in self.OPTIONAL
            if waypoint and waypoint != "lift_z" and not self.verified(dp, waypoint):
                why = dp.legal.reasons.get(f"approach:{waypoint}", "?")
                notes.append(f"skip:{self.phase}({why})" if skippable else f"blocked:{waypoint}({why})")
                if skippable:
                    self.observe(notes[-1])
                    self.phase = self.next_phase(self.phase)
                    continue
                # A required waypoint the body cannot reach: do not drive at it. The escalate head
                # owns the decision; `hold` is the only action that touches nothing.
                self.observe(notes[-1])
                self.bump("stuck")
                self.plan_blocked = True
                return Action(hold=True, score=score, note=",".join(notes))
            break
        prim, waypoint, score = self.spec(self.phase)
        goal = None
        if prim is not None and waypoint == "lift_z":
            drop = dp.goal("drop")
            z = float(drop[2]) if drop is not None else float(dp.proprio.ee_pos[2]) + 0.05
            goal = np.array([0.0, 0.0, z])
        elif prim is not None and waypoint:
            goal = np.asarray(dp.goal(waypoint), dtype=np.float64)
        return Action(primitive=prim, waypoint=waypoint, goal=goal, score=score,
                      note=",".join(notes) or None)

    def on_illegal(self, dp, want: Action) -> Action:
        """Re-plan when the pre-filter rejected the *primitive* of the current phase."""
        if want.primitive is Primitive.LIFT:
            self.bump("grip")
            self.observe("no_grip")
            self.phase = "reopen" if self.retries.get("grip", 0) < self.max_retry else "descend"
            return Action(hold=True, score=want.score, note="replan:lift->reopen")
        if want.primitive is Primitive.CLOSE:
            # `already_shut`: the jaws are shut, which is the grip this phase was asking for.
            self.phase = "lift"
            return Action(hold=True, score=want.score, note="replan:close->lift")
        if want.primitive is Primitive.RELEASE:
            self.phase = self.next_phase(self.phase)
            return Action(hold=True, score=want.score, note=f"replan:release({self.phase})")
        return Action(hold=True, score=want.score, note=f"illegal:{want.primitive.value}")

    # -- executor feedback -----------------------------------------------------------
    def report(self, status: str, waypoint: str = ""):
        phase = self.phase
        if status.startswith("trip:"):
            self.observe(f"trip:{status.split(':', 1)[1]}")
            return
        ok = status in OK
        if not ok:
            n = self.bump(phase)
            self.observe(f"{status}:{phase}(#{n})")
        if phase == "close":
            if status == "blocked":
                self.phase = "lift"
            else:
                n = self.bump("grip")
                self.observe("no_grip")
                self.phase = "reopen" if n < self.max_retry else "descend"
            return
        if phase == "reopen":
            self.phase = "descend"
            return
        if phase == "settle":
            return
        if ok:
            self.phase = self.next_phase(phase)
        elif self.phase in self.OPTIONAL and self.retries.get(phase, 0) >= self.max_retry:
            self.phase = self.next_phase(phase)  # give up on the optional phase, keep going
