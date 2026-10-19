"""The reach expert: the careful one, and the reference for what "correct" looks like.

home -> hover -> descend -> close -> lift -> carry -> release -> retreat.

It reads the pre-filter's per-waypoint reasons, so the physically-unwinnable branch is a decision
and not a coincidence: `approach:grasp` rejecting with ik / limits / orientation / torque / table
means *the object itself cannot be gripped*, while APPROACH stays legal because the home and drop
waypoints still are. That distinction is the whole point of PLAN 2.4 -- if the expert instead
grinds at an unreachable block, the failure is uninformative; if it reports, the sample is labelled
escalate-correct against the physics.

This expert is deliberately the least willing to report on its own: on a winnable placement it has
no reason to, so nearly every one of its escalates is a right one.
"""

from spinjev.experts.machine import PickMachine


class ReachExpert(PickMachine):
    name = "reach"
    description = "hover above the object, descend, close, lift, carry, release, retreat"
    #: the arm is already parked at HOME_Q, so the plan opens above the object and leaves the rest
    #: pose to the retreat after the release (PickMachine.start documents why)
    start = "hover"
