"""The direct-grasp expert: no hover, straight down onto the grasp goal.

Skipping the pre-grasp waypoint is what a fast, over-confident policy does, and it buys two things
the dataset needs:

* a different failure surface -- the descent arrives with the jaws already at the block instead of
  settling onto it from above, so it pushes blocks over and trips the table guard in ways the reach
  expert does not;
* samples where the *act* branch was taken in a state that was not really winnable. This head
  treats a missing detection as a tracking problem rather than a reason to report
  (`unseen` weights almost nothing), so on the occluded branch it descends, closes on air, re-opens,
  and only reports once the grip has failed twice.

It is still strict about physics: a grasp waypoint the pre-filter rejects with ik / limits / torque
is unwinnable, and it reports on the first look like everyone else.
"""

from spinjev.experts.machine import PickMachine


class DirectGraspExpert(PickMachine):
    name = "direct"
    description = "descend straight to the grasp goal, no hover"
    EXCLUDE = ("hover",)
    start = "descend"    # no hover to open on: straight to the grasp goal
    #: bold: it takes a long time to get nervous enough to report
    tau = 0.80
    max_retry = 2
    stall_after = 2
    grip_after = 2
    ESCALATE_W = {
        "unseen": 0.6,           # "the detector lost it" is a tracking problem, not a report
        "object_unreachable": 5.5,
        "plan_blocked": 5.5,
        "grip_failed": 1.8,
        "stalled": 1.6,
        "drop_unreachable": 4.0,
        "tripped": 4.0,
        "unreliable": 0.6,         # like `unseen`: a bad estimate is a tracking problem here
    }
