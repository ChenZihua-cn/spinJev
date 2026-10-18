"""Stage-1 perception: true object pose -> what a detector would report.

See PLAN 2.5. Feeding the sim's ground-truth pose straight into the observation would turn the
occlusion branch into "read a flag", so every object pose in the observation goes through here.
阶段 4 replaces this class with the real vision pipeline; the call signature is the interface.
"""

from spinjev.perception.simulator import Perceptor

__all__ = ["Perceptor"]
