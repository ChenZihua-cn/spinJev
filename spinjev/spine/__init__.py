"""The spine: primitive controllers, hard guards, candidate pre-filtering.

It is the D/A converter of the architecture -- it turns a discrete `Choice` plus a goal into
continuous joint setpoints, and refuses commands the physics will not honour. It knows nothing
about which backend it is talking to beyond the `kin`/`act`/`sense` surface.
"""

from spinjev.spine.primitives import DECIM, Spine, Primitive

__all__ = ["DECIM", "Spine", "Primitive"]
