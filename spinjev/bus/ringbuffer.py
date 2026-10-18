"""Fixed-capacity sliding window of proprioceptive frames -- the "spinal grey matter" of README's
bus section: raw high-rate signals stop here, and the cerebellum only ever sees what the
compressor makes of them.

Capacity is in *decision ticks*, not seconds: the spine runs one tick every `DECIM` physics steps
(10 x 2 ms = 20 ms -> 50 Hz), so 2 s of window is 100 frames. A fixed-size preallocated ring means
collection cost cannot grow with episode length and there is nothing to garbage collect between the
millions of ticks a data run steps through.
"""

from collections import deque

from spinjev.bus.schema import Proprio
from spinjev.spine.primitives import DECIM, PHYS_DT

WINDOW_S = 2.0  # PLAN 2.2: the cerebellum sees the last ~2 s, as statistics
TICK_DT = DECIM * PHYS_DT
WINDOW_TICKS = int(round(WINDOW_S / TICK_DT))


class RingBuffer:
    """Stores the frames it is handed, oldest first, and drops the oldest past `capacity`."""

    def __init__(self, capacity=WINDOW_TICKS, tick_dt=TICK_DT):
        self.capacity = int(capacity)
        self.tick_dt = float(tick_dt)
        self.window_s = self.capacity * self.tick_dt
        self._buf = deque(maxlen=self.capacity)
        self._dropped = 0

    def __len__(self):
        return len(self._buf)

    def __getitem__(self, i):
        return self._buf[i]

    @property
    def seconds_per_frame(self):
        return self.tick_dt

    def push(self, proprio: Proprio) -> Proprio:
        if len(self._buf) == self.capacity:
            self._dropped += 1
        self._buf.append(proprio)
        return proprio

    def clear(self):
        self._buf.clear()
        self._dropped = 0

    def full(self):
        return len(self._buf) == self.capacity

    @property
    def dropped(self):
        """Frames evicted since the last clear: proof a run really was longer than the window."""
        return self._dropped

    def latest(self):
        return self._buf[-1] if self._buf else None

    def oldest(self):
        return self._buf[0] if self._buf else None

    def frames(self):
        """Oldest -> newest. The compressor and the log read the window; nobody mutates it."""
        return list(self._buf)

    def tail(self, n):
        """The newest `n` frames, oldest first."""
        return list(self._buf)[-int(n):]

    def span(self):
        """Time actually covered, from the frame's own timestamps -- shorter than `window_s` until
        the ring has filled, and shorter still if the driver sampled it slower than 50 Hz."""
        if len(self._buf) < 2:
            return 0.0
        return float(self._buf[-1].t - self._buf[0].t)
