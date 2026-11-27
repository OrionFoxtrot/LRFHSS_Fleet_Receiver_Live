"""Advisory rr 2 source health; raw rtl_tcp cannot report true sample loss."""
from collections import deque
import math
import time


class RR2Health:
    """Confirm only persistent deficits after the client has caught up.

    The estimate is a conservative lower bound after allowing 100 ppm clock
    drift and one USB transfer plus one read block. It cannot distinguish an
    upstream pause from lost samples, nor detect gaps smaller than that allowance.
    No diagnostic changes the decoder or transport.
    """
    def __init__(self, sample_rate=1_000_000, transfer_bytes=1024*1024, *, clock=time.monotonic):
        self.fs = sample_rate
        self.clock = clock
        # Stock/fallback rtl_tcp does not expose its USB transfer size. Keep
        # the larger station allowance when that size is unknown.
        if transfer_bytes is None:
            transfer_bytes = 1024 * 1024
        self.burst_s = transfer_bytes / (2 * sample_rate) + .05
        self.bytes = 0
        self.origin = None
        self.origin_bytes = 0
        self.last_data = self.last_idle = float('-inf')
        self.lags = deque()
        self.deficits = deque()
        self.estimated_samples = 0

    def received(self, count):
        if count <= 0:
            return
        self.bytes += count
        self.last_data = self.clock()
        if self.origin is None:
            self.origin = self.last_data
            self.origin_bytes = self.bytes

    def idle(self):
        self.last_idle = self.clock()

    @staticmethod
    def _keep_window(points, now):
        # Retain the point immediately before the ten-second boundary.
        while len(points) > 1 and points[1][0] <= now - 10.:
            points.popleft()

    def snapshot(self, lag_s):
        now = self.clock()
        lag_s = max(0., float(lag_s))
        self.lags.append((now, lag_s))
        self._keep_window(self.lags, now)
        growing = (now - self.lags[0][0] >= 10. and
                   lag_s - self.lags[0][1] >= .2 and lag_s > .2)
        warning = lag_s >= .5 or growing
        caught_up = (lag_s <= .2 and now - self.last_idle <= self.burst_s + .25 and
                     now - self.last_data <= self.burst_s + .25)
        deficit = 0.
        if self.origin is not None:
            elapsed = max(0., now - self.origin)
            deficit = elapsed * self.fs - (self.bytes - self.origin_bytes) / 2
            excess = max(0., deficit - elapsed * self.fs * 100e-6 - self.burst_s * self.fs)
            if caught_up:
                self.deficits.append((now, excess))
                self._keep_window(self.deficits, now)
                if now - self.deficits[0][0] >= 10.:
                    self.estimated_samples = max(self.estimated_samples,
                        math.floor(min(value for _, value in self.deficits)))
            else:
                # Buffered IQ may still catch up; do not count its deficit as loss.
                self.deficits.clear()
        return dict(lag_s=lag_s, lag_warning=warning, lag_growing=growing,
                    estimated_source_drop_samples=self.estimated_samples,
                    source_drops_confirmed=None, sample_shortfall=max(0., deficit),
                    clock_drift_allowance_ppm=100, delivery_allowance_s=self.burst_s,
                    estimate_confirmation_s=10., caught_up=caught_up)
