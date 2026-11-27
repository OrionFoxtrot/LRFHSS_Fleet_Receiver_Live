"""Live-source health: a no-data watchdog and the ADC level of the RTL-SDR stream.

Field readiness (2026-10-28, Audit 02 B7 remainder). A stalled dongle or a hung
rtl_tcp used to look exactly like "no ChipSats transmitting". The front ends now
feed every received block to a SourceMonitor and call check() while they wait:

* stall alarm: no IQ byte for ``stall_s`` seconds (reported once per stall, and
  again when data resumes);
* level: RMS of the 8-bit I/Q components in ADC steps (LSB) and the fraction of
  components at the rails (0 or 255), measured on a cheap subsample of each block.

The level warnings are guidance for setting the manual tuner gain. The recordings
in the test-data bundle have about 0.5 LSB of noise per component: the ADC
rounding then adds noise of its own, so raise the gain until the quiet-band level
is a few LSB, but keep the clipped fraction near zero for close, strong ChipSats.
The monitor never touches the decoder and costs well under a millisecond per block.
"""
from __future__ import annotations

import time

import numpy as np

STALL_S = 3.0                 # no IQ for this long -> alarm
LOW_LEVEL_LSB = 1.5           # RMS per component below this -> gain probably too low
CLIP_FRACTION = 1e-4          # components at the rails above this -> gain too high
SUBSAMPLE = 16                # every 16th I/Q pair of a block is measured
LEVEL_WINDOW = 20             # blocks in the rolling level estimate (~2 s at 100 kB blocks)


class SourceMonitor:
    def __init__(self, stall_s=STALL_S, low_level_lsb=LOW_LEVEL_LSB, clip_fraction=CLIP_FRACTION,
                 clock=time.monotonic):
        self.stall_s, self.low_level_lsb, self.clip_limit = stall_s, low_level_lsb, clip_fraction
        self.clock = clock
        self.started = self.last_data = clock()
        self.bytes = 0
        self.stalled = False
        self.stalls = 0
        self.longest_gap_s = 0.
        self._rms = []
        self._clip = []
        self.level_state = None          # None until measured, then "ok" / "low" / "clipping"

    def note_bytes(self, raw) -> list[str]:
        """Record one received block; returns the warnings whose state changed."""
        now = self.clock()
        if not raw:
            return []
        self.longest_gap_s = max(self.longest_gap_s, now-self.last_data)
        self.last_data = now
        self.bytes += len(raw)
        out = []
        if self.stalled:
            self.stalled = False
            out.append(f"IQ data resumed after {now-self._stall_began:.1f} s without samples.")
        a = np.frombuffer(raw, dtype=np.uint8, count=len(raw)//2*2)[::2*SUBSAMPLE]
        b = np.frombuffer(raw, dtype=np.uint8, count=len(raw)//2*2)[1::2*SUBSAMPLE]
        if a.size:
            x = np.concatenate((a, b)).astype(np.float32)-127.5
            self._rms.append(float(np.sqrt(np.mean(x*x))))
            self._clip.append(float(np.mean((a == 0) | (a == 255) | (b == 0) | (b == 255))))
            del self._rms[:-LEVEL_WINDOW], self._clip[:-LEVEL_WINDOW]
            state = ("clipping" if max(self._clip) > self.clip_limit
                     else "low" if np.median(self._rms) < self.low_level_lsb else "ok")
            if state != self.level_state and len(self._rms) >= min(LEVEL_WINDOW, 5):
                if state == "low":
                    out.append(f"RTL-SDR level is {np.median(self._rms):.2f} LSB RMS: the gain is probably too low "
                               f"(ADC rounding adds noise below about {self.low_level_lsb:g} LSB).")
                elif state == "clipping":
                    out.append(f"RTL-SDR is clipping ({100*max(self._clip):.3f}% of samples at the rails): "
                               "lower the gain.")
                elif self.level_state is not None:
                    out.append(f"RTL-SDR level back in range ({np.median(self._rms):.2f} LSB RMS).")
                self.level_state = state
        return out

    def check(self) -> list[str]:
        """Call while waiting for data; returns a stall warning once per stall."""
        now = self.clock()
        if not self.stalled and now-self.last_data >= self.stall_s:
            self.stalled = True
            self.stalls += 1
            self._stall_began = self.last_data
            return [f"No IQ data from the RTL-SDR for {now-self.last_data:.1f} s: the dongle or rtl_tcp "
                    "may have stalled (check the USB connection). Decoded packets cannot appear until it resumes."]
        return []

    def snapshot(self) -> dict:
        now = self.clock()
        return dict(bytes=self.bytes, data_age_s=now-self.last_data, stalled=self.stalled,
                    stalls=self.stalls, longest_gap_s=max(self.longest_gap_s, now-self.last_data if self.stalled else 0.),
                    level_rms_lsb=float(np.median(self._rms)) if self._rms else None,
                    clip_fraction=float(max(self._clip)) if self._clip else None,
                    level_state=self.level_state)
