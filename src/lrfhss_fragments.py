"""Payload quality observations, independent of Qt and application telemetry.

These are measured signal/noise quantities and scheduling evidence, not bit
error probabilities. Cochannel interference can look like a strong signal.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
import math
from contextlib import contextmanager
from contextvars import ContextVar
import numpy as np


@dataclass
class FragmentResult:
    index: int
    expected_frequency_hz: float
    expected_start_sample: int
    bit_count: int
    timing_offset_samples: int | None = None
    raw_soft_bits: np.ndarray = field(default_factory=lambda: np.empty(0))
    soft_bits: np.ndarray = field(default_factory=lambda: np.empty(0))
    timing_score: float | None = None
    mean_abs_soft: float | None = None
    saturated_soft_fraction: float | None = None
    invalid_phase_fraction: float | None = None
    inband_power: float | None = None
    estimated_noise_power: float | None = None
    excess_power_fraction: float | None = None
    predicted_cochannel_fraction: float = 0.
    quality: str = "unavailable"
    valid: bool = False  # compatibility: a usable extracted fragment
    usable: bool = False
    spectral_quality_measured: bool = False
    rejection_reason: str | None = None

    def diagnostic(self):
        """JSON-safe scalar observations; large soft arrays stay out of GUI logs."""
        return {k: v for k, v in vars(self).items() if k not in ("raw_soft_bits", "soft_bits")}


def spectral_power(raw, frequency_hz, fs):
    """Hann periodogram: integrate +/-350 Hz; estimate noise at 800..2500 Hz.

    Median sideband bin power / ln(2) estimates the mean of exponential noise
    power. This assumes locally white complex noise; adjacent interference or
    colored noise can bias it. No desired/interferer separation is implied.
    Power units are normalized input IQ amplitude squared, not calibrated W.
    """
    n = len(raw)
    if n < 32:
        return 0., 0.
    window = np.hanning(n)
    nfft = 1 << (n-1).bit_length()
    psd = abs(np.fft.fft(raw*window, n=nfft))**2 / (nfft*np.sum(window**2))
    freq = np.fft.fftfreq(nfft, 1/fs)
    delta = abs((freq-frequency_hz+fs/2) % fs-fs/2)
    main = delta <= 350.
    side = (delta >= 800.) & (delta <= 2500.)
    power = float(np.sum(psd[main]))
    noise = float(np.median(psd[side])/math.log(2)*np.count_nonzero(main))
    return power, noise


_pending_quality = ContextVar('lrfhss_pending_quality', default=None)

class DeferredQuality:
    """Call-scoped immutable raw views; never transported in worker results."""
    def __init__(self):
        self.pending = []
        self.computed = 0

    def defer(self, result, raw, fs):
        # Worker input snapshots are immutable. A public synchronous caller must
        # likewise not mutate its input while decode_payload is executing.
        view = raw.view()
        view.flags.writeable = False
        self.pending.append((result, view, fs))

    def measure_all(self):
        try:
            for result, raw, fs in self.pending:
                measure_quality(result, raw, fs)
                self.computed += 1
        finally:
            self.pending.clear()

@contextmanager
def deferred_quality():
    lease = DeferredQuality()
    token = _pending_quality.set(lease)
    try:
        yield lease
    finally:
        lease.pending.clear()
        _pending_quality.reset(token)

def measure_quality(result, raw, fs):
    """The existing periodogram, weight and classification, without phase work."""
    p, n = spectral_power(raw, result.expected_frequency_hz, fs)
    weight = float(np.clip(1-n/p, 0, 1)) if p > 0 else 0.
    result.inband_power, result.estimated_noise_power = p, n
    result.excess_power_fraction = weight
    result.soft_bits = result.raw_soft_bits * weight
    result.quality = ('noise_dominated' if weight <= .5 else
                      'weak_excess' if weight <= .75 else
                      'cochannel_possible' if result.predicted_cochannel_fraction else
                      'signal_present')
    result.spectral_quality_measured = True


def observe_quality(result, raw, phase, centers, fs, look, overlaps=()):
    """Attach diagnostics and a noise-excess shrinkage weight to one fragment.

    The weight max(0, (P-N)/P) is an estimated excess-power fraction. Zero gives
    equal Viterbi branch costs (an erasure); it is not a confidence percentage.
    Known cochannel occupancy is recorded but never automatically erased.
    """
    result.mean_abs_soft = float(np.mean(abs(result.raw_soft_bits)))
    result.saturated_soft_fraction = float(np.mean(abs(result.raw_soft_bits) >= 1.))
    delta = phase[centers+look]-phase[centers-look]
    delta = np.where(abs(delta) > np.pi/2, delta-np.copysign(2*np.pi, delta), delta)
    result.invalid_phase_fraction = float(np.mean(abs(delta[1:]) > np.pi/2))
    # Union of known other-transmitter scheduled intervals sharing our 700 Hz
    # analysis band. This is geometrical occupancy, independent of IQ truth.
    start = result.expected_start_sample
    stop = start+(result.bit_count+1)*4*look
    intervals = sorted((max(start, lo), min(stop, hi)) for lo, hi, freq in overlaps
                       if abs(freq-result.expected_frequency_hz) < 700 and hi > start and lo < stop)
    occupied = 0
    end = start
    for lo, hi in intervals:
        occupied += max(0, hi-max(lo, end))
        end = max(end, hi)
    result.predicted_cochannel_fraction = occupied/max(1, stop-start)
    result.valid = result.usable = True
    pending = _pending_quality.get()
    if pending is None:
        measure_quality(result, raw, fs)
    else:
        result.quality = 'not_measured_raw_first'
        pending.defer(result, raw, fs)



def fragment_summary(fragments):
    counts = dict(Counter(f.quality for f in fragments))
    weights = [f.excess_power_fraction for f in fragments if f.excess_power_fraction is not None]
    return {"quality_counts": counts, "fragments_obtained": sum(f.valid for f in fragments),
        "usable_fragments": sum(f.usable for f in fragments),
        "spectral_quality_measured": all(f.spectral_quality_measured for f in fragments),
        "spectral_fragments_measured": sum(f.spectral_quality_measured for f in fragments),
        "spectral_fragments_unmeasured": sum(f.usable and not f.spectral_quality_measured for f in fragments),
        "strongly_degraded": counts.get("noise_dominated", 0)+counts.get("unavailable", 0),
        "predicted_cochannel_fragments": sum(f.predicted_cochannel_fraction > 0 for f in fragments),
        "minimum_excess_power_fraction": min(weights) if weights else None,
        "note": "Excess-power fraction is not a probability; strong cochannel errors can be unflagged"}
