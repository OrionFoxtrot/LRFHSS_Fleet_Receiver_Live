"""One-shot live tuner calibration, separate from packet reception.

The gain target is a few ADC steps of background with 6 dB of observed peak
headroom. Trigger calibration measures background energy excursions in the same
header filters as the coarse detector. It does not optimize packet sensitivity
or change the independent continuous sync-search gates.
"""
from __future__ import annotations

import math
import struct

import numpy as np
from scipy import signal

from lrfhss_live import (FS, SPB, DEFAULT_TRIGGER_DB, TRIGGER_HEADER_HZ,
                         DEFAULT_GRID, DEFAULT_BW_CODE, DEFAULT_HOPSEQ,
                         DEFAULT_HEADER_COUNT, hop_frequencies, u8_iq_to_complex)
from lrfhss_source_monitor import CLIP_FRACTION

# librtlsdr's advertised gain tables (tenths of dB). rtl_tcp's RTL0 header
# provides the tuner type and table length, but does not transmit the table.
# https://github.com/osmocom/rtl-sdr/blob/master/src/librtlsdr.c
_GAIN_TABLES = {
    1: (-10, 15, 40, 65, 90, 115, 140, 165, 190, 215, 240, 290, 340, 420),
    2: (-99, -40, 71, 179, 192),
    3: (-99, -73, -65, -63, -60, -58, -54, 58, 61, 63, 65, 67, 68, 70,
        71, 179, 181, 182, 184, 186, 188, 191, 197),
    5: (0, 9, 14, 27, 37, 77, 87, 125, 144, 157, 166, 197, 207, 229,
        254, 280, 297, 328, 338, 364, 372, 386, 402, 421, 434, 439, 445, 480, 496),
}
_GAIN_TABLES[6] = _GAIN_TABLES[5]
TARGET_RMS = 4.0
PEAK_LIMIT = 64.0             # ADC centre to rail is 127.5 steps


class AutoSetCancelled(Exception):
    pass


def supported_gains(dongle_header: bytes) -> tuple[float, ...]:
    if len(dongle_header) != 12 or dongle_header[:4] != b"RTL0":
        raise ValueError("Invalid RTL-SDR dongle header")
    tuner, count = struct.unpack("!II", dongle_header[4:])
    table = _GAIN_TABLES.get(tuner, ())
    if not table or len(table) != count:
        raise ValueError("Auto Set does not recognize this tuner's gain table; set gain manually.")
    # A saved 0 dB becomes automatic gain when passed to rtl_tcp's -g on the
    # next Start Receiver. Use a nonzero supported setting for this workflow.
    return tuple(g / 10 for g in table if g != 0)


def adc_level(raw: bytes) -> dict:
    a = np.frombuffer(raw, dtype=np.uint8)
    if len(a) < 4000 or len(a) % 2:
        raise ValueError("Not enough complete IQ samples for Auto Set")
    pairs = a.reshape(-1, 2).astype(np.float32)
    # Remove DC separately in I and Q. Otherwise DC can masquerade as noise.
    centred = pairs - pairs.mean(axis=0)
    frames = centred[:len(centred)//2000*2000].reshape(-1, 2000, 2)
    rms = np.sqrt(np.mean(frames*frames, axis=(1, 2)))
    return dict(rms_lsb=float(np.quantile(rms, .2)),
                peak_lsb=float(np.quantile(np.abs(pairs-127.5), .9999)),
                clip_fraction=float(np.mean(np.any((pairs == 0) | (pairs == 255), axis=1))))


def _safe(level):
    return level["clip_fraction"] <= CLIP_FRACTION and level["peak_lsb"] <= PEAK_LIMIT


def choose_trigger(raw: bytes, previous_db: float) -> tuple[float, str]:
    """Estimate background-only coarse energy margin, bounded to 3--8 dB.

    Use quiet 50 ms frames, excluding bursts before measuring symbol-window
    excursions. A busy/nonstationary scan keeps the user's previous margin.
    There is no claim that a short quiet scan establishes the optimum threshold.
    """
    z = u8_iq_to_complex(raw)
    frame = FS // 20
    if len(z) < FS:
        return previous_db, "Trigger kept: less than one second of background."
    hops = hop_frequencies(DEFAULT_GRID, DEFAULT_BW_CODE, DEFAULT_HOPSEQ,
                           DEFAULT_HEADER_COUNT, 0)
    margins = []
    for frequency, order, cutoff in ((TRIGGER_HEADER_HZ, 4, 3000),
                                      (hops[1], 6, 5000), (hops[2], 6, 5000)):
        n = np.arange(len(z), dtype=np.float64)
        y = signal.sosfilt(signal.butter(order, cutoff, fs=FS, output="sos"),
                           z * np.exp(-2j*np.pi*frequency*n/FS))
        # Skip filter startup. Baseline uses the same channel-power units as
        # the coarse trigger; the 99.9th percentile is an energy excursion,
        # not a predicted false-packet probability.
        p = np.abs(y[frame:])**2
        count = len(p)//frame
        p = p[:count*frame]
        powers = p.reshape(count, frame).mean(axis=1)
        quiet = powers <= 1.5*float(np.quantile(powers, .1))
        if np.count_nonzero(quiet) < max(12, math.ceil(.5*count)):
            return previous_db, "Trigger kept: too little stable background; retry during a quiet interval."
        baseline = float(np.median(powers[quiet]))
        if baseline <= 1e-14 or not math.isfinite(baseline):
            return previous_db, "Trigger kept: background level could not be measured."
        cumulative = np.r_[0., np.cumsum(p)]
        starts = np.arange(0, len(p)-SPB+1, SPB//4)
        # Keep only windows fully within a quiet frame.
        keep = quiet[starts//frame] & quiet[(starts+SPB-1)//frame]
        energy = (cumulative[starts+SPB]-cumulative[starts])/SPB
        margin = 10*math.log10(float(np.quantile(energy[keep]/baseline, .999))) + .5
        margins.append(margin)
    required = max(DEFAULT_TRIGGER_DB, *margins)
    if required > 8:
        return previous_db, "Trigger kept: background excursions are too large; inspect interference."
    # Round upwards so the displayed tenth-dB margin covers the estimate.
    return math.ceil(required*10)/10, "Trigger estimates background rejection; weak-packet sensitivity still needs live validation."


def run_autoset(gains, initial_gain, previous_trigger, *, set_gain, capture,
                cancelled=lambda: False, progress=lambda text: None):
    """Bounded scan using callbacks; set_gain must drain settling/queued IQ.

    capture(seconds) returns fresh unsigned-8 IQ. No decoder or recorder is
    running during calibration. Cancellation/failure publishes no new settings.
    """
    gains = tuple(sorted(gains))
    if not gains:
        raise ValueError("No supported manual gain settings")
    observations = []
    gain = min(gains, key=lambda g: abs(g-initial_gain))
    for attempt in range(5):
        if cancelled():
            raise AutoSetCancelled()
        progress(f"Auto Set: measuring gain {gain:g} dB ({attempt+1}/5)...")
        set_gain(gain)
        level = adc_level(capture(.25))
        observations.append(dict(gain_db=gain, **level))
        rms, peak = level["rms_lsb"], level["peak_lsb"]
        if rms < .1:
            raise ValueError("IQ has no measurable variation; check the source before Auto Set.")
        change = min(20*math.log10(TARGET_RMS/rms), 20*math.log10(PEAK_LIMIT/max(peak, .5)))
        if level["clip_fraction"] > CLIP_FRACTION:
            change = min(change, -8.)
        target = gain + change
        next_gain = min(gains, key=lambda g: abs(g-target))
        if change < 0:
            next_gain = max((g for g in gains if g <= target), default=gains[0])
        if next_gain in [row["gain_db"] for row in observations]:
            break
        gain = next_gain
    safe = [row for row in observations if _safe(row)]
    if not safe:
        raise ValueError("No gain with sufficient ADC headroom was found; lower gain or reduce the RF input.")
    best = min(safe, key=lambda row: abs(math.log(max(row["rms_lsb"], .1)/TARGET_RMS)))
    gain = best["gain_db"]
    if cancelled():
        raise AutoSetCancelled()
    progress(f"Auto Set: checking background at {gain:g} dB and measuring trigger...")
    set_gain(gain)
    raw = capture(2.)
    final_level = adc_level(raw)
    if not _safe(final_level):
        raise ValueError("Signal levels changed during Auto Set; settings kept. Retry with representative traffic.")
    trigger, note = choose_trigger(raw, previous_trigger)
    if cancelled():
        raise AutoSetCancelled()
    if final_level["rms_lsb"] < 1.5:
        note += " ADC level is still low at the available gain."
    summary = (f"Auto Set: gain {gain:g} dB, trigger {trigger:g} dB "
               f"({final_level['rms_lsb']:.1f} LSB background). {note}")
    return dict(gain_db=gain, trigger_db=trigger, summary=summary,
                note=note, level=final_level, observations=observations)
