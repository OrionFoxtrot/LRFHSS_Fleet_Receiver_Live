#!/usr/bin/env python3
"""
LR-FHSS live/offline decoder specialized for the RadioLib STM32WL profile used
by the DeSCENT/ChipSat test setup.

Profile:
  RF center      915 MHz
  LR-FHSS BW     722.66 kHz (code 0x06)
  grid           narrow / 3.90625 kHz (grid=1)
  coding rate    2/3 (CR=1)
  headers        3
  symbol rate    488.28125 symbols/s
  sync word      0x12AD101B
  hop sequence   normally 0x100 (decoded from header)

Input:
  rtl_sdr.exe native unsigned-8-bit interleaved IQ, 1.000 MS/s.

This is a compact bench/research receiver. It deliberately omits the original
MATLAB receiver's simulation, SIC/collision cancellation and multi-profile code.
See THIRD_PARTY_NOTICES.md for source/reference attribution.
"""
from __future__ import annotations
import argparse, collections, functools, math, os, shutil, socket, subprocess, sys, time
from dataclasses import dataclass, field
import numpy as np
from scipy import signal
from lrfhss_readable import print_payload
from lrfhss_tracking import PacketTrack, TrackManager
from lrfhss_acquisition import HeaderAcquisition, HeaderObservation, sync_score
from lrfhss_metrics import measure, phase_cycles
from lrfhss_fragments import FragmentResult, observe_quality, fragment_summary, deferred_quality

FS = 1_000_000
BITRATE = 488.28125
SPB = int(round(FS / BITRATE))         # 2048
LOOK = SPB // 4                        # 512
PHASE_SLOPE = (math.pi / 2.0) / SPB
STAY_HDR = int(round(0.233472 * FS))   # 233472 samples
STAY_DATA = int(round(0.1024 * FS))    # 102400 samples
PLL_STEP_HZ = 0.95367431640625

DEFAULT_CENTER_HZ = 915_000_000
DEFAULT_GAIN_DB = 40.0
DEFAULT_TRIGGER_DB = 3.0
DEFAULT_CARRIER_SEARCH_HZ = 4500.0
DEFAULT_HOPSEQ = 0x100
DEFAULT_BW_CODE = 0x06
DEFAULT_GRID = 1
DEFAULT_CR = 1
DEFAULT_HEADER_COUNT = 3

# First header hop for the known RadioLib default profile (hop sequence 0x100).
TRIGGER_HEADER_HZ = 144_531.25

# Semtech LR-FHSS convolutional-code output tables.
HEADER_OUT = np.array([
    [0,3],[1,2],[2,1],[3,0],[2,1],[3,0],[0,3],[1,2],
    [3,0],[2,1],[1,2],[0,3],[1,2],[0,3],[3,0],[2,1]
], dtype=np.int8)

PAY_OUT = np.array([
    [0,7],[3,4],[7,0],[4,3],[6,1],[5,2],[1,6],[2,5],
    [1,6],[2,5],[6,1],[5,2],[7,0],[4,3],[0,7],[3,4],
    [4,3],[7,0],[3,4],[0,7],[2,5],[1,6],[5,2],[6,1],
    [5,2],[6,1],[2,5],[1,6],[3,4],[0,7],[4,3],[7,0],
    [7,0],[4,3],[0,7],[3,4],[1,6],[2,5],[6,1],[5,2],
    [6,1],[5,2],[1,6],[2,5],[0,7],[3,4],[7,0],[4,3],
    [3,4],[0,7],[4,3],[7,0],[5,2],[6,1],[2,5],[1,6],
    [2,5],[1,6],[5,2],[6,1],[4,3],[7,0],[3,4],[0,7]
], dtype=np.int8)

# Semtech header interleaver, zero based.
HEADER_INTERLEAVER = np.array([
    0,18,36,54,72,4,22,40,58,76,8,26,44,62,12,30,
    48,66,16,34,52,70,1,19,37,55,73,5,23,41,59,77,
    9,27,45,63,13,31,49,67,17,35,53,71,2,20,38,56,
    74,6,24,42,60,78,10,28,46,64,14,32,50,68,3,21,
    39,57,75,7,25,43,61,79,11,29,47,65,15,33,51,69
], dtype=np.int16)
HEADER_DEINTERLEAVER = np.argsort(HEADER_INTERLEAVER)

CHANNEL_COUNTS = [80,176,280,376,688,792,1480,1584,3120,3224]
POLY1 = [33,45,48,51,54,57]
POLY2 = [65,68,71,72]
POLY3 = [142,149]

def _bits(v: int, n: int) -> np.ndarray:
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)

HEADER_OUT_BITS = np.array([[_bits(int(v),2) for v in row] for row in HEADER_OUT])
PAY_OUT_BITS = np.array([[_bits(int(v),3) for v in row] for row in PAY_OUT])
_HDR_P0 = np.arange(16) // 2
_HDR_P1 = _HDR_P0 + 8
_HDR_BIT = np.arange(16) & 1

def bits_to_int(b) -> int:
    v = 0
    for x in b:
        v = (v << 1) | int(x)
    return v

def bits_to_bytes(b) -> bytes:
    b = np.asarray(b, dtype=np.uint8)
    return bytes(bits_to_int(b[i:i+8]) for i in range(0, len(b), 8))

def crc_msb(data: bytes, poly: int, width: int, init: int) -> int:
    crc = init
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    for byte in data:
        crc ^= int(byte) << (width - 8)
        for _ in range(8):
            crc = (((crc << 1) ^ poly) & mask) if (crc & top) else ((crc << 1) & mask)
    return crc

def crc8_bits(b32) -> np.ndarray:
    return _bits(crc_msb(bits_to_bytes(b32), 0x2F, 8, 0xFF), 8)

def crc16_bits(b) -> np.ndarray:
    return _bits(crc_msb(bits_to_bytes(b), 0x755B, 16, 0xFFFF), 16)

def u8_iq_to_complex(raw: bytes | np.ndarray) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray, memoryview)) else np.asarray(raw, dtype=np.uint8)
    if len(a) & 1:
        a = a[:-1]
    i = (a[0::2].astype(np.float32) - 127.5) / 127.5
    q = (a[1::2].astype(np.float32) - 127.5) / 127.5
    return i + 1j*q

_U8_PAIR_LUT = None


def u8_iq_to_complex_fast(raw: bytes | np.ndarray) -> np.ndarray:
    """Bit-identical to u8_iq_to_complex for EVERY input, about 2-3x faster on large spans.

    Batch 9.10 payload-worker conversion. The 65,536-entry complex64 table is
    built by calling u8_iq_to_complex itself on every possible (I, Q) byte pair,
    so each output sample is exactly the reference value for its two input
    bytes. The little-endian u16 view makes the index I + 256*Q on every host.
    """
    global _U8_PAIR_LUT
    if _U8_PAIR_LUT is None:
        table = u8_iq_to_complex(np.arange(65536, dtype="<u2").view(np.uint8))
        table.flags.writeable = False
        _U8_PAIR_LUT = table
    a = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray, memoryview)) else np.ascontiguousarray(raw, dtype=np.uint8)
    if len(a) & 1:
        a = a[:-1]
    return _U8_PAIR_LUT[a.view("<u2")]

def mix_lowpass(z: np.ndarray, freq_hz: float, cutoff_hz: float) -> np.ndarray:
    n = np.arange(len(z), dtype=np.float64)
    bb = z * np.exp(-1j * 2*np.pi * freq_hz * n / FS)
    sos = signal.butter(6, cutoff_hz, btype="low", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, bb)

def demod_soft(bb: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return _soft_from_phase(np.angle(bb), centers)


def _soft_from_phase(ph: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # Identical phase-slope math; header timing trials share one angle array.
    vals = np.zeros(len(centers), dtype=np.float64)
    for k, c in enumerate(centers.astype(int)):
        d = ph[c + LOOK] - ph[c - LOOK]
        if abs(d) > math.pi/2:
            d -= math.copysign(2*math.pi, d)
        if abs(d) > math.pi/2:
            d = 0.0
        vals[k] = (d / (2*LOOK)) / PHASE_SLOPE

    # Match the simple frequency-drift bias correction used by the MATLAB receiver.
    p80 = np.percentile(np.abs(vals), 80)
    use = np.abs(vals) < p80
    if np.any(use):
        denom = int(np.sum(use))
        adj = 0.0
        for neg in (True, False):
            m = use & ((vals < 0) if neg else (vals >= 0))
            if np.any(m):
                mean = float(np.mean(vals[m]))
                dev = mean + 1.0 if neg else mean - 1.0
                adj += (np.sum(m) / denom) * dev
        vals -= adj
    return np.clip(vals, -1.0, 1.0)

def deinterleave_header(soft113: np.ndarray) -> np.ndarray:
    # Physical header: guard + 40 encoded + 32 sync + 40 encoded (+ final guard).
    coded80 = np.concatenate([soft113[1:41], soft113[73:113]])
    return coded80[HEADER_DEINTERLEAVER]

def viterbi_header(soft80: np.ndarray):
    """The original exhaustive header trellis, vectorized over start/end states.

    All 16 initial states and 16 endpoints, strict-less-than tie breaking and
    CRC selection are preserved. Payload Viterbi is deliberately unchanged.
    """
    obs = np.asarray(soft80, dtype=float).reshape(40, 2)
    metric = np.full((16, 16), np.inf)
    np.fill_diagonal(metric, 0.0)
    prev = np.empty((40, 16, 16), dtype=np.int8)
    e0 = 2*HEADER_OUT_BITS[_HDR_P0, _HDR_BIT].astype(float) - 1
    e1 = 2*HEADER_OUT_BITS[_HDR_P1, _HDR_BIT].astype(float) - 1
    for t, r in enumerate(obs):
        c0 = metric[:, _HDR_P0] + np.sum((e0-r)**2, axis=1)
        c1 = metric[:, _HDR_P1] + np.sum((e1-r)**2, axis=1)
        choose1 = c1 < c0
        metric = np.where(choose1, c1, c0)
        prev[t] = np.where(choose1, _HDR_P1, _HDR_P0)
    rows = np.arange(16)[:, None]
    states = np.broadcast_to(np.arange(16), (16, 16)).copy()
    bits = np.empty((16, 16, 40), dtype=np.uint8)
    for t in range(39, -1, -1):
        bits[:, :, t] = states & 1
        states = prev[t, rows, states]
    crc = np.full((16, 16), 0xff, dtype=np.uint8)
    for t in range(32):
        top = (crc >> 7) ^ bits[:, :, t]
        crc = (crc << 1) ^ (top * 0x2f)
    received = np.zeros((16, 16), dtype=np.uint8)
    for t in range(32, 40):
        received = (received << 1) | bits[:, :, t]
    valid = crc == received
    if not np.any(valid):
        return None
    costs = np.where(valid, metric, np.inf)
    init, end = np.unravel_index(np.argmin(costs), costs.shape)
    return float(metric[init, end]), bits[init, end].copy()

def parse_header(b: np.ndarray) -> dict:
    return {
        "payload_len": bits_to_int(b[0:8]),
        "modulation": bits_to_int(b[8:11]),
        "cr": bits_to_int(b[11:13]),
        "grid": int(b[13]),
        "hop": int(b[14]),
        "bw_code": bits_to_int(b[15:19]),
        "hop_seq_id": bits_to_int(b[19:28]),
        "sync_index": bits_to_int(b[28:30]),
    }

def estimate_frequency(z: np.ndarray, start: int, duration_s: float, nominal: float, span_hz: float) -> float:
    s = max(0, int(start))
    e = min(len(z), s + int(duration_s*FS))
    x = z[s:e]
    if len(x) < 1000:
        return nominal
    nfft = 1 << int(math.ceil(math.log2(len(x))))
    X = np.fft.fftshift(np.fft.fft(x*np.hanning(len(x)), n=nfft))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1/FS))
    m = (f >= nominal-span_hz) & (f <= nominal+span_hz)
    return float(f[m][np.argmax(np.abs(X[m])**2)])

@dataclass
class Header:
    start: int
    freq_hz: float
    cost: float
    info: dict

def decode_first_header(z: np.ndarray, onset: int, diagnostics=None) -> Header | None:
    """Decode the first physical header near an energy-onset estimate."""
    freq = estimate_frequency(z, onset + int(0.01*FS), 0.20, TRIGGER_HEADER_HZ, 5000)
    detail = {"frequency_hz": freq, "cfo_hz": freq-TRIGGER_HEADER_HZ,
              "header_crc_ok": False, "viterbi_metric": None,
              "sync_correlation": None, "reason": "header_crc"}
    lo = max(0, onset - int(0.03*FS))
    hi = min(len(z), onset + int(0.29*FS))
    if hi-lo < int(0.25*FS):
        if diagnostics is not None:
            diagnostics.update(detail, reason="header_iq_incomplete")
        return None
    bb = mix_lowpass(z[lo:hi], freq, 250.0)
    phase = np.angle(bb)

    # The trigger is based on a 2 ms power average; on the validated RTL-SDR
    # capture the first symbol center is ~4.25 symbols after that onset.
    expected = onset + int(round(4.25*SPB))
    best = None
    best_score = None
    for st in range(expected-1024, expected+1025, 64):
        centers = (st-lo) + np.arange(113)*SPB
        if centers[0]-LOOK < 0 or centers[-1]+LOOK >= len(bb):
            continue
        soft = _soft_from_phase(phase, centers)
        dec = viterbi_header(deinterleave_header(soft))
        if dec is None:
            continue
        cost, bits = dec
        info = parse_header(bits)
        if detail["viterbi_metric"] is None or cost < detail["viterbi_metric"]:
            detail.update(header_crc_ok=True, viterbi_metric=cost, header_fields=info,
                          sync_correlation=sync_score(soft[41:73]), reason="header_profile")
        if not (info["cr"] == DEFAULT_CR and info["grid"] == DEFAULT_GRID and
                info["hop"] == 1 and info["bw_code"] == DEFAULT_BW_CODE):
            continue
        if info["modulation"] != 0 or info["sync_index"] != 2 or info["payload_len"] < 1:
            continue
        if info["hop_seq_id"] != DEFAULT_HOPSEQ:
            continue
        if abs(hop_frequencies(info["grid"], info["bw_code"], info["hop_seq_id"], 3, 0)[0]
               - TRIGGER_HEADER_HZ) >= 1:
            continue
        # Keep the established energy + CRC/profile fallback for weak isolated
        # packets. New sync thresholds do not veto this successful legacy path.
        if best is None or cost < best.cost:
            best = Header(start=st, freq_hz=freq, cost=cost, info=info)
            best_score = sync_score(soft[41:73])
    if diagnostics is not None:
        if best is not None:
            detail.update(header_crc_ok=True, viterbi_metric=best.cost,
                          sync_correlation=best_score, header_fields=best.info, reason="validated")
        diagnostics.update(detail)
    return best


def decode_physical_header(z, base, candidate, refinement_span=None, counters=None, batch_header=False, header_grids=None, frequency_selection=None, frequency_reuse=False,
                           validation_reuse=False, previous_validation=None):
    """Preserve original refinement by default; explicit local hypotheses are optional.

    The caller schedules a fallback only after ordinary PHDR opportunities.
    Every hypothesis uses the unchanged CRC/profile/sync/quality validator.
    """
    anchor = candidate.header_start-base
    if candidate.source == "energy":
        nominal, span = candidate.nominal_hz, 5000
    else:
        nominal, span = candidate.frequency_hz, 250
    if refinement_span is not None:
        nominal, span = candidate.frequency_hz, refinement_span
    with measure(counters if counters is not None else collections.Counter(), "cfo_refinement"):
        estimate = estimate_frequency if header_grids is None else header_grids.estimate
        selection = None
        if frequency_reuse and candidate.source == "sync" and span in (0,100,250):
            from lrfhss_frequency_reuse import select
            selection, hit = select(z, base, candidate, FS, header_grids if header_grids is not None else False, frequency_selection)
            if counters is not None:
                counters["frequency_reuse_hits" if hit else "frequency_reuse_misses"] += 1
                counters["frequency_reuse_invalidations"] += int(frequency_selection is not None and not hit)
            freq = dict(selection.frequencies)[float(span)]
        else:
            freq = (estimate(z, anchor+2000, .20, nominal, span)
                    if span else nominal)
    with measure(counters if counters is not None else collections.Counter(), "physical_header_validation"):
        validator = _validate_physical_frequency_batched if batch_header else _validate_physical_frequency
        if validation_reuse and selection is not None:
            from lrfhss_header_reuse import validate
            header, detail, previous, hit = validate(validator, z, base, candidate, freq,
                selection.identity, batch_header, previous_validation)
            if counters is not None:
                counters['validation_reuse_hits' if hit else 'validation_reuse_misses'] += 1
            detail['previous_validation'] = previous
        else:
            header, detail = validator(z, base, candidate, freq)
        if selection is not None:
            detail["frequency_selection"] = selection
        return header, detail


def _validate_physical_frequency(z, base, candidate, freq):
    """Batch 5 physical-header validation math, unchanged after frequency selection."""
    anchor = candidate.header_start-base
    bb = mix_lowpass(z, freq, 250.0)
    phase = np.angle(bb)
    best = None
    diagnostic = {"header_crc_ok": None, "viterbi_metric": None,
                  "sync_correlation": candidate.sync_correlation,
                  "frequency_hz": freq, "cfo_hz": freq-candidate.nominal_hz,
                  "reason": "header_correlation"}
    rejection_rank = -1
    # Seven coarse timing trials reject spurious correlation peaks cheaply.
    # A plausible header unlocks the complete original-resolution refinement.
    coarse = [0, -512, 512, -1024, 1024, -1536, 1536]
    trials = coarse.copy()
    refined = False
    for delta in trials:
        st = anchor+delta
        centers = st+np.arange(113)*SPB
        if centers[0]-LOOK < 0 or centers[-1]+LOOK >= len(bb):
            continue
        soft = _soft_from_phase(phase, centers)
        score = sync_score(soft[41:73])
        if score < .45:
            continue
        dec = viterbi_header(deinterleave_header(soft))
        if dec is None:
            if rejection_rank < 0:
                diagnostic.update(header_crc_ok=False, reason="header_crc", sync_correlation=score)
            continue
        cost, bits = dec
        info = parse_header(bits)
        rank, reason = 0, "header_profile"
        valid_profile = (info["modulation"] == 0 and info["cr"] == DEFAULT_CR
                         and info["grid"] == DEFAULT_GRID and info["hop"] == 1
                         and info["bw_code"] == DEFAULT_BW_CODE and info["payload_len"] >= 1)
        allowed_id = info["hop_seq_id"] in getattr(candidate, "allowed_hop_ids", (DEFAULT_HOPSEQ,))
        if valid_profile and not allowed_id:
            reason = "hop_id_not_allowed"
        if valid_profile and allowed_id:
            rank, reason = 1, "physical_header_index"
            if info["sync_index"] == DEFAULT_HEADER_COUNT-1-candidate.physical_index:
                rank, reason = 2, "header_frequency_plan"
                nominal_hops = hop_frequencies(info["grid"], info["bw_code"], info["hop_seq_id"], 3, 0)
                if abs(nominal_hops[candidate.physical_index]-candidate.nominal_hz) < 1:
                    rank, reason = 3, "header_quality"
                    if not refined:
                        trials.extend(d for d in range(-1536, 1537, 64) if d not in coarse)
                        refined = True
                    if score >= .45 and cost <= 44 and abs(freq-candidate.nominal_hz) <= getattr(candidate, "cfo_limit_hz", 5000.0):
                        if best is None or cost < best[0].cost:
                            best = (Header(st+base, freq, cost, info), score)
        if rank > rejection_rank or (rank == rejection_rank and
                                     cost < diagnostic.get("viterbi_metric", float("inf"))):
            rejection_rank = rank
            diagnostic.update(header_crc_ok=True, viterbi_metric=cost,
                              sync_correlation=score, header_fields=info,
                              reason=reason, refined_header_start=st+base)
    if best is None:
        return None, diagnostic
    header, score = best
    diagnostic.update(header_crc_ok=True, viterbi_metric=header.cost,
                      sync_correlation=score, header_fields=header.info,
                      reason="validated", refined_header_start=header.start)
    return header, diagnostic

def _validate_physical_frequency_batched(z, base, candidate, freq):
    """Batch 5 physical-header validation math, unchanged after frequency selection."""
    anchor = candidate.header_start-base
    bb = mix_lowpass(z, freq, 250.0)
    phase = _SampledPhase(bb)
    best = None
    diagnostic = {"header_crc_ok": None, "viterbi_metric": None,
                  "sync_correlation": candidate.sync_correlation,
                  "frequency_hz": freq, "cfo_hz": freq-candidate.nominal_hz,
                  "reason": "header_correlation"}
    rejection_rank = -1
    # Seven coarse timing trials reject spurious correlation peaks cheaply.
    # A plausible header unlocks the complete original-resolution refinement.
    coarse = [0, -512, 512, -1024, 1024, -1536, 1536]
    trials = coarse.copy()
    refined = False
    from lrfhss_header_batch import viterbi_headers
    import sys
    decoded = {}
    timing_soft = {}
    def prepare(deltas):
        softs=[];keys=[]
        delta_array=np.asarray(deltas,dtype=int)
        centers=anchor+delta_array[:,None]+np.arange(113)*SPB
        valid=(centers[:,0]-LOOK>=0)&(centers[:,-1]+LOOK<len(bb))
        trial_soft=_soft_trials(phase,centers[valid]) if np.any(valid) else []
        for delta,soft in zip(delta_array[valid],trial_soft):
            score=sync_score(soft[41:73])
            timing_soft[delta]=(soft,score)
            if score<.45:continue
            keys.append(delta);softs.append(deinterleave_header(soft))
        for key,value in zip(keys,viterbi_headers(softs,sys.modules[__name__])):
            decoded[key]=value
    prepare(coarse)
    for delta in trials:
        st = anchor+delta
        centers = st+np.arange(113)*SPB
        if centers[0]-LOOK < 0 or centers[-1]+LOOK >= len(bb):
            continue
        soft, score = timing_soft[delta]
        if score < .45:
            continue
        dec = decoded.get(delta)
        if dec is None:
            if rejection_rank < 0:
                diagnostic.update(header_crc_ok=False, reason="header_crc", sync_correlation=score)
            continue
        cost, bits = dec
        info = parse_header(bits)
        rank, reason = 0, "header_profile"
        valid_profile = (info["modulation"] == 0 and info["cr"] == DEFAULT_CR
                         and info["grid"] == DEFAULT_GRID and info["hop"] == 1
                         and info["bw_code"] == DEFAULT_BW_CODE and info["payload_len"] >= 1)
        allowed_id = info["hop_seq_id"] in getattr(candidate, "allowed_hop_ids", (DEFAULT_HOPSEQ,))
        if valid_profile and not allowed_id:
            reason = "hop_id_not_allowed"
        if valid_profile and allowed_id:
            rank, reason = 1, "physical_header_index"
            if info["sync_index"] == DEFAULT_HEADER_COUNT-1-candidate.physical_index:
                rank, reason = 2, "header_frequency_plan"
                nominal_hops = hop_frequencies(info["grid"], info["bw_code"], info["hop_seq_id"], 3, 0)
                if abs(nominal_hops[candidate.physical_index]-candidate.nominal_hz) < 1:
                    rank, reason = 3, "header_quality"
                    if not refined:
                        extra=[d for d in range(-1536,1537,64) if d not in coarse]
                        prepare(extra)
                        trials.extend(extra)
                        refined = True
                    if score >= .45 and cost <= 44 and abs(freq-candidate.nominal_hz) <= getattr(candidate, "cfo_limit_hz", 5000.0):
                        if best is None or cost < best[0].cost:
                            best = (Header(st+base, freq, cost, info), score)
        if rank > rejection_rank or (rank == rejection_rank and
                                     cost < diagnostic.get("viterbi_metric", float("inf"))):
            rejection_rank = rank
            diagnostic.update(header_crc_ok=True, viterbi_metric=cost,
                              sync_correlation=score, header_fields=info,
                              reason=reason, refined_header_start=st+base)
    if best is None:
        return None, diagnostic
    header, score = best
    diagnostic.update(header_crc_ok=True, viterbi_metric=header.cost,
                      sync_correlation=score, header_fields=header.info,
                      reason="validated", refined_header_start=header.start)
    return header, diagnostic

def _lfsr_next(state: int, n_grid: int, poly: int, xseed: int):
    while True:
        lsb = state & 1
        state >>= 1
        if lsb:
            state ^= poly
        hop = xseed if xseed == state else (xseed ^ state)
        if hop <= n_grid:
            return hop-1, state

def hop_frequencies(grid: int, bw_code: int, hopseq: int, header_count: int, num_frag: int) -> np.ndarray:
    n_grid = CHANNEL_COUNTS[bw_code] // (8 if grid == 1 else 52)
    if n_grid in (10,22,28,30,35,47):
        state, poly, xseed = 6, POLY1[(hopseq >> 6) & 7], hopseq & 0x3F
    elif n_grid in (60,62):
        state, poly, xseed = 56, POLY1[(hopseq >> 6) & 7], hopseq & 0x3F
    elif n_grid in (86,99):
        state, poly, xseed = 6, POLY2[(hopseq >> 7) & 3], hopseq & 0x7F
    elif n_grid in (185,198):
        state, poly, xseed = 6, POLY3[(hopseq >> 8) & 1], hopseq & 0xFF
    elif n_grid in (390,403):
        state, poly, xseed = 6, 264, hopseq & 0x1FF
    else:
        raise ValueError(f"Unsupported LR-FHSS n_grid={n_grid}")

    for _ in range(4-header_count):
        _, state = _lfsr_next(state, n_grid, poly, xseed)

    out = []
    for current_hop in range(header_count + num_frag):
        ni, state = _lfsr_next(state, n_grid, poly, xseed)
        if ni >= n_grid//2:
            ni -= n_grid
        channels_per_grid = 8 if grid == 1 else 52
        grid_offset = (1 + (n_grid & 1)) * (channels_per_grid//2)
        grid_steps = 4096 if grid == 1 else 26624
        steps = -ni*grid_steps - grid_offset*512
        if current_hop < header_count and ((header_count-current_hop) & 1) == 0:
            steps += 256
        out.append(steps * PLL_STEP_HZ)
    return np.asarray(out)

def payload_layout(payload_len: int):
    # decoded bits = payload bytes + 16-bit payload CRC + 6 terminating bits
    decoded_with_crc_tail = 8*(payload_len+2) + 6
    coded = int(math.ceil(decoded_with_crc_tail*1.5))  # CR 2/3 puncturing
    nfrag = int(math.ceil(coded/48))
    frag_bits = [48]*nfrag
    frag_bits[-1] = coded - 48*(nfrag-1)
    return coded, frag_bits

def deinterleave_payload(payload: np.ndarray, nbits: int) -> np.ndarray:
    payload = np.asarray(payload, dtype=float)
    y = 0
    while y*y < nbits:
        y += 1
    step = y
    step_v = step >> 1
    step <<= 1
    st_idx = st_idx_init = pos = 0

    # Keep index 0 unused here to reproduce the reference algorithm's
    # 1-based indexing without ambiguity.
    de = np.zeros(len(payload)+1, dtype=float)
    for i in range(2, len(payload)+1):
        pos += step
        if pos >= nbits:
            st_idx += step_v
            if st_idx >= step:
                st_idx_init += 1
                st_idx = st_idx_init
            pos = st_idx
        de[pos] = payload[i-1]
    de[2:] = de[1:-1].copy()
    de[1] = payload[0]
    return de[1:]

def viterbi_payload_cr23(soft: np.ndarray):
    vals = np.asarray(soft, dtype=float)
    ninfo = int(math.floor(len(vals)/1.5))
    metric = np.full(64, np.inf)
    metric[0] = 0.0
    prev_state = np.full((ninfo,64), -1, dtype=np.int16)
    prev_bit = np.zeros((ninfo,64), dtype=np.int8)
    cursor = 0

    for i in range(1, ninfo+1):
        if i & 1:
            r = vals[cursor:cursor+2]; cursor += 2
            obsidx = [0,1]
        else:
            r = vals[cursor:cursor+1]; cursor += 1
            obsidx = [1]
        new = np.full(64, np.inf)
        for st in range(64):
            if not np.isfinite(metric[st]):
                continue
            for bit in (0,1):
                ns = 2*(st % 32) + bit
                expected = 2*PAY_OUT_BITS[st, bit, obsidx].astype(float)-1
                m = metric[st] + float(np.sum((expected-r)**2))
                if m < new[ns]:
                    new[ns] = m
                    prev_state[i-1, ns] = st
                    prev_bit[i-1, ns] = bit
        metric = new

    # Terminated convolutional code: final state is zero.
    st = 0
    if not np.isfinite(metric[st]):
        return None
    b = np.zeros(ninfo, dtype=np.uint8)
    for i in range(ninfo-1, -1, -1):
        b[i] = prev_bit[i, st]
        st = int(prev_state[i, st])
        if st < 0 and i:
            return None

    if len(b) < 22:
        return None
    body_crc = b[:-6]
    data = body_crc[:-16]
    rx_crc = body_crc[-16:]
    crc_ok = np.array_equal(crc16_bits(data), rx_crc)
    return float(metric[0]), data, crc_ok

def dewhiten(data_bits: np.ndarray) -> bytes:
    lfsr = np.ones(8, dtype=np.uint8)
    out = []
    for k in range(len(data_bits)//8):
        b = data_bits[k*8:(k+1)*8]
        swapped = np.concatenate([b[4:8], b[0:4]])
        plain = np.bitwise_xor(swapped, lfsr)
        out.append(bits_to_int(plain))
        fb = int(lfsr[0] ^ lfsr[2] ^ lfsr[3] ^ lfsr[4])
        lfsr = np.concatenate([lfsr[1:], np.array([fb], dtype=np.uint8)])
    return bytes(out)

@dataclass
class Packet:
    payload: bytes
    crc_ok: bool
    header: Header
    payload_start: int
    fragment_count: int
    viterbi_cost: float
    # Optional additions preserve the six-field positional/GUI packet contract.
    fragments: tuple[FragmentResult, ...] = ()
    payload_diagnostics: dict = field(default_factory=dict)

def extract_payload_fragments(z: np.ndarray, header: Header, overlaps=()):
    """Copy-independent DSP work unit: IQ + canonical header + schedule context.

    This function does not access a live ring or mutate TrackManager. It is the
    fragment-extraction half of a future bounded payload-worker queue boundary.
    """
    info = header.info
    _, frag_bits = payload_layout(info["payload_len"])
    hops = hop_frequencies(info["grid"], info["bw_code"], info["hop_seq_id"],
                           DEFAULT_HEADER_COUNT, len(frag_bits))

    # Anchor the entire known hop plan to the measured first header. This is
    # more robust than independently estimating the center of every GMSK hop.
    common_cfo = header.freq_hz - hops[0]
    payload_start = header.start + DEFAULT_HEADER_COUNT*STAY_HDR
    fragments = []

    for j, nbit in enumerate(frag_bits):
        nominal_start = payload_start + j*STAY_DATA
        freq = hops[DEFAULT_HEADER_COUNT+j] + common_cfo
        lo = max(0, nominal_start-2500)
        hi = min(len(z), nominal_start+STAY_DATA+2500)
        result = FragmentResult(j, float(freq), nominal_start, nbit)
        fragments.append(result)
        if hi-lo < STAY_DATA:
            result.rejection_reason = "fragment_window_unavailable"
            continue
        bb = mix_lowpass(z[lo:hi], float(freq), 350.0)
        # Same arithmetic and timing search as Batch 2+3. Only share the phase
        # array between the 65 trials, as header validation already does.
        phase = np.angle(bb)

        best = None
        for shift in range(-1024, 1025, 32):
            st = nominal_start + shift
            centers = (st-lo) + LOOK + np.arange(nbit+1)*SPB
            if centers[0]-LOOK < 0 or centers[-1]+LOOK >= len(bb):
                continue
            soft = _soft_from_phase(phase, centers)
            score = float(np.mean(np.abs(soft)))
            if best is None or score > best[0]:
                best = (score, soft, shift, centers.astype(int))
        if best is None:
            result.rejection_reason = "no_valid_timing_trial"
            continue
        result.timing_score, soft, result.timing_offset_samples, centers = best
        result.raw_soft_bits = soft[1:].copy()  # discard leading fragment guard
        measure_start = nominal_start+result.timing_offset_samples
        raw = z[max(0, measure_start):min(len(z), measure_start+(nbit+1)*SPB)]
        observe_quality(result, raw, phase, centers, FS, LOOK, overlaps)
    return tuple(fragments)


def decode_payload(z: np.ndarray, header: Header, diagnostics=None, overlaps=()) -> Packet | None:
    """Preserve the raw CRC-valid result; try one noise-weighted FEC rescue.

    No CRC search over masks or thresholds: both attempts use the same existing
    interleaver, puncturing, trellis and CRC. Failed bytes remain failed bytes.
    """
    detail = diagnostics if diagnostics is not None else {}
    started, cpu = time.perf_counter(), time.process_time()
    detail.update(stage="fragment_extraction", header_acquired=True,
                  all_fragment_windows=False, soft_stream_produced=False,
                  viterbi_output_produced=False, fec_attempts=[], selected_method=None)
    if header.info["cr"] != 1:
        detail.update(stage="unsupported_coding_rate")
        return None
    with deferred_quality() as quality:
        fragments = extract_payload_fragments(z, header, overlaps)
        detail["fragment_extraction_wall_s"] = time.perf_counter()-started
        detail["fragment_extraction_cpu_s"] = time.process_time()-cpu
        detail["fragment_quality"] = fragment_summary(fragments)
        detail["fragments"] = [f.diagnostic() for f in fragments]
        if not all(f.valid for f in fragments):
            detail["reason"] = "one_or_more_fragment_windows_unavailable"
            return None
        detail["all_fragment_windows"] = True
        fec_start, fec_cpu = time.perf_counter(), time.process_time()
        coded, _ = payload_layout(header.info["payload_len"])
        raw = np.concatenate([f.raw_soft_bits for f in fragments])
        detail["soft_stream_produced"] = len(raw) == coded
        dec = None
        for method, soft in (("raw", raw), ("noise_weighted", None)):
            if method == "noise_weighted":
                quality.measure_all()
                detail["fragment_quality"] = fragment_summary(fragments)
                detail["fragments"] = [f.diagnostic() for f in fragments]
                soft = np.concatenate([f.soft_bits for f in fragments])
                if np.array_equal(soft, raw):
                    break
            result = viterbi_payload_cr23(deinterleave_payload(soft, coded))
            detail["fec_attempts"].append({"method": method, "viterbi_output": result is not None,
                "phy_crc_ok": bool(result[2]) if result is not None else False,
                "viterbi_cost": float(result[0]) if result is not None else None})
            if result is not None and (dec is None or result[2]):
                dec = result
                detail["selected_method"] = method
            if result is not None and result[2]:
                break
        detail["fec_crc_wall_s"] = time.perf_counter()-fec_start
        detail["fec_crc_cpu_s"] = time.process_time()-fec_cpu
        detail["spectral_quality_measured"] = all(f.spectral_quality_measured for f in fragments)
        detail["quality_periodograms"] = quality.computed
        detail["viterbi_output_produced"] = dec is not None
        detail["total_wall_s"] = time.perf_counter()-started
        detail["total_cpu_s"] = time.process_time()-cpu
        if dec is None:
            detail.update(stage="viterbi", reason="no_terminated_trellis_output")
            return None
        cost, data_bits, crc_ok = dec
        detail.update(stage="complete" if crc_ok else "phy_crc", phy_crc_ok=bool(crc_ok))
        return Packet(dewhiten(data_bits), bool(crc_ok), header,
                      header.start+DEFAULT_HEADER_COUNT*STAY_HDR, len(fragments), cost,
                      fragments, detail)

# Batch 10.2: bounded payload retry after a PHY CRC failure. The forensics
# (docs/ENGINEERING.md, section 9.1) found every payload-CRC loss of the standard
# fleet recoverable from the same IQ with a header estimate a few tens of Hz and
# up to about 650 samples away from the receiver's. Only packets that FAILED are
# retried, so every packet the receiver decodes without it is unchanged.
PAYLOAD_RETRY_GRID = tuple([(0, df) for df in (50., -50., 100., -100.)] +
                           [(dt, df) for dt in (-256, -512, 256) for df in (0., 50., -50., 100., -100.)])


def payload_retry_hypotheses(mode, alternates=()):
    """Ordered (dt samples, df Hz) offsets from the primary header estimate.

    "alternates": the packet's other validated header copies only;
    "grid": those first, then the fixed PAYLOAD_RETRY_GRID. Near-duplicates of the
    primary estimate or of an earlier hypothesis are dropped.
    """
    if mode == "off":
        return ()
    out = []
    def add(dt, df):
        dt, df = int(dt), float(df)
        if abs(dt) < 16 and abs(df) < 10:
            return
        if any(abs(dt-a) < 16 and abs(df-b) < 10 for a, b in out):
            return
        out.append((dt, df))
    for dt, df in alternates:
        add(dt, df)
    if mode == "grid":
        for dt, df in PAYLOAD_RETRY_GRID:
            add(dt, df)
    return tuple(out)


def _descent_application_crc_ok(payload: bytes) -> bool:
    from descent_payload import PACKET_LENGTH, crc16_ccitt_false
    return (len(payload) == PACKET_LENGTH and
            int.from_bytes(payload[53:55], "little") == crc16_ccitt_false(payload[:53]))


def retry_payload(z: np.ndarray, header: Header, detail: dict, overlaps, hypotheses, require_app_crc=True):
    """Try each hypothesis in order after a failed decode_payload(); first pass wins.

    A hypothesis is accepted only when the PHY CRC-16 passes and, with
    require_app_crc, the DeSCENT application CRC-16 passes too (both 16-bit
    checks guard the extra trials against false decodes). The returned Packet
    keeps the track's ORIGINAL header, so packet coordinates are unchanged; the
    offsets used are recorded in detail["payload_retry"].
    """
    record = dict(trials=0, accepted=None, rejected_app_crc=0)
    detail["payload_retry"] = record
    started, cpu = time.perf_counter(), time.process_time()
    try:
        for dt, df in hypotheses:
            record["trials"] += 1
            trial = Header(header.start+dt, header.freq_hz+df, header.cost, header.info)
            sub = {}
            pkt = decode_payload(z, trial, sub, overlaps)
            if pkt is None or not pkt.crc_ok:
                continue
            if require_app_crc and not _descent_application_crc_ok(pkt.payload):
                record["rejected_app_crc"] += 1
                continue
            record["accepted"] = [dt, df]
            detail.update({k: v for k, v in sub.items() if k in (
                "all_fragment_windows", "soft_stream_produced", "viterbi_output_produced",
                "selected_method", "fragment_quality", "fragments", "fec_attempts")})
            detail.update(stage="complete", phy_crc_ok=True, reason=None)
            return Packet(pkt.payload, True, header, header.start+DEFAULT_HEADER_COUNT*STAY_HDR,
                          pkt.fragment_count, pkt.viterbi_cost, pkt.fragments, detail)
        return None
    finally:
        record["wall_s"] = time.perf_counter()-started
        record["cpu_s"] = time.process_time()-cpu


def packet_duration_samples(payload_len: int) -> int:
    _, frag = payload_layout(payload_len)
    return DEFAULT_HEADER_COUNT*STAY_HDR + len(frag)*STAY_DATA

def printable(payload: bytes) -> str:
    try:
        s = payload.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    return s if all((c.isprintable() or c in "\r\n\t") for c in s) else ""

# Keep source-frozen reference kernels for independent numerical checks and
# existing historical AST contracts. Operational calls bind explicitly to the
# Batch 9.5 implementations; these references are not the optimized code.
reference_mix_lowpass = mix_lowpass
reference_soft_from_phase = _soft_from_phase
reference_extract_payload_fragments = extract_payload_fragments
from lrfhss_dsp import (mix_lowpass, soft_from_phase as _soft_from_phase,
                        soft_trials as _soft_trials, extract_payload_fragments,
                        SampledPhase as _SampledPhase)
import lrfhss_fragments as _fragment_module
reference_spectral_power = _fragment_module.spectral_power
from lrfhss_dsp import spectral_power as _spectral_power
_fragment_module.spectral_power = _spectral_power

U8_IQ_DTYPE = "u8iq"
# Batch 10.4b/c: a SIC-dependent packet not admitted to a payload slot within this
# time after its start is skipped (its payload IQ must still be in the ring: 16 s).
SIC_PAYLOAD_DEADLINE_S = 12.0
# Batch 10.4l: payload attempts per SIC-found packet (its SIC attempt plus recoveries from
# new primary estimates); each attempt uses an estimate not tried before.
SIC_TRACK_MAX_ATTEMPTS = 3


class Ring:
    def __init__(self, seconds=12.0):
        self.keep = int(seconds*FS)
        self.chunks = collections.deque()
        # Optional Batch 9.10 companion store of the SAME samples as the source
        # u8 I/Q bytes (2 bytes/sample), kept only when raw bytes are supplied.
        # Identical start/eviction coordinates as `chunks`; owner-only.
        self.raw_chunks = collections.deque()
        self.end = 0
        self.retention_floor = None

    def add(self, start: int, z: np.ndarray, raw: bytes | None = None):
        self.chunks.append((start, z))
        if raw is not None:
            if len(raw) != 2*len(z):
                raise ValueError("raw u8 I/Q length does not match converted samples")
            self.raw_chunks.append((start, raw))
        self.end = start + len(z)
        cutoff = self.end-self.keep
        if self.retention_floor is not None:
            cutoff = min(cutoff, self.retention_floor)
        while self.chunks and self.chunks[0][0]+len(self.chunks[0][1]) < cutoff:
            self.chunks.popleft()
        while self.raw_chunks and self.raw_chunks[0][0]+len(self.raw_chunks[0][1])//2 < cutoff:
            self.raw_chunks.popleft()

    def clear(self):
        self.chunks.clear()
        self.raw_chunks.clear()

    def raw_snapshot(self, start: int, end: int):
        """Owned u8 I/Q bytes for samples [start, end), or None if unavailable.

        u8_iq_to_complex is elementwise, so converting these bytes reproduces
        the complex64 ring samples bit-for-bit (Batch 9.10 payload transport).
        """
        if (not self.raw_chunks or end <= start or start < self.raw_chunks[0][0]
                or end > self.raw_chunks[-1][0]+len(self.raw_chunks[-1][1])//2):
            return None
        parts = []; expected = end
        for s, raw in reversed(self.raw_chunks):
            n = len(raw)//2
            if s >= end:
                continue
            if s+n <= start:
                break
            a = max(start, s); b = min(end, s+n)
            if b != expected:
                return None
            parts.append(memoryview(raw)[2*(a-s):2*(b-s)])
            expected = a
            if a == start:
                break
        if expected != start:
            return None
        return b"".join(reversed(parts))

    def get(self, start: int, end: int):
        if not self.chunks or start < self.chunks[0][0] or end > self.end:
            return None
        parts = []
        for s,x in self.chunks:
            a=max(start,s); b=min(end,s+len(x))
            if b>a:
                parts.append(x[a-s:b-s])
        if not parts:
            return None
        out=np.concatenate(parts)
        return out if len(out)==end-start else None

    def snapshot_bytes(self, start: int, end: int):
        """One immutable copy of a contiguous ring window, with dtype.

        Ring entries remain owner-only. Memoryviews live only through bytes.join;
        workers receive owned bytes and never a ring view. Walk recent chunks
        backwards so short header windows do not scan the entire 16-second ring.
        The conservative ndarray fallback preserves behavior for unusual layouts.
        """
        if not self.chunks or start < self.chunks[0][0] or end > self.end or end <= start:
            return None
        parts=[];count=0;dtype=None;expected=end
        for s,x in reversed(self.chunks):
            if s >= end:
                continue
            if s+len(x) <= start:
                break
            a=max(start,s);b=min(end,s+len(x))
            if b<=a:
                continue
            if b != expected or not x.flags.c_contiguous or (dtype is not None and x.dtype.str!=dtype):
                z=self.get(start,end)
                return (z.tobytes(),z.dtype.str) if z is not None else None
            dtype=x.dtype.str;parts.append(memoryview(x[a-s:b-s]).cast('B'))
            count+=b-a;expected=a
            if a==start:
                break
        if count != end-start or expected != start:
            return None
        return b''.join(reversed(parts)),dtype

from lrfhss_payload_jobs import PayloadJob, PayloadPool
from lrfhss_source_monitor import SourceMonitor
from lrfhss_iq_recorder import IQRecorder, default_recording_path
from pathlib import Path
import uuid

# Measured Batch 8 profile. LiveDecoder's legacy defaults remain API-compatible.
BATCH08_PROFILE = dict(header_workers=4, header_queue_capacity=16,
    header_candidate_capacity=1024, header_validation_budget=None,
    batch_header=True, header_grid_cache=True, payload_workers=6,
    payload_queue_capacity=4)


BATCH09_PROFILE = dict(BATCH08_PROFILE, header_workers=7, payload_workers=5, payload_queue_capacity=8,
    payload_fec_backend="numpy", acquisition_options={"noise_mode":"guard20"},
    legacy_noise_robust=False, exact_refinement_dedup=False)

# Configurable presets, not CPU-model detection. Windows acceptance requires
# the supplied local benchmark; the 6/5 preset was selected on the Work host.
BATCH09_5_PROFILE = dict(BATCH09_PROFILE, header_workers=6)
BATCH09_5_PORTABLE_PROFILE = dict(BATCH09_5_PROFILE, header_workers=5)
# 9.5b changes only immutable snapshot construction by default. Larger ordered
# result capacity remains an opt-in experiment, not a throughput claim.
BATCH09_5B_PROFILE = dict(BATCH09_5_PROFILE, snapshot_copy_mode="direct")

BATCH09_6_REFERENCE = dict(BATCH09_5B_PROFILE, acquisition_execution="inline",
    acquisition_search_backend="reference", phdr_frequency_reuse=False)
BATCH09_6_EXPERIMENTAL = dict(BATCH09_5B_PROFILE, acquisition_execution="process",
    acquisition_search_backend="batched", phdr_frequency_reuse=True)
BATCH09_6_PROFILE = dict(BATCH09_6_REFERENCE)

# Batch 9.7 builds on the clean-tested Batch 9.6 process acquisition path.
BATCH09_7_REFERENCE = dict(BATCH09_6_EXPERIMENTAL, header_workers=9,
    payload_workers=3, payload_queue_capacity=32)
BATCH09_7_PROFILE = dict(BATCH09_7_REFERENCE, header_shared_iq=True,
    header_validation_reuse=True, header_idle_service=True)

# One immutable snapshot and worker transaction per physical candidate.
BATCH09_8_REFERENCE = dict(BATCH09_7_PROFILE, header_workers=8)
BATCH09_8_PROFILE = dict(BATCH09_8_REFERENCE, candidate_transactions=True, header_oscillator_reuse=False)

# Same queues, workers, acquisition and payload processing. Only PHDR DSP varies.
BATCH09_9_REFERENCE = dict(BATCH09_8_PROFILE, phdr_frontend="reference")
BATCH09_9_PROFILE = dict(BATCH09_8_PROFILE, phdr_frontend="screened", phdr_decimation=64)

# Batch 9.10 Change 01/02 step (kept as the comparison/equivalence reference):
# identical receiver arithmetic; payload IQ crosses the process boundary as the
# source u8 I/Q bytes (4x smaller) and is converted in the worker with an exact
# table built from the same u8_iq_to_complex used by the owner ring.
BATCH09_10_U8_PROFILE = dict(BATCH09_9_PROFILE, payload_transport="u8")

# Change 04 step (adopted after A/B #2, kept as the comparison reference): the u8
# transport plus P2, the bit-identical grouped timing-trial bias loop (payload
# and PHDR paths). 8 header / 3 payload workers; queue bounds unchanged from 9.9.
BATCH09_10_P2_PROFILE = dict(BATCH09_10_U8_PROFILE, dsp_soft_trials="grouped")
BATCH09_10_P2_7H4P_PROFILE = dict(BATCH09_10_P2_PROFILE, header_workers=7, payload_workers=4)

# Batch 9.10 adopted profile. Change 12 (after replica A/B #3): u8 + P2 plus
# the DEFERRED header-replica skip (Changes 10-11). A matched replica is dropped
# only after its packet's payload passed the PHY CRC; otherwise it is decoded.
# The immediate skip (Change 08) stays withdrawn (Audit 02, A1).
BATCH09_10_PROFILE = dict(BATCH09_10_P2_PROFILE, header_replica_skip="deferred")
# Opt-in rr 2 profile, qualified on the Intel NUC. Existing profiles stay unchanged.
# GUI-level logging; no process priority, affinity, power or validation changes.
NUC_RR2_20S_V2_PROFILE = dict(BATCH09_10_PROFILE, allowed_hop_ids=(256, 299),
    acquisition_processes=2, header_workers=5, payload_workers=2, trigger_db=10.,
    header_candidate_capacity=2048, candidate_admission_ranks=False, header_queue_capacity=4,
    header_cfo_fft='float32_guarded', header_cfo_tie_margin=.01, phdr_sic='off')
# A/B arm E name (= the adopted profile).
BATCH09_10_REPLICA_SKIP_PROFILE = dict(BATCH09_10_PROFILE)
# A/B arm I: the withdrawn immediate skip, research comparison only.
BATCH09_10_REPLICA_SKIP_IMMEDIATE_PROFILE = dict(BATCH09_10_P2_PROFILE, header_replica_skip="on")
# Batch 10.2 candidate (not adopted): the adopted profile plus the bounded payload
# retry after a PHY CRC failure (other header copies first, then a fixed grid).
BATCH10_2_PAYLOAD_RETRY_PROFILE = dict(BATCH09_10_PROFILE, payload_retry="grid")
# Batch 10 receiver WITHOUT SIC: the 10.2 candidate, adopted after Windows sweep #9
# (948/948 CRC-valid, 1.10x 2/2). The front-end profile until 10.4h; still selectable
# (CLI --throughput-no-sic). The 9.10 release record (acceptance, memory qualification)
# stays on BATCH09_10_PROFILE.
BATCH10_PROFILE = dict(BATCH10_2_PAYLOAD_RETRY_PROFILE)
# Batch 10.4 candidate (not adopted): the front-end profile plus PHDR successive
# interference cancellation on 2 separate, below-normal-priority SIC workers.
BATCH10_4_SIC_PROFILE = dict(BATCH10_PROFILE, phdr_sic="on", sic_workers=2, sic_payload_allowance=2)
# Offline/yield variant: finish every SIC tile at end of stream (file decoding).
BATCH10_4_SIC_FLUSH_PROFILE = dict(BATCH10_4_SIC_PROFILE, sic_eof="flush")
# Variant: one PHDR worker traded for a fourth payload worker (recovered packets
# add about 25% more payload jobs; the payload pool saturates first).
BATCH10_4_SIC_7H4P_PROFILE = dict(BATCH10_4_SIC_PROFILE, header_workers=7, payload_workers=4)
# Batch 10.4e control: the same 7 PHDR / 4 payload split without SIC, so a SIC
# sweep can be compared with a control that differs only by SIC.
BATCH10_7H4P_PROFILE = dict(BATCH10_PROFILE, header_workers=7, payload_workers=4)
# Batch 10.4g candidate: 7H/4P SIC on ONE SIC worker (at most one core of SIC work,
# one rebuild/search plus one SIC payload outstanding), to test whether the 1.25x
# margin returns while keeping part of the 1.10x yield.
BATCH10_4_SIC_7H4P_1W_PROFILE = dict(BATCH10_4_SIC_7H4P_PROFILE, sic_workers=1, sic_payload_allowance=1)
# Batch 10 FRONT-END profile (GUI, CLI --throughput), adopted 2026-10-01 after Windows
# sweep #24: the one-worker SIC receiver with the 10.4h proportional duty, 6 / 6 runs
# passing incl. two 300 s runs at 1.25x, all 948 reference packets kept in every run.
BATCH10_FRONTEND_PROFILE = dict(BATCH10_4_SIC_7H4P_1W_PROFILE)
FRONTEND_SIC_PRESSURE_MODE = "lateness"

class TruncatedIQError(ValueError):
    """finish() found an incomplete trailing unsigned I/Q byte pair (Change 11/12).

    Every complete sample was decoded first; the packets are held for
    take_unreturned_packets(). A subclass of ValueError so existing callers
    still see a ValueError; front ends catch exactly this type.
    """


def _publication_boundary(method):
    """Outermost public packet-returning call: account for every collected packet.

    Batch 9.10 invariant. Internal nested calls pass straight through. When the
    outermost call returns, every packet collected during it is counted as
    returned; a collected packet missing from the returned list is appended
    (and recorded) instead of being lost. If the call raises, the packets it
    had already collected are retained and returned first by the next public
    call (or by take_unreturned_packets()). Decode semantics are unchanged.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "_callback_depth", 0):
            # Change 11 (Audit 02 B4): a packet collected by a nested call made
            # from a callback would also be returned by the outer call. The
            # error is deferred to the outer call like any callback error.
            raise RuntimeError("LiveDecoder public calls are not re-entrant from an event callback")
        if self._api_depth:
            return method(self, *args, **kwargs)
        if self._inflight_packets:
            # Collected by a direct internal call and returned to that caller.
            self.publication["returned_via_internal_call"] += len(self._inflight_packets)
            self._inflight_packets = []
        self._api_depth = 1
        self._deferred_callback_error = None
        try:
            out = method(self, *args, **kwargs)
        except BaseException:
            self._api_depth = 0
            self._deferred_callback_error = None
            self._retain_inflight_after_exception()
            raise
        self._api_depth = 0
        deferred, self._deferred_callback_error = self._deferred_callback_error, None
        if deferred is not None:
            # An event callback failed during this call. The receiver finished
            # its bookkeeping first (so no result is half-incorporated); the
            # packets it collected are kept for the next public call, then the
            # caller sees the callback's exception.
            self._retain_inflight_after_exception()
            raise deferred
        return self._account_returned(out)
    return wrapper


class LiveDecoder:
    """One IQ stream with independent packet hypotheses.

    Continuous sync search watches all three configured header frequencies.
    Acquisition runs on every 50 ms block even while other tracks await data.
    Completed payloads execute in bounded spawn workers. The calling thread
    alone owns acquisition, tracks, event callbacks and result publication.
    """
    BLOCK_BYTES = 100_000

    def __init__(self, trigger_db=DEFAULT_TRIGGER_DB, verbose=False, *, event_callback=None,
                 max_active_tracks=256, payload_workers=1, payload_queue_capacity=4,
                 payload_finish_timeout=120.0, payload_test_delay=0.0,
                 payload_test_fail_tracks=(), payload_test_gate=None,
                 candidate_local_retry=True, header_validation_budget=2, adaptive_header_budget=False,
                 candidate_scheduler="quality_age", schedule_phase_s=None,
                 candidate_clustering="legacy", batch_header=False, header_grid_cache=False,
                 header_workers=0, header_queue_capacity=16, header_candidate_capacity=256,
                 header_max_age_s=12., header_test_gate=None, header_test_delay=0.,
                 acquisition_options=None, payload_fec_backend="numpy",
                 payload_native_library=None, profile_payload_kernels=False,
                 capture_fec_vectors=False, exact_refinement_dedup=False, legacy_noise_robust=False,
                 header_poll_before_acquisition=True, header_transport_pump=True,
                 snapshot_copy_mode="direct", header_outstanding_capacity=None,
                 acquisition_execution="inline", acquisition_search_backend="reference",
                 acquisition_capacity=8, acquisition_timeout=30., phdr_frequency_reuse=False,
                 header_shared_iq=False, header_validation_reuse=False, header_idle_service=False,
                 candidate_transactions=False, candidate_transaction_mode="deferred",
                 header_oscillator_reuse=False, phdr_frontend="reference", phdr_decimation=64,
                 payload_transport="complex64", dsp_soft_trials="reference",
                 header_replica_skip="off", payload_retry="off", phdr_sic="off", sic_workers=2,
                 sic_queue_capacity=4, sic_tile_s=1.0, sic_options=None, sic_test_gate=None,
                 sic_eof="skip", sic_max_inflight=None, sic_payload_allowance=2, sic_pressure=None,
                 allowed_hop_ids=(DEFAULT_HOPSEQ,), acquisition_processes=1,
                 candidate_admission_ranks=True, header_cfo_fft='float64', header_cfo_tie_margin=.01):
        from lrfhss_hop_ids import normalize_hop_ids, header_search_plan
        self.allowed_hop_ids = normalize_hop_ids(allowed_hop_ids)
        if self.allowed_hop_ids != (DEFAULT_HOPSEQ,) and phdr_sic != "off":
            raise ValueError("SIC is untested with multiple hop IDs; select SIC off")
        acquisition_options = dict(acquisition_options or {})
        if not isinstance(candidate_admission_ranks, bool):
            raise ValueError('candidate_admission_ranks must be boolean')
        if not candidate_admission_ranks:
            # This only omits the admission event's diagnostic rank. The
            # quality/age scheduler and its scheduled ranks stay unchanged.
            acquisition_options['admission_ranks'] = False
        if "channel_map" in acquisition_options:
            raise ValueError('channel_map is derived from allowed_hop_ids')
        if isinstance(acquisition_processes, bool) or acquisition_processes not in (1, 2):
            raise ValueError('acquisition_processes must be 1 or 2')
        if acquisition_processes == 2:
            if acquisition_execution != "process" or len(self.allowed_hop_ids) > 2:
                raise ValueError('Two acquisition processes require process mode and at most two hop IDs')
            if len(self.allowed_hop_ids) == 2:
                from lrfhss_acquisition_split import partition_channels
                preliminary_hops, preliminary_map = header_search_plan(self.allowed_hop_ids, hop_frequencies)
                partition_channels(preliminary_hops, dict(acquisition_options, channel_map=preliminary_map))
        self.acquisition_processes = acquisition_processes
        if snapshot_copy_mode not in ("direct", "concatenate"):
            raise ValueError("Unknown snapshot copy mode")
        if payload_transport not in ("complex64", "u8"):
            raise ValueError("Unknown payload transport")
        self.payload_transport = payload_transport
        # Batch 9.10: initial PHDR candidates that sit exactly where an
        # established packet's own header replica is expected. "off" (default)
        # is unchanged behaviour; "shadow" validates everything and only
        # records the prediction and the actual outcome; "deferred" (Change 10)
        # parks them and drops them only after the matched packet passed its
        # PHY CRC, otherwise validates them; "on" (research arm only) skips
        # them immediately and can lose a packet whose first header was
        # corrupted yet CRC-valid (Audit 02, A1).
        if header_replica_skip not in ("off", "shadow", "on", "deferred"):
            raise ValueError("Unknown header_replica_skip mode")
        if header_replica_skip != "off" and not header_workers:
            raise ValueError("header_replica_skip requires the asynchronous PHDR pipeline")
        self.header_replica_skip = header_replica_skip
        # Batch 10.2: bounded payload retry after a PHY CRC failure ("off" is the
        # unchanged behaviour; see payload_retry_hypotheses / retry_payload).
        if payload_retry not in ("off", "alternates", "grid", "adaptive"):
            raise ValueError("Unknown payload_retry mode")
        self.payload_retry = payload_retry
        # Batch 10.4: PHDR successive interference cancellation ("off" is the
        # unchanged receiver; see lrfhss_sic_owner.SicScheduler).
        if phdr_sic not in ("off", "on"):
            raise ValueError("Unknown phdr_sic mode")
        if phdr_sic != "off" and not header_workers:
            raise ValueError("phdr_sic requires the asynchronous PHDR pipeline")
        self.phdr_sic = phdr_sic
        # At end of stream, "skip" abandons SIC work still queued (a live stream
        # has no end; the capacity gates require a short final drain); "flush"
        # finishes every tile first (offline file decoding).
        if sic_eof not in ("skip", "flush"):
            raise ValueError("Unknown sic_eof mode")
        self.sic_eof = sic_eof
        import lrfhss_dsp
        if dsp_soft_trials not in lrfhss_dsp.SOFT_TRIALS_IMPLEMENTATIONS:
            raise ValueError("Unknown dsp_soft_trials implementation")
        # Owner-process DSP (inline/legacy paths); workers get it per job.
        self.dsp_soft_trials = dsp_soft_trials
        lrfhss_dsp.set_soft_trials(dsp_soft_trials)
        if schedule_phase_s is not None:
            raise ValueError("Batch 9.6 is schedule-blind; schedule_phase_s is forbidden")
        if acquisition_execution not in ("inline", "process"):
            raise ValueError("Unknown acquisition execution mode")
        self.acquisition_execution = acquisition_execution
        self.phdr_frequency_reuse = phdr_frequency_reuse
        self.header_shared_iq = header_shared_iq
        self.header_validation_reuse = header_validation_reuse
        self.header_idle_service = header_idle_service
        if candidate_transactions and (not header_workers or not phdr_frequency_reuse):
            raise ValueError("Candidate transactions require PHDR workers and exact frequency reuse")
        self.candidate_transactions = candidate_transactions
        if candidate_transaction_mode not in ('deferred', 'eager'):
            raise ValueError('Unknown candidate transaction mode')
        self.candidate_transaction_mode = candidate_transaction_mode
        self.header_oscillator_reuse = header_oscillator_reuse
        if header_cfo_fft not in ('float64','float32_guarded'):
            raise ValueError('Unknown header CFO FFT mode')
        if header_cfo_fft!='float64' and (not header_workers or not header_grid_cache):
            raise ValueError('Guarded CFO FFT requires PHDR workers and the header grid cache')
        self.header_cfo_fft=header_cfo_fft
        if header_cfo_tie_margin != .01:
            raise ValueError('The qualified CFO near-tie guard is fixed at 1%')
        self.header_cfo_tie_margin=float(header_cfo_tie_margin)
        if phdr_frontend not in ('reference', 'screened'):
            raise ValueError('Unknown PHDR frontend mode')
        if phdr_decimation not in (16, 32, 64, 128, 256, 512):
            raise ValueError('Unsupported PHDR decimation factor')
        if phdr_frontend != 'reference' and (not header_workers or not candidate_transactions or not batch_header):
            raise ValueError('Screened PHDR frontend requires batched candidate transactions and workers')
        self.phdr_frontend, self.phdr_decimation = phdr_frontend, phdr_decimation
        self.acquisition_process = None
        self.input_abs_sample = 0
        self._committing_acquisition = False
        self.snapshot_copy_mode = snapshot_copy_mode
        self.trigger_ratio=10**(trigger_db/10)
        self.verbose=verbose
        self.ring=Ring(16.0 if header_workers else 12.0)
        self.abs_sample=0
        self.noise=None
        self.sos=signal.butter(4,3000,btype="low",fs=FS,output="sos")
        self.zi=np.zeros((self.sos.shape[0],2),dtype=np.complex128)
        from lrfhss_noise import SpectralNoise
        self._legacy_noise_estimator = SpectralNoise(FS,64,self.sos,guard=True) if legacy_noise_robust else None
        self._event_callback = event_callback
        self.track_manager = TrackManager(event_callback=self._track_event,
                                          max_active=max_active_tracks)
        self._raw_pending = bytearray()
        self._finished = False
        self.finish_reason = None         # set by finish(): "finished" = every admitted job drained (10.4k)
        # Continuation of one observed energy burst, not a receiver dead time.
        self._burst_onset = None
        self._burst_end = -1
        # One exact continuing-burst identity outlives the bounded track
        # history. Otherwise a long continuous burst can re-offer its original
        # onset after history eviction, when its immutable IQ is long gone.
        self._offered_energy_onset = None
        header_hops = hop_frequencies(DEFAULT_GRID, DEFAULT_BW_CODE, DEFAULT_HOPSEQ,
                                      DEFAULT_HEADER_COUNT, 0)
        if self.allowed_hop_ids != (DEFAULT_HOPSEQ,):
            header_hops, channel_map = header_search_plan(self.allowed_hop_ids, hop_frequencies,
                                                         DEFAULT_GRID, DEFAULT_BW_CODE)
            acquisition_options["channel_map"] = channel_map
        self.acquisition = HeaderAcquisition(FS, SPB, header_hops, self.trigger_ratio,
                                             self._acquisition_event, scheduler=candidate_scheduler,
                                             schedule_phase_s=schedule_phase_s, clustering=candidate_clustering,
                                             **dict({"noise_mode":"guard20", "search_backend":acquisition_search_backend}, **(acquisition_options or {})))
        self.batch_header = batch_header
        self.header_poll_before_acquisition = header_poll_before_acquisition
        from lrfhss_header_grids import HeaderSearchGrids
        self.header_grids = HeaderSearchGrids(FS) if header_grid_cache else None
        from lrfhss_timing import Timings
        self.timings = Timings()
        self.processing = collections.Counter()
        self.payload_counts = collections.Counter()
        self._energy_candidates = 0
        self.receiver_id = uuid.uuid4().hex
        if payload_retry == "adaptive":
            from lrfhss_adaptive_retry import AdaptivePayloadPool
            self.payload_pool = AdaptivePayloadPool(payload_workers, payload_queue_capacity, payload_test_gate)
        else:
            self.payload_pool = PayloadPool(payload_workers, payload_queue_capacity, payload_test_gate)
        if payload_finish_timeout <= 0:
            raise ValueError("payload finish timeout must be positive")
        self.payload_finish_timeout = payload_finish_timeout
        self._payload_test_delay = payload_test_delay
        self._payload_test_fail_tracks = frozenset(payload_test_fail_tracks)
        if payload_fec_backend not in ("python", "numpy", "native"):
            raise ValueError("Unknown payload FEC backend")
        self.payload_fec_backend = payload_fec_backend
        self.payload_native_library = payload_native_library
        self.profile_payload_kernels = profile_payload_kernels
        self.capture_fec_vectors = capture_fec_vectors
        self._payload_pending = {}
        self._payload_processing = set()
        # Batch 9.10 publication ledger. Everything here is bounded: per-job
        # state lives only while the job is outstanding; history is counts plus
        # small fixed-length identity samples.
        self._payload_inbox = collections.deque()
        self._api_depth = 0
        self._callback_depth = 0
        self._deferred_callback_error = None
        self._inflight_packets = []
        self._recovered_packets = []
        self.publication = collections.Counter()
        self._publication_samples = collections.deque(maxlen=64)
        self.runtime_counts = collections.Counter()
        if header_validation_budget is not None and header_validation_budget < 1:
            raise ValueError("header validation budget must be positive or None")
        if adaptive_header_budget and header_validation_budget is None:
            raise ValueError("Adaptive budget cannot be unlimited")
        from lrfhss_budget import AdaptiveHeaderBudget
        self.adaptive_budget=AdaptiveHeaderBudget() if adaptive_header_budget else None
        self.header_validation_budget = header_validation_budget
        self.candidate_local_retry = candidate_local_retry
        self._refinement_pending = collections.deque()
        self.exact_refinement_dedup = exact_refinement_dedup
        self._header_tokens = header_validation_budget
        if header_workers < 0 or header_candidate_capacity < 1:
            raise ValueError("Invalid header worker/candidate capacity")
        self.acquisition.MAX_PENDING = header_candidate_capacity
        self.header_pipeline = None
        if header_workers:
            from lrfhss_header_pipeline import HeaderPipeline
            self.header_pipeline = HeaderPipeline(self, header_workers, header_queue_capacity,
                max_age_s=header_max_age_s, test_gate=header_test_gate, test_delay=header_test_delay,
                outstanding_capacity=header_outstanding_capacity)
        self._payload_enqueued_wall = {}
        if self.header_pipeline is not None and header_transport_pump:
            self.acquisition.progress_callback = self.header_pipeline.pool.pump
        if acquisition_execution == "process":
            from lrfhss_acquisition_process import AcquisitionProcess
            process_class = AcquisitionProcess
            if acquisition_processes == 2 and len(self.allowed_hop_ids) == 2:
                from lrfhss_acquisition_split import SplitAcquisitionProcess
                process_class = SplitAcquisitionProcess
            options = dict({"noise_mode":"guard20", "search_backend":acquisition_search_backend},
                           **(acquisition_options or {}))
            if options.get("diagnostic_callback") is not None:
                raise ValueError("Use owner candidate events for process-mode diagnostics")
            self.acquisition_process = process_class(FS, SPB, header_hops,
                self.trigger_ratio, options, capacity=acquisition_capacity,
                timeout=acquisition_timeout)
            # Maximum lead is capacity * one 50-ms block. No arbitrary ring growth.
            self.ring.keep += acquisition_capacity*(self.BLOCK_BYTES//2)
        self.sic = None
        if phdr_sic == "on":
            from lrfhss_sic_owner import SicScheduler
            self.sic = SicScheduler(self, workers=sic_workers, capacity=sic_queue_capacity, tile_s=sic_tile_s,
                                    max_inflight=sic_max_inflight, payload_allowance=sic_payload_allowance,
                                    pressure=sic_pressure, options=sic_options, test_gate=sic_test_gate)
        # SIC-dependent payload jobs in flight (bounded by sic_payload_allowance)
        self._sic_payload_jobs = set()
        self._primary_sic_payload_jobs = set()
        # Input lag: reported by the caller (note_input_lag), else estimated from
        # the wall clock against the samples received (real-time sources).
        self._input_lag_note = None
        self._input_clock_offset = None

    def _acquisition_event(self, event, sample, **fields):
        self._track_event(dict(event=event, sample=sample,
                               active_tracks=len(self.track_manager.active_tracks), **fields))

    def _timed(self, stage, function, *args, **kwargs):
        started = time.perf_counter()
        cpu = time.process_time()
        thread_cpu = time.thread_time()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = time.perf_counter()-started
            self.timings.add(stage, elapsed)
            self.processing[stage+"_wall_s"] += elapsed
            self.processing[stage+"_cpu_s"] += time.process_time()-cpu
            self.processing[stage+"_owner_thread_cpu_s"] += time.thread_time()-thread_cpu
            self.processing[stage+"_calls"] += 1
            self.processing[stage+"_max_s"] = max(self.processing[stage+"_max_s"], elapsed)

    def _track_event(self, event):
        event.setdefault("receiver_id", getattr(self, "receiver_id", None))
        event.setdefault("owner_wall", time.perf_counter())
        sic = getattr(self, 'sic', None)
        pressure_transition = event['event'] in getattr(sic, 'primary_loss_events', ()) and sic.observe_primary_loss(event)
        if self.verbose:
            print(f"[track {event.get('track_id', '-')}] {event['event']} "
                  f"at {event['sample']/FS:.6f}s active={event['active_tracks']} "
                  f"{event.get('reason', '')}")
        if self._event_callback is not None:
            self._callback_depth += 1
            try:
                self._event_callback(event)
            except BaseException as exc:
                # Batch 9.10: never abandon a worker result half-incorporated.
                # Inside a public call the first callback error is deferred to
                # the end of that call (_publication_boundary re-raises it).
                # Change 11 (Audit 02 A5): this includes KeyboardInterrupt and
                # other BaseExceptions; an interruption replaces an ordinary
                # deferred error so it is never masked.
                self.publication["event_callback_exceptions"] += 1
                if not self._api_depth:
                    raise
                previous = self._deferred_callback_error
                if previous is None or (isinstance(previous, Exception) and not isinstance(exc, Exception)):
                    self._deferred_callback_error = exc
            finally:
                self._callback_depth -= 1

        if pressure_transition:
            self._acquisition_event('sic_suspended', self.abs_sample, smoothed=dict(sic.gate.smoothed))

    def _retain_inflight_after_exception(self):
        if self._inflight_packets:
            self.publication["interrupted_public_calls"] += 1
            self.publication["packets_recovered_after_exception"] += len(self._inflight_packets)
            self._recovered_packets.extend(self._inflight_packets)
            self._inflight_packets = []

    def _reset_header_budget(self, phase):
        if self.adaptive_budget is None:
            self._header_tokens=self.header_validation_budget
            return
        candidates=[c for c in self.acquisition.pending if self.abs_sample>=c.header_start+int(.335*FS)]+list(self._refinement_pending)
        age=max(((self.abs_sample-c.observed_sample)/FS for c in candidates),default=0.)
        self._header_tokens=self.adaptive_budget.update(len(candidates),age)
        self._acquisition_event("header_budget",self.abs_sample,phase=phase,active_budget=self._header_tokens,
            lower_bound=2,upper_bound=4,pressure_backlog=len(candidates),oldest_age_s=age,
            transitions=self.adaptive_budget.transitions,high_streak=self.adaptive_budget.high_streak,low_streak=self.adaptive_budget.low_streak)

    def note_input_lag(self, seconds):
        """Caller-measured input lag (seconds the block being fed is behind its
        real-time deadline). Used only by overload control of optional work."""
        self._input_lag_note = (max(0., float(seconds)), time.perf_counter())

    def note_input_idle(self):
        """The live adapter observed an empty input socket, before idle service.

        Waiting for the next USB/TCP burst is not processing backlog. Re-anchor
        the fallback clock at this measured empty-source boundary; otherwise
        ordinary >50 ms deliveries prevent SIC's two-second calm hold forever
        after its first suspension. While the socket stays readable, lag still
        grows normally if decoding falls behind. Explicit caller lag reports
        and every worker/queue pressure signal retain their existing authority.
        """
        self._input_clock_offset = time.perf_counter() - self.input_abs_sample/FS

    def input_lag_s(self):
        """Current input lag: a caller report from the last second, else a wall-clock estimate."""
        now = time.perf_counter()
        note = self._input_lag_note
        if note is not None and now - note[1] <= 1.0:
            return note[0]
        offset = now - self.input_abs_sample/FS
        if self._input_clock_offset is None or offset < self._input_clock_offset:
            self._input_clock_offset = offset
        return max(0., offset - self._input_clock_offset)

    @_publication_boundary
    def feed_raw(self, raw: bytes):
        """Return ordinary Packet objects; accept arbitrary byte chunk boundaries."""
        if self._finished:
            raise RuntimeError("Cannot feed IQ after finish()")
        self._reset_header_budget("feed")
        active_budget=self._header_tokens
        validations_before=self.runtime_counts["header_validations_attempted"]
        self._raw_pending.extend(raw)
        # A completed header used to wait through the entire next acquisition
        # call before the owner could commit it and dispatch waiting work.
        # Service existing RF-complete work first, with the SAME per-feed
        # budget and ordered commit path. No IQ, thresholds or jobs are pruned.
        decoded = (self._service() if self.header_pipeline is not None and
                   self.header_poll_before_acquisition else self.poll())
        # Batch 9.10: remove each block from the pending buffer BEFORE processing
        # it, so an exception can never cause the same IQ to be fed twice.
        while len(self._raw_pending) >= self.BLOCK_BYTES:
            block = bytes(self._raw_pending[:self.BLOCK_BYTES])
            del self._raw_pending[:self.BLOCK_BYTES]
            decoded.extend(self._feed_block(block))
        attempts=self.runtime_counts["header_validations_attempted"]-validations_before
        self.runtime_counts["validations_per_feed_max"]=max(self.runtime_counts["validations_per_feed_max"],attempts)
        self._acquisition_event("header_feed_summary",self.abs_sample,validations_attempted=attempts,
            validation_backlog_depth=len(self.acquisition.pending),refinement_backlog_depth=len(self._refinement_pending),active_validation_budget=active_budget)
        return decoded

    def _feed_block(self, raw):
        if self.acquisition_process is not None:
            return self._feed_acquisition_process(raw)
        z=u8_iq_to_complex(raw)
        start=self.abs_sample
        self.abs_sample += len(z)
        self.input_abs_sample = self.abs_sample
        anchors = [max(0,c.header_start-35000) for c in
                   list(self.acquisition.pending)+list(self._refinement_pending)]
        anchors += [max(0,t.detected_sample-50000) for t in self.track_manager.active_tracks.values()
                    if t.header is None]
        self.ring.retention_floor = min(anchors) if anchors else None
        if self.header_pipeline is not None and self.ring.retention_floor is not None:
            self.ring.retention_floor = max(self.ring.retention_floor, self.abs_sample-self.ring.keep)
        if self.payload_transport == "u8":
            self.ring.add(start, z, raw)
        else:
            self.ring.add(start,z)
        self.runtime_counts["max_ring_iq_bytes"]=max(self.runtime_counts["max_ring_iq_bytes"],sum(x.nbytes for _,x in self.ring.chunks))
        self._timed("legacy_trigger", self._trigger, z, start)
        self._timed("acquisition", self.acquisition.feed, z, start)
        return self._service()

    def _feed_acquisition_process(self, raw):
        pipeline = self.acquisition_process
        decoded = []
        while pipeline.outstanding >= pipeline.capacity:
            before = time.perf_counter()
            decoded.extend(self._drain_acquisition(wait=True))
            pipeline.counters["blocked_admission_wall_s"] += time.perf_counter()-before
        start = self.input_abs_sample
        z = u8_iq_to_complex(raw)
        self.input_abs_sample += len(z)
        # The process cannot lead the committed owner state by more than capacity
        # blocks. Ring's fixed extension covers that lead plus normal retention.
        self.ring.retention_floor = None
        if self.payload_transport == "u8":
            self.ring.add(start, z, raw)
        else:
            self.ring.add(start, z)
        self.runtime_counts["max_ring_iq_bytes"] = max(
            self.runtime_counts["max_ring_iq_bytes"], sum(x.nbytes for _,x in self.ring.chunks))
        pipeline.submit(raw, start)
        decoded.extend(self._drain_acquisition())
        return decoded

    def _drain_acquisition(self, wait=False):
        if self.acquisition_process is None or self._committing_acquisition:
            return []
        self._committing_acquisition = True
        decoded = []
        try:
            for result in self.acquisition_process.collect(wait=wait):
                start, end = result["start"], result["end"]
                if start != self.abs_sample:
                    raise RuntimeError("Acquisition watermark is not contiguous")
                z = self.ring.get(start, end)
                if z is None:
                    raise RuntimeError("Acquisition referenced expired IQ")
                self.abs_sample = end
                self._reset_header_budget("acquisition_commit")
                self._timed("legacy_trigger", self._trigger, z, start)
                for fields in result["offers"]:
                    self.acquisition._offer(*fields)
                self.acquisition.noise = result["noise"]
                self.acquisition.energy_references = result["energy_references"]
                self.acquisition.counts.update(result["counts"])
                for key, value in result["processing"].items():
                    if key.endswith("_max_s"):
                        self.acquisition.processing[key] = max(self.acquisition.processing[key], value)
                    else:
                        self.acquisition.processing[key] += value
                self.processing["acquisition_process_wall_s"] += result["wall_s"]
                self.processing["acquisition_process_cpu_s"] += result["cpu_s"]
                decoded.extend(self._service())
        finally:
            self._committing_acquisition = False
        return decoded

    def _trigger(self,z,start):
        # The historical energy/autoset path is calibrated only for 0x100.
        if DEFAULT_HOPSEQ not in getattr(self, "allowed_hop_ids", (DEFAULT_HOPSEQ,)):
            return
        n=np.arange(len(z),dtype=np.float64)
        phase=phase_cycles(TRIGGER_HEADER_HZ,start,FS)
        mixed=z*np.exp(-1j*2*np.pi*(TRIGGER_HEADER_HZ*n/FS+phase))
        y,self.zi=signal.sosfilt(self.sos,mixed,zi=self.zi)
        p=np.abs(y)**2
        cp=float(np.mean(p))
        if self.noise is None:
            self.noise=cp
        if self._legacy_noise_estimator is not None:
            measured_noise = self._legacy_noise_estimator.update(y[(-start)%64::64],len(z)/FS)
            if measured_noise is not None:
                self.noise = measured_noise
        elif cp < self.noise*self.trigger_ratio:
            self.noise=0.97*self.noise+0.03*cp

        # Keep the baseline block-energy gate and 2.048 ms onset estimator.
        # Every distinct energy burst can propose a candidate, even while an
        # earlier packet remains active. Adjacent observations of the *same*
        # burst retain their original onset and are deduplicated by the manager.
        win=2048
        if cp <= self.noise*self.trigger_ratio or len(p) < win:
            return
        cs=np.cumsum(np.concatenate(([0.0],p)))
        ma=(cs[win:]-cs[:-win])/win
        hit=np.flatnonzero(ma > self.noise*self.trigger_ratio)
        if not len(hit):
            return
        groups=np.split(hit, np.flatnonzero(np.diff(hit) > win)+1)
        for group in groups:
            onset=start+int(group[0])
            energy_end=start+int(group[-1])+win
            if self._burst_onset is not None and onset <= self._burst_end+win:
                onset=self._burst_onset
            else:
                self._burst_onset=onset
            self._burst_end=energy_end
            if (onset == self._offered_energy_onset and self.ring.chunks and
                    onset+340000 < self.ring.chunks[0][0]):
                # The identical hypothesis was already offered, and its full
                # header window now predates retained IQ. Keep normal manager
                # duplicate accounting while the original window is usable.
                self.runtime_counts["legacy_exact_duplicate_observations_suppressed"] += 1
                continue
            track = self._timed("track_creation_update", self.track_manager.offer_candidate, onset, self.abs_sample, TRIGGER_HEADER_HZ)
            if track is not None:
                self._offered_energy_onset = onset
                track.acquisition_energy_ratio = cp/self.noise
                self._energy_candidates += 1

    def _accept_physical(self, physical, index, *, candidate_id, source, score, energy,
                         track=None):
        info = physical.info
        if info["hop_seq_id"] not in getattr(self, "allowed_hop_ids", (DEFAULT_HOPSEQ,)):
            self.runtime_counts["disallowed_hop_id_rejections"] += 1
            return None
        _, fragments = payload_layout(info["payload_len"])
        hops = hop_frequencies(info["grid"], info["bw_code"], info["hop_seq_id"],
                               DEFAULT_HEADER_COUNT, len(fragments))
        cfo = float(physical.freq_hz-hops[index])
        start = physical.start-index*STAY_HDR
        # Packet.header keeps its established first-header reference semantics.
        # Actual countdown bits, RF frequency and time remain in HeaderObservation.
        canonical_info = info.copy()
        canonical_info["sync_index"] = DEFAULT_HEADER_COUNT-1
        canonical = Header(start, float(hops[0]+cfo), physical.cost, canonical_info)
        observation = HeaderObservation(candidate_id, index, physical.start, start,
            physical.freq_hz, cfo, score, physical.cost, info.copy(), source, energy)
        end = start+packet_duration_samples(info["payload_len"])+int(.03*FS)
        manager = self.track_manager
        associated = self._timed("header_association", manager.observe_header, canonical, observation, sample=self.abs_sample,
            payload_start=start+DEFAULT_HEADER_COUNT*STAY_HDR, expected_end=end, hops_hz=hops,
            candidate_track=track)
        if associated is not None and source != "sic" and getattr(associated, "sic_origin", False):
            # Batch 10.4h: the primary path validated a packet SIC found first. It no longer
            # depends on SIC (``sic_origin`` = current dependence; ``sic_discovered`` keeps the
            # provenance): primary payload admission and primary publication rules apply. A
            # payload already submitted is not submitted again (it left active_tracks).
            associated.sic_origin = False
            self.runtime_counts["sic_tracks_confirmed_by_primary"] += 1
            self._acquisition_event("sic_track_confirmed_by_primary", self.abs_sample,
                                    track_id=associated.track_id, physical_header=index+1)
        if associated is not None and source != "sic" and getattr(associated, "sic_discovered", False):
            self._note_primary_estimate(associated, canonical, cfo, start, end, hops, physical.cost, index)
            self._recover_failed_track(associated)
        return associated

    def _has_new_estimate(self, track):
        """A primary estimate exists that no payload attempt of this track has used, and the
        attempt budget allows another one (10.4l)."""
        est = getattr(track, "primary_estimate", None)
        if est is None or getattr(track, "payload_attempt", 0) + 1 >= SIC_TRACK_MAX_ATTEMPTS:
            return False
        return (est["header"].start, est["header"].freq_hz) not in getattr(track, "attempted_estimates", set())

    def _recover_failed_track(self, track):
        """Batch 10.4l (audit 2026-10-05, F1): a SIC-found packet whose decode failed (published
        as a CRC-failed packet, or no packet) gets one more attempt when a primary header brings
        an estimate not tried yet, while its IQ is retained. The earlier failed packet stays a
        diagnostic record; a CRC-valid result is published with ``recovers_failed_job``."""
        manager = self.track_manager
        failed = ((track.status == "completed" and track.reason == "phy_crc_failed")
                  or (track.status == "rejected" and track.reason == "payload_decode_failed"))
        if (not failed or track.track_id in manager.active_tracks or track.track_id in manager.payload_tracks
                or self._finished or self.payload_pool.closed or not self._has_new_estimate(track)):
            return False
        manager.revive(track, self.abs_sample, "new_primary_estimate_after_failure")
        self.runtime_counts["sic_failed_track_recoveries"] += 1
        self._schedule_primary_attempt(track, "new_primary_estimate_after_failure")
        return True

    def _note_primary_estimate(self, track, header, cfo, start, end, hops, cost, index):
        """Batch 10.4k (review 2026-10-02, finding 2): a SIC-found packet keeps the best
        primary estimate (lowest header cost, then lowest copy index). Before payload
        submission it replaces the SIC estimate at once; after submission it is the
        estimate a failed SIC attempt retries with. The SIC observation stays in
        ``track.headers`` and becomes a payload-retry alternate."""
        best = getattr(track, "primary_estimate", None)
        if best is not None and (best["cost"], best["index"]) <= (cost, index):
            return
        track.primary_estimate = dict(header=header, cfo_hz=cfo, payload_start=start+DEFAULT_HEADER_COUNT*STAY_HDR,
                                      expected_end=end, hops_hz=tuple(hops), cost=cost, index=index)
        if track.track_id in self.track_manager.active_tracks:
            self._apply_primary_estimate(track, "primary_confirmation")

    def _apply_primary_estimate(self, track, reason):
        """Re-anchor the track on its primary estimate; False if it already uses it."""
        est = getattr(track, "primary_estimate", None)
        h = track.header
        if est is None or (h is not None and h.start == est["header"].start and h.freq_hz == est["header"].freq_hz):
            return False
        self.track_manager.reanchor(track, est["header"], cfo_hz=est["cfo_hz"], payload_start=est["payload_start"],
                                    expected_end=est["expected_end"], hops_hz=est["hops_hz"],
                                    sample=self.abs_sample, reason=reason)
        self.runtime_counts["sic_tracks_reanchored_to_primary"] += 1
        return True

    def _other_transmissions(self, target, base):
        """Snapshot schedules of validated overlapping tracks, in local samples.

        Include recently completed tracks so processing order cannot hide the
        first packet's occupied slots from the second packet's diagnostics.
        """
        spans = []
        known = tuple(self.track_manager._known())
        for other in known:
            if other is target or other.header is None:
                continue
            if (other.expected_end_sample <= target.header.start or
                    other.header.start >= target.expected_end_sample):
                continue
            h = other.header
            for j in range(DEFAULT_HEADER_COUNT):
                start = h.start+j*STAY_HDR-base
                spans.append((start, start+STAY_HDR, other.expected_hops_hz[j]+other.cfo_hz))
            _, bits = payload_layout(h.info["payload_len"])
            for j, nbit in enumerate(bits):
                start = other.payload_start+j*STAY_DATA-base
                spans.append((start, start+(nbit+1)*SPB,
                              other.expected_hops_hz[DEFAULT_HEADER_COUNT+j]+other.cfo_hz))
        return tuple(spans)

    def _service(self):
        decoded=self.poll()
        manager=self.track_manager
        if self.header_pipeline is not None:
            self.header_pipeline.service()
        else:
            self._service_headers()
        self._submit_ready_payloads()
        return decoded

    def _service_headers(self):
        manager=self.track_manager
        # Preserve the successful first-header energy calibration, then process
        # continuously discovered hypotheses. Neither path reserves the channel.
        for track in list(manager.active_tracks.values()):
            onset=track.detected_sample
            track.last_processed_sample=self.abs_sample
            if track.header is None and self.abs_sample >= onset+int(.34*FS):
                if self._header_tokens == 0:
                    continue
                # Reserve service for old sync candidates under sustained legacy traffic.
                if (self._header_tokens == 1 and self.acquisition.pending and
                    self.abs_sample-min(c.observed_sample for c in self.acquisition.pending)>int(.6*FS)):
                    continue
                self._consume_header_token()
                base=max(0,onset-int(.05*FS))
                stop=onset+int(.34*FS)
                zz=self._timed("header_extraction", self.ring.get, base,stop)
                if zz is None:
                    manager.retire(track, "expired", self.abs_sample, "header_iq_unavailable")
                    continue
                detail = {}
                h=self._timed("legacy_header", decode_first_header, zz, onset-base, detail)
                if h is None:
                    manager.retire(track, "rejected", self.abs_sample, "header_validation_failed",
                                   stage="header_validation", rejection_detail=detail,
                                   candidate_sample=onset+round(4.25*SPB), physical_header=1,
                                   inferred_packet_start=onset+round(4.25*SPB),
                                   energy_ratio=track.acquisition_energy_ratio,
                                   **{k: detail.get(k) for k in ("frequency_hz", "cfo_hz",
                                       "header_crc_ok", "viterbi_metric", "sync_correlation")})
                    continue
                score=detail.get("sync_correlation", 0.0)
                h.start += base
                self._accept_physical(h, 0, candidate_id=-track.track_id, source="legacy_energy",
                                      score=score, energy=track.acquisition_energy_ratio, track=track)
        # Old local retries receive one slot before newer candidates, preventing starvation.
        if (self._refinement_pending and self._header_tokens != 0 and
                self.abs_sample-self._refinement_pending[0].observed_sample>int(1.5*FS)):
            self._service_refinements(max_attempts=1)
        for candidate in self.acquisition.ready(self.abs_sample, self._header_tokens):
            self._consume_header_token()
            base=max(0, candidate.header_start-int(.035*FS))
            stop=candidate.header_start+int(.29*FS)
            zz=self._timed("header_extraction", self.ring.get, base,stop)
            if zz is None:
                self._acquisition_event("header_candidate_rejected", self.abs_sample,
                    candidate_id=candidate.candidate_id, candidate_sample=candidate.header_start,
                    physical_header=candidate.physical_index+1, reason="header_iq_unavailable")
                continue
            self._acquisition_event("header_validation_started", self.abs_sample,
                candidate_id=candidate.candidate_id, candidate_sample=candidate.header_start,
                inferred_packet_start=candidate.header_start-candidate.physical_index*STAY_HDR,
                physical_header=candidate.physical_index+1, source=candidate.source,
                candidate_cfo_hz=candidate.frequency_hz-candidate.nominal_hz,
                candidate_age_samples=self.abs_sample-candidate.observed_sample)
            h, detail=self._timed("search_header", decode_physical_header, zz, base, candidate,
                                 counters=self.processing, batch_header=self.batch_header, header_grids=self.header_grids,
                                 frequency_reuse=self.phdr_frequency_reuse,
                                 frequency_selection=getattr(candidate,"frequency_selection",None))
            selection=detail.pop("frequency_selection",None)
            if selection is not None:candidate.frequency_selection=selection
            self._acquisition_event("header_validation_result", self.abs_sample,
                candidate_id=candidate.candidate_id, candidate_sample=candidate.header_start,
                inferred_packet_start=candidate.header_start-candidate.physical_index*STAY_HDR,
                physical_header=candidate.physical_index+1, valid=h is not None,
                candidate_cfo_hz=candidate.frequency_hz-candidate.nominal_hz,
                refinement_delta_hz=detail["frequency_hz"]-candidate.frequency_hz, **detail)
            if (h is None and self.candidate_local_retry and candidate.source == "sync"
                    and (candidate.sync_correlation or 0) >= .6):
                if len(self._refinement_pending)<self.acquisition.MAX_PENDING:
                    self._refinement_pending.append(candidate)
                else:
                    self._acquisition_event("header_refinement_rejected",self.abs_sample,
                        candidate_id=candidate.candidate_id,reason="refinement_capacity")
            if h is None:
                self._acquisition_event("header_candidate_rejected", self.abs_sample,
                    candidate_id=candidate.candidate_id, candidate_sample=candidate.header_start,
                    physical_header=candidate.physical_index+1,
                    inferred_packet_start=candidate.header_start-candidate.physical_index*STAY_HDR,
                    source=candidate.source, energy_ratio=candidate.energy_ratio, **detail)
                continue
            self._accept_physical(h, candidate.physical_index,
                candidate_id=candidate.candidate_id, source=candidate.source,
                score=detail["sync_correlation"], energy=candidate.energy_ratio)
        self.runtime_counts["refinement_backlog_high_watermark"]=max(
            self.runtime_counts["refinement_backlog_high_watermark"],len(self._refinement_pending))
        self._service_refinements()
    def _submit_ready_payloads(self):
        """Primary packets first; SIC-dependent packets only within the SIC allowance (Batch 10.4b)."""
        manager=self.track_manager
        sic_dependent=[]
        for track in [t for t in list(manager.active_tracks.values())
                      if getattr(t, "awaiting_primary_admission", False)]:
            self._admit_waiting(track)                    # admitted before: first claim on free slots (10.4l)
        for track in list(manager.active_tracks.values()):
            h=track.header
            if h is None or self.abs_sample < track.expected_end_sample:
                continue
            if getattr(track, "awaiting_primary_admission", False):
                continue                                  # waits for a slot instead of being rejected
            if getattr(track, "sic_origin", False):
                sic_dependent.append(track)
                continue
            track.status = "ready_for_payload"
            self._timed("payload_submission", self._submit_payload, track)
        for track in sorted(sic_dependent, key=lambda t: t.header.start):     # earliest deadline first
            if track.track_id not in manager.active_tracks:
                continue
            # the deadline applies whatever the admission state (Batch 10.4c)
            if self.abs_sample >= track.header.start+int(SIC_PAYLOAD_DEADLINE_S*FS):
                self.runtime_counts["sic_payload_expired"] += 1
                manager.retire(track, "expired", self.abs_sample, "sic_payload_deadline",
                               stage="payload_admission")
                continue
            track.status = "ready_for_payload"
            if self._timed("payload_submission", self._submit_sic_payload, track):
                self.runtime_counts["sic_payload_submitted"] += 1
            else:
                track.status = "pending_payload"
                self.runtime_counts["sic_payload_admission_deferred"] += 1

    def _consume_header_token(self):
        if self._header_tokens is not None:
            self._header_tokens -= 1
        self.runtime_counts["header_validations_attempted"]+=1

    def _service_refinements(self, max_attempts=None):
        """Do not let an early fallback replace a later baseline-valid PHDR.

        Wait until all three original PHDR validation opportunities have arrived.
        Preserve the canonical header of every already validated packet.
        """
        pending = collections.deque()
        attempted=0
        for c in self._refinement_pending:
            packet_start=c.header_start-c.physical_index*STAY_HDR
            if (self.abs_sample < packet_start+int(.95*FS) or self._header_tokens == 0
                    or (max_attempts is not None and attempted>=max_attempts)) :
                pending.append(c)
                continue
            awaiting_original=any(abs(other.header_start-other.physical_index*STAY_HDR-packet_start)<=8192
                                  for other in self.acquisition.pending)
            awaiting_original |= any(t.header is None and abs(t.detected_sample+8704-packet_start)<=8192
                                     for t in self.track_manager.active_tracks.values())
            if awaiting_original:
                pending.append(c)
                continue
            known=any(t.header is not None and abs(t.header.start-packet_start)<=4096
                      for t in self.track_manager._known())
            if known:
                self._acquisition_event("header_refinement_skipped", self.abs_sample,
                    candidate_id=c.candidate_id, reason="established_packet_preserved")
                continue
            base=max(0,c.header_start-35000);zz=self._timed("header_extraction", self.ring.get,base,c.header_start+290000)
            if zz is None:
                self._acquisition_event("header_candidate_rejected", self.abs_sample,
                    candidate_id=c.candidate_id, reason="refinement_iq_unavailable")
                continue
            spans=getattr(c,"retry_spans",(100,0))
            for position,span in enumerate(spans):
                if self._header_tokens == 0 or (max_attempts is not None and attempted>=max_attempts):
                    c.retry_spans=spans[position:]
                    pending.append(c)
                    break
                self._consume_header_token();attempted+=1
                h,detail=self._timed("local_header",decode_physical_header,zz,base,c,span,self.processing,self.batch_header,self.header_grids,
                    frequency_reuse=self.phdr_frequency_reuse, frequency_selection=getattr(c,"frequency_selection",None))
                selection=detail.pop("frequency_selection",None)
                if selection is not None:c.frequency_selection=selection
                self._acquisition_event("header_local_refinement",self.abs_sample,
                    candidate_id=c.candidate_id,candidate_sample=c.header_start,
                    inferred_packet_start=packet_start,physical_header=c.physical_index+1,
                    candidate_cfo_hz=c.frequency_hz-c.nominal_hz,span_hz=span,
                    valid=h is not None,refinement_delta_hz=detail["frequency_hz"]-c.frequency_hz,**detail)
                if h is not None:
                    self._accept_physical(h,c.physical_index,candidate_id=c.candidate_id,
                        source="candidate_local_retry",score=detail["sync_correlation"],energy=c.energy_ratio)
                    break
        self._refinement_pending=pending

    def _snapshot(self, stage, start, stop):
        if self.snapshot_copy_mode == "direct":
            return self._timed(stage+"_snapshot_direct", self.ring.snapshot_bytes, start, stop)
        z = self._timed(stage+"_extraction", self.ring.get, start, stop)
        if z is None:
            return None
        return self._timed(stage+"_snapshot_copy", z.tobytes), z.dtype.str

    def _payload_job(self, track):
        """(job_id, base, PayloadJob) for a ready track, or None after retiring it (IQ gone)."""
        manager = self.track_manager
        h = track.header
        job_id = self._job_id(track)
        # Earliest fragment filter input; headers are not needed by payload DSP.
        base = max(0, track.payload_start-2500)
        if self.payload_transport == "u8":
            raw = self._timed("payload_snapshot_u8", self.ring.raw_snapshot, base, track.expected_end_sample)
            snapshot = None if raw is None else (raw, U8_IQ_DTYPE)
        else:
            snapshot = self._snapshot("payload", base, track.expected_end_sample)
        if snapshot is None:
            manager.retire(track, "expired", self.abs_sample, "payload_iq_unavailable",
                           stage="payload_extraction")
            return None
        job = PayloadJob(job_id, self.receiver_id, track.track_id, h.start, base,
                         h.start-base, h.freq_hz, h.cost, tuple(sorted(h.info.items())),
                         snapshot[0], snapshot[1], self._other_transmissions(track, base),
                         track.expected_hops_hz, track.cfo_hz, self._payload_test_delay,
                         track.track_id in self._payload_test_fail_tracks,
                         self.payload_fec_backend, self.payload_native_library,
                         self.profile_payload_kernels, self.capture_fec_vectors,
                         soft_trials=self.dsp_soft_trials,
                         payload_retry=self._payload_retry_hypotheses(track))
        return job_id, base, job

    def _register_payload(self, track, job_id, base):
        h = track.header
        track.attempted_estimates = getattr(track, "attempted_estimates", set()) | {(h.start, h.freq_hz)}
        track.awaiting_primary_admission = False
        self._payload_pending[job_id] = (track, base)
        self.runtime_counts["payload_jobs_submitted"] += 1
        self.publication["payload_submitted"] += 1
        self._payload_enqueued_wall[job_id] = time.perf_counter()
        self.timings.add("payload_track_age_at_admission_rf", (self.abs_sample-h.start)/FS)
        self.track_manager.queue_payload(track, self.abs_sample, job_id)

    def _job_id(self, track):
        """Payload job identity; a later attempt for the same track gets its own id (10.4k)."""
        attempt = getattr(track, "payload_attempt", 0)
        return f"{self.receiver_id}:{track.track_id}:{track.header.start}" + (f":a{attempt}" if attempt else "")

    def _submit_payload(self, track):
        """A primary packet gets first claim on the primary payload pool."""
        manager = self.track_manager
        h = track.header
        job_id = self._job_id(track)
        if not self.payload_pool.can_admit():
            if getattr(self, "_exhaustive_file", False):
                # File-only backpressure preserves the track until its normal
                # payload job can be admitted; no validation rule changes.
                self.runtime_counts["file_payload_admission_waits"] += 1
                track.awaiting_primary_admission = True
                track.status = "pending_payload"
                return
            self.runtime_counts["payload_jobs_rejected"] += 1
            manager.retire(track, "queue_rejected", self.abs_sample, "processing_capacity",
                           stage="payload_admission", job_id=job_id)
            return
        built = self._payload_job(track)
        if built is None:
            return
        job_id, base, job = built
        try:
            admitted = self.payload_pool.submit(job)
        except Exception as exc:
            self.runtime_counts["payload_jobs_failed"] += 1
            manager.retire(track, "failed", self.abs_sample, "worker_start_failure",
                           stage="payload_admission", job_id=job_id, error=repr(exc))
            return
        if not admitted:
            raise RuntimeError("single-owner payload admission invariant violated")
        self._register_payload(track, job_id, base)

    def _submit_sic_payload(self, track):
        """SIC payload on an idle primary worker when available, else the SIC pool."""
        if self.sic is None or not self.sic.payload_slot_free():
            return False
        built = self._payload_job(track)
        if built is None:
            return True                       # retired (IQ gone): nothing left to wait for
        job_id, base, job = built
        primary_waiting = bool(self.payload_pool.waiting) or any(
            not getattr(t, "sic_origin", False) and t.header is not None
            and self.abs_sample >= t.expected_end_sample
            for t in self.track_manager.active_tracks.values())
        if not self.sic.submit_payload(job, primary_pool=None if primary_waiting else self.payload_pool):
            return False
        if self.sic.pending[job_id].get("primary_pool"):
            self._primary_sic_payload_jobs.add(job_id)
            self.runtime_counts["sic_payload_on_primary"] += 1
        self._sic_payload_jobs.add(job_id)
        self._register_payload(track, job_id, base)
        return True

    def _sic_payload_ready(self, track):
        return (getattr(track, "sic_origin", False) and track.header is not None
                and self.abs_sample >= track.expected_end_sample)

    def _retire_active_at_eof(self):
        """End of stream: incomplete tracks expire. With sic_eof="flush", a complete
        SIC-dependent packet still waiting for its payload allowance is kept for the
        drain (Batch 10.4c); with "skip" it is abandoned and counted."""
        flush = (self.sic is not None and self.sic_eof == "flush" and self.sic.disabled is None
                 and self.sic.payload_allowance > 0)
        for track in list(self.track_manager.active_tracks.values()):
            if (getattr(self, "_exhaustive_file", False) and
                    getattr(track, "awaiting_primary_admission", False) and
                    track.header is not None and self.abs_sample >= track.expected_end_sample):
                continue
            if self._sic_payload_ready(track):
                if flush:
                    continue
                self.runtime_counts["sic_payload_abandoned_at_eof"] += 1
            self.track_manager.retire(track, "expired", self.abs_sample, "end_of_stream")

    def _abandon_sic_payloads(self, reason, handoff=False):
        """SIC is closing: its in-flight payload jobs are abandoned optional work.

        Ledger: cancelled (so every submitted job stays accounted), but not the
        primary payload_jobs_cancelled gate; counted as sic_payload_abandoned.

        handoff=True (end of stream, SIC switched off; Batch 10.4k, review finding 1): a job
        whose track a primary header has confirmed is not optional work. It is withdrawn
        from the SIC pool and resubmitted on the primary payload pool under a new job id,
        with the primary estimate. The SIC job is closed in the ledger as handed off
        (cancelled), so exactly one result can be published for the track."""
        handed = []
        for job_id in list(self._sic_payload_jobs):
            if handoff and job_id in self._primary_sic_payload_jobs:
                continue  # already on primary: retain identity and normal drain/retry
            self._primary_sic_payload_jobs.discard(job_id)
            entry = self._payload_pending.pop(job_id, None)
            self._sic_payload_jobs.discard(job_id)
            self._payload_enqueued_wall.pop(job_id, None)
            self._payload_processing.discard(job_id)
            if entry is None:
                continue
            if handoff and not getattr(entry[0], "sic_origin", False):
                self.publication["cancelled"] += 1
                self.publication["sic_payload_handed_off"] += 1
                self.runtime_counts["sic_payload_handed_off"] += 1
                self._publication_samples.append(dict(kind="sic_payload_handed_off", job_id=job_id, reason=reason))
                handed.append(entry[0])
                continue
            self.publication["cancelled"] += 1
            self.runtime_counts["sic_payload_abandoned"] += 1
            self._publication_samples.append(dict(kind="sic_payload_abandoned", job_id=job_id, reason=reason))
            self.track_manager.retire(entry[0], "cancelled", self.abs_sample, reason,
                                      stage="payload_processing", job_id=job_id)
        # forwarded but not yet incorporated SIC payload messages belong to abandoned jobs
        self._payload_inbox = collections.deque(m for m in self._payload_inbox if m[1] in self._payload_pending)
        for track in handed:
            self._resubmit_on_primary(track, "sic_payload_handoff_" + reason)

    def _resubmit_on_primary(self, track, reason):
        """A queued track gets a new payload attempt on the primary pool (10.4k)."""
        self.track_manager.reopen_payload(track, self.abs_sample, reason)
        self._schedule_primary_attempt(track, reason)

    def _schedule_primary_attempt(self, track, reason):
        """An active track's next attempt on the primary pool, with its primary estimate.

        Batch 10.4l (audit 2026-10-05, F2): this packet was already admitted once, so a full
        primary pool does not reject it. It waits (``awaiting_primary_admission``) and is
        submitted first when a slot frees, until its IQ deadline (header start +
        SIC_PAYLOAD_DEADLINE_S, inside the ring); finish() keeps draining while it waits."""
        self._apply_primary_estimate(track, reason)
        track.payload_attempt = getattr(track, "payload_attempt", 0) + 1
        track.awaiting_primary_admission = True
        track.status = "pending_payload"
        self._admit_waiting(track)

    def _admit_waiting(self, track):
        """Submit a waiting track if its IQ is complete and a primary slot is free."""
        if self.abs_sample < track.expected_end_sample:
            return False
        if self.abs_sample >= track.header.start + int(SIC_PAYLOAD_DEADLINE_S*FS):
            self.runtime_counts["primary_handoff_expired"] += 1
            self.track_manager.retire(track, "expired", self.abs_sample, "primary_admission_deadline",
                                      stage="payload_admission")
            return False
        if not self.payload_pool.can_admit():
            self.runtime_counts["primary_handoff_waits"] += 1
            return False
        track.status = "ready_for_payload"
        self._timed("payload_submission", self._submit_payload, track)
        return True

    def _admission_waiting(self):
        """Tracks admitted before that wait for a primary slot with complete IQ (10.4l)."""
        return [t for t in self.track_manager.active_tracks.values()
                if getattr(t, "awaiting_primary_admission", False) and self.abs_sample >= t.expected_end_sample]

    def _retry_with_primary_estimate(self, track, job_id):
        """Batch 10.4k (review finding 2): a SIC attempt failed, but a primary header has since
        confirmed the packet with a different estimate. Once per track, the failed result is
        superseded (worker completed, no packet published) and the packet is decoded again on
        the primary pool with the primary estimate; that attempt's result is final."""
        if (getattr(track, "sic_origin", False) or self._finished or self.payload_pool.closed
                or not self._has_new_estimate(track)):
            return False
        self.publication["worker_completed_no_packet"] += 1
        self.publication["superseded_by_primary_retry"] += 1
        self.runtime_counts["sic_failed_primary_retries"] += 1
        self._publication_samples.append(dict(kind="superseded_by_primary_retry", job_id=job_id))
        self._resubmit_on_primary(track, "sic_attempt_failed_primary_estimate")
        return True

    def _payload_retry_hypotheses(self, track):
        """Retry offsets for this track's payload job (Batch 10.2)."""
        # Exhaustive files may spend the existing bounded retry work after a
        # failed primary decode. Do not alter the selected profile or live/default
        # file policy; the worker still requires PHY and application CRC passes.
        mode = "grid" if getattr(self, "_exhaustive_file", False) else self.payload_retry
        if mode == "off":
            return ()
        h = track.header
        alternates = sorted(((obs.packet_start-h.start, obs.cfo_hz-track.cfo_hz)
                             for obs in track.headers.values()),
                            key=lambda a: (abs(a[0]), abs(a[1])))
        mode = "grid" if mode == "adaptive" else mode
        return payload_retry_hypotheses(mode, alternates)

    @_publication_boundary
    def service_file_backlog(self):
        """Explicit file-only owner service; never advance RF or retry eligibility.

        Renew the per-call CPU admission budget while the reader waits. The
        ordinary live idle API intentionally does not renew that budget.
        """
        if not getattr(self, "_exhaustive_file", False):
            raise RuntimeError("Exhaustive file service was not enabled")
        if self._finished:
            return []
        self.note_input_idle()
        self._reset_header_budget("exhaustive_file_wait")
        return self._timed("exhaustive_file_service", self._service)

    @_publication_boundary
    def service_idle(self):
        """Owner-only progress while the RF source has no new bytes ready.

        Input waiting is not worker waiting. Service only queued completions;
        do not invent RF samples, advance a clock, reset validation budgets or
        change retry eligibility. No background thread mutates receiver state.
        """
        if self._finished or not self.header_idle_service:
            return []
        pools=[self.payload_pool]
        if self.header_pipeline is not None:pools.append(self.header_pipeline.pool.pool)
        ready=any(not slot['outbox'].empty() for pool in pools for slot in pool.slots)
        ready=ready or (self.acquisition_process is not None and not self.acquisition_process.outbox.empty())
        return self._timed('idle_service',self._service) if ready else []

    @_publication_boundary
    def poll(self):
        """Collect ready worker results without waiting; call only from owner thread."""
        decoded = self._drain_acquisition()
        manager = self.track_manager
        # Messages leave the inbox one at a time: if incorporating one raises,
        # the rest stay owner-held for the next poll instead of being dropped.
        self._payload_inbox.extend(self.payload_pool.poll())
        while self._payload_inbox:
            # Change 11 (Audit 02 A5): the diagnostics observer runs while the
            # message is still owner-held in the inbox. If it raises, the result
            # is retried by the next poll (or reported as stranded by close()).
            kind, job_id, result = self._payload_inbox[0]
            if self.payload_pool.observer is not None:
                self.payload_pool.observer("result_incorporation",job_id,dict(wall=time.perf_counter(),kind=kind))
            if kind == "deferred":
                self._payload_inbox.popleft()
                self._payload_processing.discard(job_id)
                self._payload_pending[job_id][0].status = "payload_retry_pending"
                continue
            if job_id in self._primary_sic_payload_jobs and self.sic is not None:
                self.sic.collect_primary_payload(kind, job_id, result)
            self._payload_inbox.popleft()
            track, base = self._payload_pending[job_id]
            common = dict(track_id=track.track_id, job_id=job_id,
                          packet_start=track.header.start, header=track.header.info.copy())
            if kind == "started":
                self.timings.add("payload_queue_wait", result["worker_wall"]-self._payload_enqueued_wall[job_id])
                self.timings.add("payload_track_age_at_start_rf", (self.abs_sample-track.header.start)/FS)
                self._payload_processing.add(job_id)
                track.status = "payload_processing"
                self.runtime_counts["payload_jobs_started"] += 1
                self.payload_counts["payload_decode_attempts"] += 1
                self.runtime_counts["payload_max_workers_processing"] = max(
                    self.runtime_counts["payload_max_workers_processing"], len(self._payload_processing))
                self._acquisition_event("payload_worker_started", self.abs_sample, **common, **result)
                continue
            self._payload_processing.discard(job_id)
            del self._payload_pending[job_id]
            sic_job = job_id in self._sic_payload_jobs
            self._sic_payload_jobs.discard(job_id)
            self._primary_sic_payload_jobs.discard(job_id)
            enqueued = self._payload_enqueued_wall.pop(job_id)
            self.timings.add("payload_job", result.get("wall_s", 0.))
            self.timings.add("payload_admission_to_completion", result.get("worker_wall",time.perf_counter())-enqueued)
            self.timings.add("payload_track_age_at_completion_rf", (self.abs_sample-track.header.start)/FS)
            self.processing["payload_wall_s"] += result.get("wall_s", 0.)
            self.processing["payload_cpu_s"] += result.get("cpu_s", 0.)
            self.processing['payload_deserialize_wall_s'] += result.get('deserialize_wall_s',0.)
            self.processing['payload_deserialize_cpu_s'] += result.get('deserialize_cpu_s',0.)
            self.processing["payload_calls"] += 1
            self.processing["payload_max_s"] = max(self.processing["payload_max_s"], result.get("wall_s", 0.))
            if kind == "failed":
                self.runtime_counts["payload_jobs_failed"] += 1
                self.publication["worker_failed"] += 1
                self._acquisition_event("payload_worker_failed", self.abs_sample, **common, **result)
                manager.retire(track, "failed", self.abs_sample, "payload_worker_exception",
                               stage="payload_processing", job_id=job_id)
                continue
            self.runtime_counts["payload_jobs_completed"] += 1
            self.publication["worker_completed"] += 1
            self._acquisition_event("payload_worker_completed", self.abs_sample, **common,
                                    worker_wall=result["worker_wall"], wall_s=result["wall_s"])
            pkt, detail = result["packet"], result["detail"]
            for name, counters in detail.get("kernel_profile",{}).get("kernels",{}).items():
                for field,value in counters.items():
                    self.processing["kernel_payload_"+name+"_"+field] += value
            detail.update(job_id=job_id, receiver_id=self.receiver_id, track_id=track.track_id)
            for field in ("all_fragment_windows", "soft_stream_produced", "viterbi_output_produced"):
                self.payload_counts[field] += int(detail.get(field, False))
            self.payload_counts["weighted_fec_attempts"] += sum(
                a["method"] == "noise_weighted" for a in detail.get("fec_attempts", []))
            self.payload_counts["weighted_crc_rescues"] += int(
                detail.get("selected_method") == "noise_weighted" and detail.get("phy_crc_ok", False))
            for stage in ("fragment_extraction", "fec_crc", "payload_import", "payload_array_reconstruction"):
                for clock in ("wall_s", "cpu_s"):
                    self.processing[stage+"_"+clock] += detail.get(stage+"_"+clock, 0.)
            for f in detail.get("fragments", []):
                f["expected_start_sample"] += base
            self._acquisition_event("payload_result", self.abs_sample,
                                    packet_start=track.header.start, **detail)
            if ((sic_job or getattr(track, "sic_discovered", False)) and (pkt is None or not pkt.crc_ok)
                    and self._retry_with_primary_estimate(track, job_id)):
                continue
            if pkt is None:
                self.publication["worker_completed_no_packet"] += 1
                manager.retire(track, "rejected", self.abs_sample, "payload_decode_failed",
                               stage=detail.get("stage", "payload_extraction_or_trellis"), job_id=job_id)
                continue
            if not pkt.crc_ok and getattr(track, "sic_origin", False):
                # Batch 10.4: a header found only after cancellation is published
                # only when its payload confirms it (PHY CRC-16).
                self.publication["worker_completed_no_packet"] += 1
                self.publication["sic_crc_failed_not_published"] += 1
                manager.retire(track, "rejected", self.abs_sample, "sic_payload_crc_failed",
                               stage="phy_crc", job_id=job_id)
                continue
            pkt.header.start += base
            pkt.payload_start += base
            for f in pkt.fragments:
                f.expected_start_sample += base
            if getattr(track, "failed_job_published", None) is not None and pkt.crc_ok:
                pkt.payload_diagnostics["recovers_failed_job"] = track.failed_job_published
            if getattr(track, "sic_discovered", False) and not pkt.crc_ok:
                track.failed_job_published = job_id
            decoded.append(pkt)
            self._inflight_packets.append((job_id, pkt))
            self.publication["collected_packets"] += 1
            self.publication["collected_crc_ok"] += int(bool(pkt.crc_ok))
            manager.retire(track, "completed", self.abs_sample,
                           "payload_decoded" if pkt.crc_ok else "phy_crc_failed",
                           stage="complete" if pkt.crc_ok else "phy_crc", job_id=job_id,
                           crc_ok=pkt.crc_ok, payload_hex=pkt.payload.hex(),
                           payload_viterbi_metric=pkt.viterbi_cost)
            if pkt.crc_ok and self.sic is not None:
                self.sic.on_packet(track)
        if self.header_pipeline is not None:
            self.header_pipeline.collect()
            self._submit_ready_payloads()
        if self.sic is not None:
            self._timed("sic_collect", self.sic.collect)
            self._timed("sic_service", self.sic.service)
            self._submit_ready_payloads()
        if self.payload_retry == "adaptive":
            self.payload_pool.service_retries()
        return decoded

    @_publication_boundary
    def finish(self, *, stop_event=None):
        """Process the actual final samples and expire incomplete tracks at EOF.

        Never pad the IQ with artificial samples to force completion. A second
        call returns no packets, so callers cannot accidentally emit twice.
        """
        if self._finished:
            return []
        # Change 11 (Audit 02 A4): an incomplete trailing I/Q byte pair no
        # longer discards the complete samples and admitted work before it.
        # They are decoded and drained normally; the truncation is then raised,
        # and the packets are held for take_unreturned_packets().
        truncated = len(self._raw_pending) % 2
        if truncated:
            del self._raw_pending[-1]
        decoded = self._finish(stop_event)
        if truncated:
            self.publication["truncated_trailing_bytes"] += 1
            raise TruncatedIQError("Truncated IQ: an unsigned I/Q byte pair is incomplete (1 trailing byte ignored); "
                             "every complete sample was decoded and its packets are held for take_unreturned_packets()")
        return decoded

    def _finish(self, stop_event):
        decoded=[]
        if self._raw_pending:
            decoded.extend(self._feed_block(bytes(self._raw_pending)))
            self._raw_pending.clear()
        deadline = time.monotonic()+self.payload_finish_timeout
        while self.acquisition_process is not None and self.acquisition_process.outstanding:
            if stop_event is not None and stop_event.is_set():
                self.finish_reason = "receiver_stopped"
                self.close(reason="receiver_stopped")
                return decoded
            if time.monotonic() >= deadline:
                self.finish_reason = "acquisition_finalization_timeout"
                self.close(reason="acquisition_finalization_timeout")
                raise TimeoutError("Acquisition finalization timeout")
            decoded.extend(self._drain_acquisition(wait=True))
        if self.header_pipeline is not None:
            # No later sample can confirm a parked replica's packet: validate it,
            # and park nothing more while finalizing (Change 11).
            self.header_pipeline.end_of_stream()
        drain_error, failures = None, 0
        while time.monotonic()<deadline and not (stop_event is not None and stop_event.is_set()):
            self._reset_header_budget("eof")
            before=self.runtime_counts["header_validations_attempted"]
            try:
                decoded.extend(self._service())
                failures = 0
            except Exception as exc:
                # Batch 10.4k (review finding 6): a failing diagnostics hook while header
                # results are collected leaves them owner-held; keep finalizing and re-raise
                # the first error at the end (a hook failing 3 times in a row stops this loop).
                drain_error = drain_error or exc
                failures += 1
                if failures >= 3:
                    break
                continue
            if self.runtime_counts["header_validations_attempted"]==before:
                if self.header_pipeline is not None and (self.header_pipeline.pending or self.header_pipeline.released):
                    time.sleep(.001)
                else:
                    break
        self.acquisition.finish(self.abs_sample)
        self._retire_active_at_eof()
        failures = 0
        if self.sic is not None:
            if self.sic_eof == "skip":
                self._abandon_sic_payloads("end_of_stream", handoff=True)
                self.sic.close("end_of_stream")
            self.sic.eof = True
        def sic_waiting():
            # a waiting SIC packet keeps the drain alive only if it can ever be admitted;
            # an already admitted packet waiting for a primary slot always does (10.4l)
            return bool(self._admission_waiting()) or (self.sic is not None and not self.sic.pool.closed and (self.sic.busy() or (
                self.sic.payload_allowance > 0 and any(
                    self._sic_payload_ready(t) for t in self.track_manager.active_tracks.values()))))
        unfinished = False
        try:
            while ((self._payload_pending or sic_waiting())
                   and time.monotonic() < deadline
                   and not (stop_event is not None and stop_event.is_set())):
                try:
                    decoded.extend(self._service() if getattr(self, "_exhaustive_file", False) else self.poll())
                    failures = 0
                except Exception as exc:
                    # Change 11 (Audit 02 A5): a failing diagnostics hook must not
                    # make finish() close over results still owner-held in the
                    # inbox. Keep draining; re-raise the first error afterwards.
                    # A hook that keeps failing stops the drain after 3 tries.
                    drain_error = drain_error or exc
                    failures += 1
                    if failures >= 3:
                        break
                if self._payload_pending or sic_waiting():
                    time.sleep(.005)
            # record the drain outcome BEFORE cleanup retires what was still waiting
            unfinished = bool(self._payload_pending or sic_waiting())
            if self.sic is not None:
                # SIC-found packets whose IQ ended with the stream cannot complete.
                for track in list(self.track_manager.active_tracks.values()):
                    self.track_manager.retire(track, "expired", self.abs_sample, "end_of_stream")
        finally:
            if self._payload_pending:
                self.publication["finish_undrained_jobs"] += len(self._payload_pending)
            sic_busy = unfinished or sic_waiting()
            # Batch 10.4k: frontends read finish_reason to tell a complete drain from a cut one.
            self.finish_reason = (("receiver_stopped" if stop_event is not None and stop_event.is_set()
                                   else "finalization_timeout") if (self._payload_pending or sic_busy or (self.header_pipeline is not None and (self.header_pipeline.pending or self.header_pipeline.released or self.header_pipeline.parked))) else "finished")
            self.close(reason=self.finish_reason)
        if drain_error is not None:
            raise drain_error
        return decoded

    def close(self, reason="receiver_stopped"):
        """Abort uncollected work; queued/active jobs are explicit cancellations.

        EOF normally drains via finish(); stop never waits for expensive DSP.
        Already published results are never emitted again.
        """
        # Field readiness (Audit 02 B6, decoder part): every resource is released even
        # when an earlier step raises, so a failed acquisition close can no longer leave
        # header, SIC or payload worker processes running. The first error is re-raised
        # after the whole close has run.
        close_errors = []
        def _step(function, *args, **kwargs):
            try:
                function(*args, **kwargs)
            except Exception as exc:
                close_errors.append(exc)
                self.runtime_counts["close_step_errors"] += 1
        if self.acquisition_process is not None:
            _step(self.acquisition_process.close)
        if self.header_pipeline is not None:
            _step(self.header_pipeline.close, reason)
        if self.sic is not None:
            _step(self._abandon_sic_payloads, reason)
            _step(self.sic.close, reason)
        discarded_before = sum(v for k, v in self.payload_pool.shutdown_discarded.items() if k != "completed_crc_ok")
        _step(self.payload_pool.close, graceful=(reason == "finished"))
        discarded = sum(v for k, v in self.payload_pool.shutdown_discarded.items() if k != "completed_crc_ok")-discarded_before
        if discarded:
            self.publication["stranded_results"] += discarded
            self._publication_samples.append(dict(kind="shutdown_discarded_worker_results",
                count=discarded, reason=reason))
        for kind, job_id, result in self._payload_inbox:
            if kind != "started":
                packet = result.get("packet") if isinstance(result, dict) else None
                self.publication["stranded_results"] += 1
                self._publication_samples.append(dict(kind="unincorporated_worker_result",
                    job_id=job_id, result_kind=kind, crc_ok=bool(getattr(packet, "crc_ok", False)), reason=reason))
        self._payload_inbox.clear()
        for job_id, (track, _) in list(self._payload_pending.items()):
            self.runtime_counts["payload_jobs_cancelled"] += 1
            self.publication["cancelled"] += 1
            self._publication_samples.append(dict(kind="cancelled", job_id=job_id, reason=reason))
            self.track_manager.retire(track, "cancelled", self.abs_sample, reason,
                                      stage="payload_processing", job_id=job_id)
        self._payload_pending.clear()
        self._sic_payload_jobs.clear()
        self._primary_sic_payload_jobs.clear()
        self._payload_enqueued_wall.clear()
        self._payload_processing.clear()
        for track in list(self.track_manager.active_tracks.values()):
            self.track_manager.retire(track, "expired", self.abs_sample, reason)
        self.acquisition.finish(self.abs_sample)
        for c in self._refinement_pending:
            self._acquisition_event("header_refinement_cancelled",self.abs_sample,
                candidate_id=c.candidate_id,candidate_sample=c.header_start,reason=reason)
        self._refinement_pending.clear()
        self.ring.clear()
        if self.header_grids is not None:
            self.header_grids.clear()
        self.ring.retention_floor=None
        self._raw_pending.clear()
        self.acquisition.history=[np.empty(0,complex) for _ in self.acquisition.history]
        self.acquisition._search_grids.clear()
        self.acquisition.recent.clear()
        self.acquisition._recent_bins.clear()
        self.acquisition._mixers=[]
        self.acquisition._mixer_length=0
        self._finished = True
        if close_errors:
            raise close_errors[0]

    def _account_returned(self, out):
        """Close one outermost public call; see _publication_boundary."""
        returned = [p for _, p in self._recovered_packets]+list(out)
        if self._recovered_packets:
            self.publication["recovered_packets_returned"] += len(self._recovered_packets)
            self._recovered_packets = []
        present = {id(p) for p in returned}
        for job_id, pkt in self._inflight_packets:
            if id(pkt) not in present:
                self.publication["collected_missing_from_return"] += 1
                self._publication_samples.append(dict(kind="collected_missing_from_return", job_id=job_id))
                returned.append(pkt)
        self._inflight_packets = []
        self.publication["returned_packets"] += len(returned)
        self.publication["returned_crc_ok"] += sum(bool(p.crc_ok) for p in returned)
        return returned

    def take_unreturned_packets(self):
        """Packets collected by a public call that raised; returns and accounts them."""
        if self._api_depth:
            raise RuntimeError("take_unreturned_packets is an owner-level call")
        return self._account_returned([])

    def publication_invariants(self):
        """Batch 9.10 worker-to-publication accounting (see handoff section 14).

        submitted -> worker completed/failed | cancelled | outstanding
        worker completed -> collected packet | legitimate no-packet result
        collected packet -> returned (public API) | returned by a direct internal
                            call | still held for return after an exception
        collected packet == TrackManager completed track
        """
        c = self.publication
        outstanding = len(self._payload_pending)
        held = len(self._recovered_packets)+len(self._inflight_packets)
        tracks = self.track_manager.counts
        checks = dict(
            jobs_accounted=c["payload_submitted"] == c["worker_completed"]+c["worker_failed"]+c["cancelled"]+outstanding,
            results_collected=c["worker_completed"] == c["collected_packets"]+c["worker_completed_no_packet"],
            packets_returned=c["collected_packets"] == c["returned_packets"]+c["returned_via_internal_call"]+held,
            tracks_match_collected=tracks["packets_completed"] == c["collected_packets"],
            none_missing_from_return=c["collected_missing_from_return"] == 0,
            none_recovered_after_exception=c["packets_recovered_after_exception"] == 0,
            none_stranded=c["stranded_results"] == 0,
        )
        return dict(passed=all(checks.values()), checks=checks, counts=dict(c),
                    outstanding_jobs=outstanding, held_unreturned_packets=held,
                    inbox_messages=len(self._payload_inbox), samples=list(self._publication_samples))

    def receiver_stage_stats(self, *, selection_sizes=False, tracks=None):
        """Stage counters for the dashboard. Selection sizing is an explicit diagnostic."""
        if tracks is None:
            tracks = self.track_manager.stats()
        stages = {"acquisition_candidates": self._energy_candidates+self.acquisition.counts["candidates_queued"],
                  "valid_header_observations": tracks["physical_headers_validated"],
                  **{key: tracks[key] for key in ("tracks_created", "active_tracks", "packets_completed",
                      "tracks_expired", "tracks_rejected", "phy_crc_pass", "phy_crc_fail")},
                  **{key: self.payload_counts[key] for key in ("payload_decode_attempts",
                      "all_fragment_windows", "soft_stream_produced", "viterbi_output_produced",
                      "weighted_fec_attempts", "weighted_crc_rescues")}}
        stages.update(self.payload_pool.stats())
        if self.header_pipeline is not None:
            stages.update(self.header_pipeline.pool.stats())
            stages["header_oscillator_cache"] = self.header_pipeline.oscillator_stats()
        stages.update({key: value for key,value in self.runtime_counts.items()
                       if key.startswith(("header_jobs_","header_candidates_","phdr_"))})
        stages.update({key: self.runtime_counts[key] for key in (
            "payload_jobs_submitted", "payload_jobs_started", "payload_jobs_completed",
            "payload_jobs_failed", "payload_jobs_rejected", "payload_jobs_cancelled",
            "payload_max_workers_processing")})
        for key in ("max_ring_iq_bytes","refinement_backlog_high_watermark","validations_per_feed_max"):
            stages[key]=self.runtime_counts[key]
        if self.header_grids is not None:
            stages["header_grid_cache"] = (self.header_pipeline.grid_stats()
                if self.header_pipeline is not None else dict(self.header_grids.stats(),scope='owner'))
        stages["header_validations_attempted"] = self.runtime_counts["header_validations_attempted"]
        stages["validation_backlog_depth"] = len(self.acquisition.pending)
        stages["refinement_backlog_depth"] = len(self._refinement_pending)
        stages["retained_ring_iq_bytes"] = sum(z.nbytes for _,z in self.ring.chunks)
        stages["retained_ring_raw_bytes"] = sum(len(r) for _,r in self.ring.raw_chunks)
        stages["payload_transport"] = self.payload_transport
        stages["dsp_soft_trials"] = self.dsp_soft_trials
        stages["payload_retry"] = self.payload_retry
        stages["phdr_sic"] = self.phdr_sic
        stages["sic"] = self.sic.stats() if self.sic is not None else None
        for key in ("sic_payload_admission_deferred", "sic_payload_submitted", "sic_payload_expired",
                    "sic_payload_abandoned", "sic_payload_abandoned_at_eof"):
            stages[key] = self.runtime_counts[key]
        stages["payload_jobs_processing"] = len(self._payload_processing)
        stages["payload_jobs_outstanding"] = len(self._payload_pending)
        candidates = list(self.acquisition.recent)+list(self.acquisition.pending)+list(self._refinement_pending)
        if self.header_pipeline:
            candidates += [v['candidate'] for v in self.header_pipeline.pending.values() if v['candidate'] is not None]
        selections = {id(c.frequency_selection):c.frequency_selection for c in candidates if hasattr(c,'frequency_selection')}
        stages.update(frequency_selection_count=len(selections),
                      frequency_selection_bound=2048+2*self.acquisition.MAX_PENDING+
                      (self.header_pipeline.pool.outstanding_capacity if self.header_pipeline else 0))
        if selection_sizes:
            stages["frequency_selection_repr_bytes"] = sum(len(repr(v)) for v in selections.values())
        return stages

    def stats(self):
        tracks = self.track_manager.stats()
        stages = self.receiver_stage_stats(selection_sizes=True, tracks=tracks)
        return {**tracks, "timings": self.timings.summary(), "processed_samples": self.abs_sample,
                "buffered_input_bytes": len(self._raw_pending),
                "input_samples": self.input_abs_sample,
                "acquisition_process": self.acquisition_process.stats() if self.acquisition_process else None,
                "acquisition": dict(self.acquisition.counts),
                "acquisition_noise": dict(mode=self.acquisition.noise_mode,
                    channel_estimates=list(self.acquisition.noise),
                    alternate_energy_threshold_references=list(self.acquisition.energy_references
                        if self.acquisition.energy_reference == "legacy" and self.acquisition.noise_mode in ("guard20","spectral20","spectral50")
                        else self.acquisition.noise),
                    alternate_energy_reference_mode=self.acquisition.energy_reference,
                    legacy_energy_threshold_reference=self.noise),
                "payload_fec_backend": self.payload_fec_backend,
                "publication": self.publication_invariants(),
                "processing": {**dict(self.processing), **dict(self.acquisition.processing)}, "receiver_stages": stages}

def show_packet(pkt: Packet, center_hz: int, t0: float | None=None):
    t = pkt.header.start / FS
    stamp = time.strftime("%H:%M:%S") if t0 is None else f"{t:8.3f}s"

    # Only publish decoded application data after the LR-FHSS PHY CRC passes.
    if pkt.crc_ok:
        print()
        print_payload(
            pkt.payload,
            rf_meta={
                "timestamp": stamp,
            },
        )
        print(f"PHY:           LR-FHSS CRC PASS | {pkt.fragment_count} fragments")
        sys.stdout.flush()
        return

    # Keep failed PHY packets visible for receiver debugging, but do not
    # interpret their bytes as telemetry.
    hx = " ".join(f"{b:02X}" for b in pkt.payload)
    print(f"\n[{stamp}] LR-FHSS CRC FAIL | {len(pkt.payload)} B | {pkt.fragment_count} fragments")
    print(f"  HEX:   {hx}")

def read_exact_stream(stream, n):
    buf = bytearray()
    while len(buf) < n:
        x = stream.read(n-len(buf))
        if not x:
            break
        buf.extend(x)
    return bytes(buf)

class SourceReadError(OSError):
    """A source read failed after ``partial`` bytes of the block had been received
    (Batch 10.4l, audit 2026-10-05 F3): the caller decodes them before finishing."""
    def __init__(self, cause: BaseException, partial: bytes):
        super().__init__(f"{cause!r} after {len(partial)} received bytes")
        self.cause, self.partial = cause, partial


def recv_exact_serviced(sock: socket.socket, n: int, idle) -> bytes:
    return recv_block(sock, n, idle)

def recv_block(sock: socket.socket, n: int, idle=None, on_wait=None) -> bytes:
    """Up to n bytes (fewer at end of stream); ``idle`` runs while no byte is ready.
    A read error raises SourceReadError with the bytes already read (10.4l).
    ``on_wait`` (the no-data watchdog) runs whenever the wait for data continues:
    between idle services, and on each socket timeout, which is not an error."""
    import select
    buf = bytearray()
    while len(buf) < n:
        try:
            if idle is not None and not select.select([sock],[],[],.002)[0]:
                idle()
                if on_wait is not None:
                    on_wait()
                continue
            x = sock.recv(n-len(buf))
        except socket.timeout as exc:
            if on_wait is None:
                raise SourceReadError(exc, bytes(buf)) from exc    # unchanged behaviour
            on_wait()
            continue
        except OSError as exc:
            raise SourceReadError(exc, bytes(buf)) from exc
        if not x:
            break
        buf.extend(x)
    return bytes(buf)

def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        x = sock.recv(n-len(buf))
        if not x:
            break
        buf.extend(x)
    return bytes(buf)

def call_showing_unreturned(dec, call, show):
    """Run a packet-returning decoder call and show its packets (Audit 02 B5).

    If the call raises (a decoder error, or Ctrl+C), the packets it had already
    collected are fetched with take_unreturned_packets() and shown before the
    exception propagates, so an error path never silently drops decoded packets.
    """
    try:
        packets = call()
    except BaseException:
        try:
            recovered = dec.take_unreturned_packets()
        except Exception:
            recovered = []
        for pkt in recovered:
            show(pkt)
        raise
    for pkt in packets:
        show(pkt)
    return packets

def run_file(path: str, trigger_db: float, verbose: bool, *, exhaustive_file=False, **payload_options):
    dec = LiveDecoder(trigger_db, verbose, **payload_options)
    backpressure = None
    if exhaustive_file:
        import json
        from lrfhss_exhaustive_file import ExhaustiveFile
        backpressure = ExhaustiveFile(dec, warn=lambda text: print("WARNING: " + text, file=sys.stderr))
    count = 0
    try:
        with open(path, "rb") as f:
            def show_file_packet(pkt):
                nonlocal count
                show_packet(pkt, DEFAULT_CENTER_HZ, t0=0)
                count += int(pkt.crc_ok)
            while raw := f.read(100_000):
                if backpressure is not None:
                    backpressure.before_feed(lambda: call_showing_unreturned(
                        dec, dec.service_file_backlog, show_file_packet), next_samples=len(raw)//2)
                call_showing_unreturned(dec, lambda: dec.feed_raw(raw), show_file_packet)
        try:
            final = backpressure.finish() if backpressure is not None else dec.finish()
        except TruncatedIQError as exc:
            # Change 11/12: a truncated trailing byte pair is reported, but the
            # packets decoded from every complete sample are still shown.
            print(f"WARNING: {exc}")
            final = dec.take_unreturned_packets()
        except Exception:
            # Any other failure keeps its error outcome; the packets already
            # decoded are still shown first (Change 12, second review).
            for pkt in dec.take_unreturned_packets():
                show_packet(pkt, DEFAULT_CENTER_HZ, t0=0)
            raise
        for pkt in final:
            show_packet(pkt, DEFAULT_CENTER_HZ, t0=0)
            count += int(pkt.crc_ok)
        stages = dec.stats()["receiver_stages"]
        print(f"\nDecoded {count} CRC-valid packet(s).")
        if backpressure is not None:
            print("Exhaustive file: " + json.dumps(backpressure.snapshot(1.0), sort_keys=True))
        print("Payload processing:", {k: stages[k] for k in (
            "payload_jobs_submitted", "payload_jobs_completed", "payload_jobs_failed",
            "payload_jobs_rejected", "payload_jobs_cancelled", "payload_queue_high_watermark")})
        capacity_failure = any(stages[k] for k in (
            "payload_jobs_failed", "payload_jobs_rejected", "payload_jobs_cancelled"))
        return 3 if capacity_failure else 0 if count else 2
    finally:
        dec.close()


def _stop_rtl_tcp(proc):
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def run_live(rtl_tcp: str, freq: int, gain: float, ppm: int, port: int,
             trigger_db: float, verbose: bool, record_iq=None, **payload_options):
    exe = shutil.which(rtl_tcp) or (rtl_tcp if os.path.isfile(rtl_tcp) else None)
    if not exe:
        print(f"Could not find {rtl_tcp!r}. Put rtl_tcp.exe in PATH or use --rtl-tcp FULL_PATH.",
              file=sys.stderr)
        return 2

    # rtl_tcp avoids Windows stdout/text-mode issues and gives us a clean binary IQ socket.
    cmd = [
        exe, "-a", "127.0.0.1", "-p", str(port),
        "-f", str(freq), "-s", str(FS),
        "-g", str(gain), "-P", str(ppm)
    ]
    print("Starting RTL-SDR server:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=None)

    sock = None
    dec = None
    recorder = None
    try:
        deadline = time.time() + 6.0
        while time.time() < deadline:
            if proc.poll() is not None:
                print("rtl_tcp exited before the decoder could connect.", file=sys.stderr)
                return 2
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=0.5)
                break
            except OSError:
                time.sleep(0.15)
        if sock is None:
            print(f"Could not connect to rtl_tcp on port {port}.", file=sys.stderr)
            return 2
        sock.settimeout(2.0)
        info = recv_exact(sock, 12)
        if len(info) != 12 or info[:4] != b"RTL0":
            print("Did not receive a valid rtl_tcp dongle header.", file=sys.stderr)
            return 2
        # A finite timeout lets the no-data watchdog run while the dongle is silent.
        sock.settimeout(0.5)
        print("Connected. Listening for LR-FHSS... Ctrl+C to stop.")
        print("Profile: 915 MHz / 722.66 kHz / CR 2/3 / narrow grid / 3 headers / hopseq 0x100")
        dec = LiveDecoder(trigger_db, verbose, **payload_options)
        monitor = SourceMonitor()
        if record_iq:
            # Raw IQ goes to disk unchanged so the flight can be decoded again later.
            path = Path(record_iq)
            if path.is_dir() or not path.suffix:
                path = default_recording_path(path, freq, gain)
            recorder = IQRecorder(path, metadata=dict(center_hz=freq, gain_db=gain, ppm=ppm,
                                                      source="rtl_tcp", receiver="lrfhss_live CLI"))
            print(f"Recording raw IQ to {recorder.path}")
        def warn(messages):
            for text in messages:
                print(f"WARNING: {text}", file=sys.stderr, flush=True)
        def idle():
            dec.note_input_idle()
            call_showing_unreturned(dec, dec.service_idle, lambda pkt: show_packet(pkt, freq))
        while True:
            try:
                raw = recv_block(sock, 100_000, idle if dec.header_idle_service else None,
                                 on_wait=lambda: warn(monitor.check()))
                warn(monitor.note_bytes(raw))
                if recorder is not None:
                    recorder.write(raw)
            except OSError as exc:
                # the source failed: every byte received is still decoded (10.4k), including
                # the part of the block read before the failure (10.4l)
                print(f"RTL-SDR stream failed: {exc}", file=sys.stderr)
                partial = getattr(exc, "partial", b"")
                if partial and recorder is not None:
                    recorder.write(partial)
                if partial:
                    try:
                        for pkt in dec.feed_raw(partial):
                            show_packet(pkt, freq)
                    except Exception as feed_exc:
                        for pkt in dec.take_unreturned_packets():
                            show_packet(pkt, freq)
                        print(f"ERROR: decoding the last received bytes failed: {feed_exc!r}", file=sys.stderr)
                finish_live_stream(dec, freq)
                return 3
            if not raw:
                print("RTL-SDR stream ended.")
                # Batch 10.4k (review 2026-10-02, finding 3): a natural end of stream
                # decodes every received sample and admitted job (bounded); only
                # Ctrl+C stays an immediate abort.
                return finish_live_stream(dec, freq)
            call_showing_unreturned(dec, lambda: dec.feed_raw(raw), lambda pkt: show_packet(pkt, freq))
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            if dec is not None:
                dec.close()
        finally:
            if sock is not None:
                sock.close()
            _stop_rtl_tcp(proc)
            if recorder is not None:
                rec = recorder.close()
                print(f"Raw IQ: {rec['bytes_written']/1e6:.1f} MB written to {rec['path']}"
                      + (f"; {rec['bytes_dropped']} bytes dropped in {rec['gaps']} gap(s)" if rec['bytes_dropped'] else "")
                      + (f"; recording stopped early ({rec['stopped_reason']})" if rec['stopped_reason'] else ""),
                      file=sys.stderr if (rec['bytes_dropped'] or rec['stopped_reason']) else sys.stdout)
    return 0


def finish_live_stream(dec, freq):
    """End of a live stream: drain through finish(), show every packet, report the outcome.

    0 = complete drain; 1 = the stream ended inside an I/Q byte pair (every complete
    sample decoded); 3 = the drain failed or was cut by its timeout."""
    status = 0
    try:
        final = dec.finish()
    except TruncatedIQError as exc:
        print(f"WARNING: {exc}", file=sys.stderr)
        final, status = dec.take_unreturned_packets(), 1
    except Exception as exc:
        for pkt in dec.take_unreturned_packets():
            show_packet(pkt, freq)
        print(f"ERROR: end-of-stream drain failed: {exc!r}", file=sys.stderr)
        return 3
    for pkt in final:
        show_packet(pkt, freq)
    if dec.finish_reason != "finished":
        print(f"ERROR: end-of-stream drain incomplete ({dec.finish_reason}); packets still in work were cancelled.",
              file=sys.stderr)
        return 3
    return status


def main():
    from lrfhss_hop_ids import parse_hop_ids
    ap=argparse.ArgumentParser(description="Compact LR-FHSS RTL-SDR decoder for the RadioLib STM32WL 722.66-kHz profile.")
    src=ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--live",action="store_true",help="stream directly from rtl_sdr.exe")
    src.add_argument("--file",metavar="IQ",help="decode a recorded rtl_sdr uint8 IQ file")
    ap.add_argument("--exhaustive-file",action="store_true",
                    help="file input only: wait for decoder work without a real-time deadline")
    ap.add_argument("--rtl-tcp",default="rtl_tcp.exe",help="rtl_tcp.exe path/name (default: rtl_tcp.exe)")
    ap.add_argument("--freq",type=int,default=DEFAULT_CENTER_HZ,help="RF center Hz (default 915000000)")
    ap.add_argument("--carrier-search-hz",type=float,default=DEFAULT_CARRIER_SEARCH_HZ,help="Initial carrier search half-width in Hz (4500..25000; default 4500)")
    ap.add_argument("--weak-signal-search", action="store_true",
                    help="Search for header sync in low-power windows. Uses more CPU in live and file modes.")
    ap.add_argument("--gain",type=float,default=DEFAULT_GAIN_DB,help="manual tuner gain dB (default 40; use a supported tuner gain; do not use 0 because that enables AGC)")
    ap.add_argument("--ppm",type=int,default=0,help="RTL-SDR PPM correction")
    ap.add_argument("--port",type=int,default=1234,help="local rtl_tcp port (default 1234)")
    ap.add_argument("--trigger-db",type=float,default=None,help="first-header trigger threshold above noise (default 3 dB; NUC rr2 profile: 10 dB)")
    ap.add_argument("--allowed-hop-ids", type=parse_hop_ids, default=None,
                    help="experimental allowed ID set: comma-separated decimal/hex IDs, or all (default 0x100)")
    ap.add_argument("-v","--verbose",action="store_true",help="show triggers/header details")
    ap.add_argument("--record-iq", metavar="PATH", default=None,
                    help="live mode: also write the raw IQ to PATH (a file, or a folder for a timestamped name), "
                         "unchanged, so the pass can be decoded again later (about 7.2 GB per hour)")
    ap.add_argument("--acquisition-execution",choices=["inline","process"],default=None)
    ap.add_argument("--acquisition-processes",type=int,choices=[1,2],default=None,
                    help="opt-in per-ID acquisition processes; one for a shared ID (default 1)")
    ap.add_argument("--nuc-rr2-20s-v2",action="store_true",
                    help="experimental NUC rr2 v2 profile: guarded single-precision CFO FFT, 5 header / 2 payload workers, SIC off")
    ap.add_argument("--acquisition-search-backend",choices=["reference","batched"],default=None)
    ap.add_argument("--phdr-frequency-reuse",action=argparse.BooleanOptionalAction,default=None)
    ap.add_argument("--throughput", action="store_true", help="use the adopted Batch 10 front-end preset: 7 PHDR / 4 payload workers, "
                    "bounded payload retry, and PHDR interference cancellation (SIC) on one low-priority worker; worker overrides remain available")
    ap.add_argument("--throughput-no-sic", action="store_true", help="use the Batch 10 8H/3P preset without SIC (the front-end preset before 10.4h)")
    ap.add_argument("--throughput-9-10", action="store_true", help="use the Batch 9.10 preset (no payload retry)")
    ap.add_argument("--sic", action="store_true", help="experimental Batch 10.4: PHDR successive interference cancellation on 2 extra "
                    "low-priority workers (needs the asynchronous PHDR pipeline, e.g. --throughput)")
    ap.add_argument("--throughput-9-9", action="store_true", help="use the previous Batch 9.9 8H/3P preset (same decoded packets, higher CPU cost)")
    ap.add_argument("--phdr-frontend", choices=("reference", "screened"), default=None,
                    help="reference disables the Batch 9.9 screen and factored oscillator")
    ap.add_argument("--portable-profile", action="store_true", help="use the conservative Batch 9.5 5H/5P quality preset")
    ap.add_argument("--header-workers", type=int, default=None, help="PHDR processes; 0 selects synchronous compatibility mode")
    ap.add_argument("--header-queue-capacity", type=int, default=None)
    ap.add_argument("--header-candidate-capacity", type=int, default=None)
    ap.add_argument("--header-max-age-s", type=float, default=12.)
    ap.add_argument("--payload-workers", type=int, default=None)
    ap.add_argument("--payload-queue-capacity", type=int, default=None)
    ap.add_argument("--payload-fec-backend", choices=["python", "numpy", "native"], default="numpy",
                    help="payload trellis: NumPy default, Python oracle/fallback, optional native")
    ap.add_argument("--noise-estimator", choices=["guard20", "adaptive"], default="guard20",
                    help="operational guarded spectral noise or historical comparison")
    ap.add_argument("--exact-refinement-dedup", action=argparse.BooleanOptionalAction, default=False,
                    help="experimental retry coalescing; disabled after fleet retention failures")
    ap.add_argument("--payload-native-library", default=None, help="optional locally built trellis library; otherwise Python fallback")
    ap.add_argument("--adaptive-header-budget", action="store_true", help="experimental bounded 2/4 header budget")
    ap.add_argument("--header-validation-budget", type=int, default=None, help="maximum header submissions per feed_raw call; 0 means unlimited")
    ap.add_argument("--no-candidate-local-retry", action="store_true")
    ap.add_argument("--header-grid-cache", action="store_true")
    ap.add_argument("--batch-header", action="store_true")
    ap.add_argument("--candidate-scheduler", choices=["fifo", "quality_age"], default="quality_age")
    ap.add_argument("--schedule-phase-s", type=float, default=None, help="unsupported in schedule-blind Batch 9.6; non-null values are rejected")
    ap.add_argument("--candidate-clustering", choices=["legacy", "narrow"], default="legacy")
    a=ap.parse_args()
    if a.nuc_rr2_20s_v2 and any((a.portable_profile,a.throughput_9_9,a.throughput_9_10,a.throughput_no_sic,a.throughput)):
        ap.error("NUC profiles must be selected separately from other profiles")
    payload_options=(dict(NUC_RR2_20S_V2_PROFILE) if a.nuc_rr2_20s_v2 else dict(BATCH09_5_PORTABLE_PROFILE) if a.portable_profile else dict(BATCH09_9_PROFILE) if a.throughput_9_9
                     else dict(BATCH09_10_PROFILE) if a.throughput_9_10 else dict(BATCH10_PROFILE) if a.throughput_no_sic
                     else dict(BATCH10_FRONTEND_PROFILE) if a.throughput else {})
    profile_trigger = payload_options.pop("trigger_db", DEFAULT_TRIGGER_DB)
    if a.trigger_db is None:
        a.trigger_db = profile_trigger
    if a.allowed_hop_ids is not None:
        payload_options["allowed_hop_ids"] = a.allowed_hop_ids
    for key in ("payload_workers", "payload_queue_capacity", "header_workers",
                "header_queue_capacity", "header_candidate_capacity"):
        if getattr(a,key) is not None:payload_options[key]=getattr(a,key)
    for key in ("acquisition_execution","acquisition_processes","acquisition_search_backend","phdr_frequency_reuse","phdr_frontend"):
        if getattr(a,key) is not None:payload_options[key]=getattr(a,key)
    if a.header_validation_budget is not None:
        payload_options["header_validation_budget"]=a.header_validation_budget or None
    if a.adaptive_header_budget:
        payload_options["header_validation_budget"]=a.header_validation_budget or 2
    payload_options.update(header_max_age_s=a.header_max_age_s,
        payload_fec_backend=a.payload_fec_backend,payload_native_library=a.payload_native_library,
        acquisition_options={"noise_mode":a.noise_estimator,"carrier_search_hz":a.carrier_search_hz},
        exact_refinement_dedup=a.exact_refinement_dedup,
        adaptive_header_budget=a.adaptive_header_budget,
        candidate_local_retry=not a.no_candidate_local_retry,
        candidate_scheduler=a.candidate_scheduler,schedule_phase_s=a.schedule_phase_s,
        candidate_clustering=a.candidate_clustering)
    if a.weak_signal_search:
        payload_options["acquisition_options"]["gate_mode"] = "weak_signal"
        if a.nuc_rr2_20s_v2 and a.live and payload_options.get("payload_workers", 0) >= 2:
            payload_options["payload_retry"] = "adaptive"
    if payload_options.get("header_workers") == 0:
        # Synchronous compatibility mode (Audit 02 B8): the same fallbacks as
        # the GUI. The replica skip and candidate transactions need the
        # asynchronous PHDR pipeline.
        payload_options.update(header_replica_skip="off", candidate_transactions=False,
                               batch_header=False, header_grid_cache=False)
        if a.phdr_frontend is None:
            payload_options["phdr_frontend"] = "reference"
        if payload_options.get("header_validation_budget") is None:
            payload_options["header_validation_budget"] = 2
    if a.sic and payload_options.get("allowed_hop_ids", (DEFAULT_HOPSEQ,)) != (DEFAULT_HOPSEQ,):
        ap.error("SIC is untested with multiple hop IDs; omit --sic")
    if a.sic:
        if not payload_options.get("header_workers"):
            ap.error("--sic needs the asynchronous PHDR pipeline (for example --throughput)")
        payload_options.update(phdr_sic="on")
    if payload_options.get("phdr_sic") == "on" and not payload_options.get("header_workers"):
        # the preset's SIC needs the asynchronous PHDR pipeline: synchronous mode runs without it
        for key in ("phdr_sic", "sic_workers", "sic_payload_allowance"):
            payload_options.pop(key, None)
    if payload_options.get("phdr_sic") == "on":
        # live: keep the drain short; files: finish every SIC tile at the end of the recording
        payload_options["sic_pressure"] = {"mode": FRONTEND_SIC_PRESSURE_MODE}
        payload_options["sic_eof"] = "skip" if a.live else "flush"
    if a.batch_header:payload_options["batch_header"]=True
    if a.header_grid_cache:payload_options["header_grid_cache"]=True
    if a.live:
        return run_live(a.rtl_tcp,a.freq,a.gain,a.ppm,a.port,a.trigger_db,a.verbose,record_iq=a.record_iq,**payload_options)
    if a.exhaustive_file:
        return run_file(a.file,a.trigger_db,a.verbose,exhaustive_file=True,**payload_options)
    return run_file(a.file,a.trigger_db,a.verbose,**payload_options)

if __name__=="__main__":
    raise SystemExit(main())
