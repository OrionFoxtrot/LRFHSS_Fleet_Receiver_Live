"""Batch 10.4: successive interference cancellation (SIC) of physical headers.

Worker-side DSP only; no track, queue or publication state lives here. The owner
(lrfhss_sic_owner.SicScheduler) decides what to rebuild and where to search.

Why only headers: every fleet packet uses hop sequence 0x100, so copy k of
every PHDR lands on the same header channel k, and a stronger neighbour's copy
captures the weaker one (Batch 10.1). Payload fragments never collide
co-channel in this workload, so they are left alone.

1. rebuild_packet(): for a packet whose payload passed its CRC, model each of
   its three PHDR copies. The transmitted symbols follow from the decoded
   header fields alone: the 32 header bits (with the copy's countdown index),
   CRC-8, and the tail-biting convolutional code (initial state = final state,
   observed on every recording, 10.4), interleaver and sync word. Only the two
   guard symbols are decided from the IQ. A multirate least-squares fit then
   finds fine timing, residual frequency and a complex gain per 8,192 samples.
   The result is a small PhdrModel (a few hundred bytes), never a waveform.
2. search(): on one header channel and one time tile, subtract every model,
   look for sync on the cleaned channel (the acquisition's detector), keep only
   candidates whose band a subtracted model actually occupied at comparable or
   higher power, and validate them with the receiver's unchanged PHDR
   validator on a cleaned full-rate window.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy import signal

FS = 1_000_000
SPB = 2048
SYMBOLS = 114
STAY_HDR = 233_472
HEADER_COUNT = 3
SYNC_WORD = 0x12AD101B
SYNC_BITS = np.array([(SYNC_WORD >> n) & 1 for n in range(31, -1, -1)], dtype=np.uint8)
FIT_GRID = 8          # synthesized-phase grid step (samples)
FIT_BOX = 16          # boxcar pre-decimation (1 MS/s -> 62.5 kS/s)
FIT_SEARCH = 4        # further decimation for the fit search (-> 15.6 kS/s)
LOW_DECIMATION = 64   # channel search rate, as the acquisition (15.625 kS/s)
DEFAULT_BT = 1.0      # best measured cancellation (10.3): BT 1.0 ~ MSK, BT 0.3 poor
VALIDATION_MODEL_SPAN_HZ = 1000.   # models cancelled in a validation window: within this of the candidate


# ----------------------------------------------------------------------------- symbols

def phdr_bits(info, copy_index):
    """The 40 transmitted header bits of copy ``copy_index`` (32 bits + CRC-8)."""
    sync_index = HEADER_COUNT - 1 - int(copy_index)
    fields = ((info["payload_len"], 8), (info["modulation"], 3), (info["cr"], 2), (info["grid"], 1),
              (info["hop"], 1), (info["bw_code"], 4), (info["hop_seq_id"], 9), (sync_index, 2), (0, 2))
    bits = []
    for value, width in fields:
        value = int(value)
        if not 0 <= value < (1 << width):
            raise ValueError("header field out of range")
        bits.extend((value >> n) & 1 for n in range(width-1, -1, -1))
    crc = 0xff
    for b in bits:
        top = (crc >> 7) ^ b
        crc = ((crc << 1) ^ (top*0x2f)) & 0xff
    bits.extend((crc >> n) & 1 for n in range(7, -1, -1))
    return np.array(bits, dtype=np.uint8)


def encode_header(bits40, init=None):
    """Header convolutional code; tail-biting (init = final state) by default."""
    import lrfhss_live as phy
    bits40 = np.asarray(bits40, dtype=np.uint8)
    state = int("".join(map(str, bits40[-4:])), 2) if init is None else int(init)
    out = []
    for b in bits40:
        out.extend(phy.HEADER_OUT_BITS[state, int(b)])
        state = ((state << 1) | int(b)) & 15
    return np.array(out, dtype=np.uint8)


def phdr_symbols(info, copy_index, guard0=0., guard113=0.):
    """The 114 +/-1 symbols of one PHDR copy; the guards are given (-1, 0 or +1)."""
    import lrfhss_live as phy
    coded80 = encode_header(phdr_bits(info, copy_index))[phy.HEADER_INTERLEAVER]
    body = 2.0*np.concatenate([coded80[:40], SYNC_BITS, coded80[40:]]).astype(float) - 1.0
    return np.concatenate([[float(guard0)], body, [float(guard113)]])


# ----------------------------------------------------------------------------- waveform

def _pulse_integral(x, bt):
    """Integral of the unit-area GMSK frequency pulse from -inf to x (x in symbols)."""
    if bt is None:
        return np.clip(x + 0.5, 0.0, 1.0)
    from scipy.special import ndtr
    sigma = math.sqrt(math.log(2)) / (2*math.pi*bt)
    def F(u):
        return u*ndtr(u/sigma) + sigma*np.exp(-0.5*(u/sigma)**2)/math.sqrt(2*math.pi)
    return F(x + 0.5) - F(x - 0.5)


def gmsk_phase(symbols, first_center, positions, bt=DEFAULT_BT):
    """GMSK (h = 0.5) phase at sorted positions: pi/2 * sum_k a_k G((t - c_k)/T), c_k = first_center + k*T.

    Batch 10.4g: on a uniform grid whose step divides the symbol period, the interior
    symbols' contributions are one convolution of the symbol train with the shared
    pulse template (FFT), instead of a Python loop over the symbols; the edge symbols
    and every other grid use the reference loop (_gmsk_phase_loop)."""
    fast = _gmsk_phase_uniform(symbols, first_center, positions, bt)
    return fast if fast is not None else _gmsk_phase_loop(symbols, first_center, positions, bt)


def _gmsk_phase_uniform(symbols, first_center, positions, bt):
    T = SPB
    t = np.asarray(positions, dtype=float)
    n = len(t)
    if n < 2:
        return None
    h = t[1] - t[0]
    if h <= 0 or not float(T/h).is_integer() or not np.allclose(np.diff(t), h):
        return None
    stride = int(round(T/h))
    a = np.asarray(symbols, dtype=float)
    c = first_center + T*np.arange(len(a))
    i0 = np.searchsorted(t, c - 4*T)
    i1 = np.searchsorted(t, c + 4*T)
    interior = (i0 > 0) & (i1 < n) & (a != 0)
    if interior.sum() < 8:
        return None
    ks = np.flatnonzero(interior)
    k0, k1 = int(ks[0]), int(ks[-1])
    span = np.arange(k0, k1 + 1)
    # every interior symbol must share one template (same length and grid offset)
    if (not np.all(interior[k0:k1 + 1] | (a[k0:k1 + 1] == 0)) or
            np.any(i1[span] - i0[span] != i1[k0] - i0[k0]) or np.any(i0[span] - i0[k0] != stride*(span - k0))):
        return None
    L = int(i1[k0] - i0[k0])
    offset = t[i0[k0]] - c[k0]
    if np.any(np.abs((t[i0[span]] - c[span]) - offset) > 1e-6):
        return None
    template = _pulse_integral((t[i0[k0]:i1[k0]] - c[k0])/T, bt)
    train = np.zeros(stride*(k1 - k0) + 1)
    train[::stride] = a[k0:k1 + 1]
    from scipy.signal import fftconvolve
    contrib = fftconvolve(train, template)            # length stride*(k1-k0) + L
    phase = np.zeros(n)
    phase[i0[k0]:i0[k0] + len(contrib)] += contrib
    steps = np.zeros(n + 1)
    np.add.at(steps, i1[span], a[k0:k1 + 1])
    for k in np.flatnonzero(~interior & (a != 0)):        # edge symbols: exact per-symbol path
        lo, hi = int(i0[k]), int(i1[k])
        phase[lo:hi] += a[k]*_pulse_integral((t[lo:hi] - c[k])/T, bt)
        steps[hi] += a[k]
    return (math.pi/2) * (phase + np.cumsum(steps)[:-1])


def _gmsk_phase_loop(symbols, first_center, positions, bt=DEFAULT_BT):
    """Reference implementation: one symbol at a time."""
    T = SPB
    t = np.asarray(positions, dtype=float)
    phase = np.zeros(len(t)); steps = np.zeros(len(t) + 1)
    h = t[1] - t[0] if len(t) > 1 else 0.
    uniform = h > 0 and float(T/h).is_integer() and np.allclose(np.diff(t), h)
    template = None
    for k, a in enumerate(symbols):
        if a == 0:
            continue
        c = first_center + k*T
        i0, i1 = np.searchsorted(t, (c - 4*T, c + 4*T))
        if uniform and 0 < i0 and i1 < len(t):
            if template is None or template[0] != i1 - i0 or abs(template[1] - (t[i0] - c)) > 1e-6:
                template = (i1 - i0, t[i0] - c, _pulse_integral((t[i0:i1] - c)/T, bt))
            phase[i0:i1] += a*template[2]
        else:
            phase[i0:i1] += a*_pulse_integral((t[i0:i1] - c)/T, bt)
        steps[i1] += a
    return (math.pi/2) * (phase + np.cumsum(steps)[:-1])


def _tone_peak(y, fs, pad=4):
    nfft = 1 << (int(len(y)*pad - 1).bit_length())
    Y = np.abs(np.fft.fft(y, nfft))
    k = int(np.argmax(Y))
    if 0 < k < nfft-1:
        a, b, c = Y[k-1], Y[k], Y[k+1]
        den = a - 2*b + c
        k = k + (0.5*(a-c)/den if den != 0 else 0.)
    f = k*fs/nfft
    if f > fs/2:
        f -= fs
    return f, float(Y.max())


def mixer(freq_hz, n, offset=0):
    """exp(-2 pi j freq (offset + m)/FS) for m < n, complex128, via an outer product (cheaper than cos/sin)."""
    B = 1024
    blocks = -(-int(n) // B)
    inner = np.exp((-2j*np.pi*freq_hz/FS)*np.arange(B, dtype=np.float64))
    outer = np.exp((-2j*np.pi*freq_hz/FS)*(offset + B*np.arange(blocks, dtype=np.float64)))
    return (outer[:, None]*inner[None, :]).ravel()[:int(n)]


@dataclass
class PhdrModel:
    """One fitted PHDR copy: wave(t) = G(t) * exp(j Phi(t)) * exp(2 pi j F (t - t_ref)/FS) on the body.

    Phi is the GMSK phase of ``symbols`` with symbol k centred on center0 + k*SPB.
    Positions are absolute receiver samples. G is linear between gain knots.
    """
    copy_index: int
    center0: float
    symbols: np.ndarray
    freq_hz: float
    t_ref: int
    gain_t: np.ndarray
    gain: np.ndarray
    bt: float | None
    dt: int = 0
    df_hz: float = 0.
    fit_correlation: float = 0.

    @property
    def body(self):
        lo = self.center0 - SPB/2
        return lo, lo + len(self.symbols)*SPB

    @property
    def power(self):
        return float(np.mean(np.abs(self.gain)**2))


def _phase_grid(model, lo, hi):
    grid = FIT_GRID*np.arange(math.floor(lo/FIT_GRID) - 1, math.ceil(hi/FIT_GRID) + 2, dtype=float)
    return grid, gmsk_phase(model.symbols, model.center0, grid, model.bt)


def synthesize(model, positions, mix_hz=0., mix_ref=0, response=1.):
    """Model samples at sorted absolute positions, optionally mixed down by mix_hz (reference sample mix_ref).

    Channel-domain use: the caller mixed its IQ by exp(-2 pi j mix_hz (t - mix_ref)/FS),
    so the model gets the same factor; ``response`` is the caller's filter gain at
    the model frequency (a real scalar for zero-phase filters)."""
    t = np.asarray(positions, dtype=float)
    out = np.zeros(len(t), dtype=np.complex128)
    lo, hi = model.body
    i0, i1 = np.searchsorted(t, (lo, hi))
    if i1 <= i0:
        return out
    tt = t[i0:i1]
    step = tt[1] - tt[0] if len(tt) > 1 else 0.
    if step >= FIT_GRID and float(SPB/step).is_integer():
        phase = gmsk_phase(model.symbols, model.center0, tt, model.bt)      # coarse uniform grid: direct
    else:
        grid, phi = _phase_grid(model, tt[0], tt[-1])
        phase = np.interp(tt, grid, phi)
    phase += (2*math.pi/FS)*((model.freq_hz - mix_hz)*(tt - model.t_ref))
    if mix_hz:
        phase -= 2*math.pi*((mix_hz*(model.t_ref - mix_ref)/FS) % 1.0)
    gain = np.interp(tt, model.gain_t, model.gain.real) + 1j*np.interp(tt, model.gain_t, model.gain.imag)
    out[i0:i1] = response * gain * np.exp(1j*phase)
    return out


def subtract_models(z, base, models):
    """Return a new full-rate array: z minus every model, over z's absolute span."""
    out = np.array(z, dtype=np.complex128, copy=True)
    n = len(out)
    for m in models:
        lo, hi = m.body
        a = max(0, int(math.floor(lo)) - base); b = min(n, int(math.ceil(hi)) - base + 1)
        if b > a:
            out[a:b] -= synthesize(m, np.arange(base + a, base + b))
    return out


def fit_model(z, base, copy_index, center0, freq, symbols, bt=DEFAULT_BT, segment=8192, margin=4096,
              passes=((-768, 768, 128), (-64, 64, 16), (-8, 8, 8)), decide_guards=True):
    """Multirate least-squares fit of one PHDR copy in z (absolute sample of z[0] = base).

    center0/freq: the receiver's estimate for the copy (symbol-0 centre, Hz).
    Returns a PhdrModel, or None when the copy is not inside z."""
    T = SPB
    start = int(round(center0)) - base
    reach = sum(max(abs(a), abs(b)) for a, b, _ in passes)
    lo = start - T//2 - margin - reach
    hi = int(start + (SYMBOLS-0.5)*T + margin + reach)
    if lo < 0 or hi > len(z):
        return None
    n = hi - lo
    q, d = FIT_BOX, FIT_SEARCH
    t_ref = base + start
    m = n // q
    osc = mixer(freq, m*q, lo - start).astype(np.complex64)
    x16 = (np.asarray(z[lo:lo+m*q], dtype=np.complex64)*osc).reshape(m, q).mean(axis=1, dtype=np.complex128)
    r16 = signal.sosfiltfilt(signal.butter(6, 600., fs=FS/q, output="sos"), x16)
    body_len = len(symbols)*T
    body_lo = start - lo - T//2
    g0 = (q - 1)/2
    pad = -(-reach // FIT_GRID) * FIT_GRID
    grid = g0 - pad + FIT_GRID*np.arange((n + 2*pad)//FIT_GRID + 2)
    sym = np.array(symbols, dtype=float)
    guards = (0, len(sym)-1)
    core = sym.copy()
    if decide_guards:
        core[list(guards)] = 0.
    phi = gmsk_phase(core, start - lo, grid, bt)
    inside = ((grid >= body_lo) & (grid < body_lo + body_len)).astype(float)
    per16 = q // FIT_GRID
    def at16(values, dt, stride):
        j0 = pad//FIT_GRID - dt//FIT_GRID
        return values[j0: j0 + per16*m: per16*stride][: len(range(0, m, stride))]
    s_grid = np.exp(1j*phi) * inside
    rd = r16[::d]
    best = None
    for lo_, hi_, step in passes:
        centre = 0 if best is None else best[1]
        for dt in range(centre+lo_, centre+hi_+1, step):
            if abs(dt) > pad or dt % FIT_GRID:
                continue
            f, mag = _tone_peak(rd*np.conj(at16(s_grid, dt, d)), FS/(q*d), pad=2)
            if best is None or mag > best[0]:
                best = (mag, dt, f)
    _, dt, df = best
    t16 = g0 + q*np.arange(m)
    tone = np.exp(2j*np.pi*df*(t16 + lo - start)/FS)
    if decide_guards:
        # Guards: linear in the phase. Choose -1/0/+1 per guard by correlation near it.
        for gi in guards:
            unit = np.zeros(len(sym)); unit[gi] = 1.
            basis = gmsk_phase(unit, start - lo, grid, bt)
            c = body_lo + dt + (gi + 0.5)*T
            i0, i1 = np.searchsorted(t16, (c - 1.5*T, c + 1.5*T))
            base16 = at16(phi, dt, 1)[i0:i1]; b16 = at16(basis, dt, 1)[i0:i1]
            ins = at16(inside, dt, 1)[i0:i1]
            scores = {}
            for a in (-1., 0., 1.):
                s = np.exp(1j*(base16 + a*b16)) * ins * tone[i0:i1]
                scores[a] = abs(np.vdot(s, r16[i0:i1]))
            a = max(scores, key=scores.get)
            if a:
                phi = phi + a*basis
                core[gi] = a
        s_grid = np.exp(1j*phi) * inside
    s16 = at16(s_grid, dt, 1) * tone
    a0 = max(0., body_lo + dt); b0 = min(float(n), body_lo + dt + body_len)
    edges = np.arange(a0, b0 + segment, segment); edges[-1] = b0
    g, mids = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        i0, i1 = np.searchsorted(t16, (a, b))
        den = np.vdot(s16[i0:i1], s16[i0:i1]).real
        g.append(np.vdot(s16[i0:i1], r16[i0:i1]) / den if den > 0 else 0)
        mids.append((a + b)/2)
    g = np.array(g, dtype=np.complex128)
    i0, i1 = np.searchsorted(t16, (a0, b0))
    fitted = np.interp(t16[i0:i1], mids, g.real) + 1j*np.interp(t16[i0:i1], mids, g.imag)
    num = abs(np.vdot(s16[i0:i1]*fitted, r16[i0:i1]))**2
    den = np.vdot(s16[i0:i1]*fitted, s16[i0:i1]*fitted).real * np.vdot(r16[i0:i1], r16[i0:i1]).real
    return PhdrModel(copy_index=int(copy_index), center0=float(base + start + dt), symbols=core.astype(np.int8),
                     freq_hz=float(freq + df), t_ref=int(t_ref), gain_t=np.array(mids, dtype=float) + base + lo,
                     gain=g, bt=bt, dt=int(dt), df_hz=float(df),
                     fit_correlation=float(num/den) if den > 0 else 0.)


def packet_copies(header_start, header_freq, info):
    """(symbol-0 centre, frequency) of each PHDR copy from the canonical first-header reference."""
    import lrfhss_live as phy
    hops = phy.hop_frequencies(info["grid"], info["bw_code"], info["hop_seq_id"], HEADER_COUNT, 0)
    cfo = header_freq - hops[0]
    return [(int(header_start + k*STAY_HDR), float(hops[k] + cfo)) for k in range(HEADER_COUNT)]


def rebuild_packet(z, base, header_start, header_freq, info, observations=(), bt=DEFAULT_BT, copies=None):
    """PhdrModels for the three copies of one CRC-valid packet (copies outside z are skipped).

    observations: optional (copy_index, physical_start, frequency_hz) measured by the
    receiver's own validator; they replace the canonical projection for that copy.
    copies: optional subset of copy indices to fit (Batch 10.4f: copy 0 first, 1 and 2 later)."""
    measured = {int(k): (int(s), float(f)) for k, s, f in observations}
    models = []
    for k, (start, freq) in enumerate(packet_copies(header_start, header_freq, info)):
        if copies is not None and k not in copies:
            continue
        start, freq = measured.get(k, (start, freq))
        model = fit_model(z, base, k, start, freq, phdr_symbols(info, k), bt=bt)
        if model is not None:
            models.append(model)
    return models


# ----------------------------------------------------------------------------- channel search

def _msk_band_fraction(offset_hz, half_band=250.):
    """Fraction of an MSK (488.28 baud) copy's power inside +/-half_band Hz around a frequency offset_hz away."""
    T = SPB/FS
    f = np.linspace(-4000., 4000., 8001)
    x = 4*f*T
    with np.errstate(divide="ignore", invalid="ignore"):
        psd = (np.cos(2*np.pi*f*T)/(1 - x**2))**2
    psd[~np.isfinite(psd)] = (np.pi/4)**2
    inside = np.abs(f - offset_hz) <= half_band
    return float(psd[inside].sum()/psd.sum())


_BAND_TABLE = None


def band_fraction(offset_hz):
    global _BAND_TABLE
    if _BAND_TABLE is None:
        grid = np.arange(0., 2001., 10.)
        _BAND_TABLE = (grid, np.array([_msk_band_fraction(g) for g in grid]))
    return float(np.interp(abs(offset_hz), *_BAND_TABLE, right=0.))


class Channelizer:
    """Mix to a header channel, boxcar /16, zero-phase low-pass, /4: 15.625 kS/s.

    Sample j of the output is centred on absolute sample base + 64*j + 7.5 (+ offset)."""
    CUTOFF = 5500.

    def __init__(self):
        self.sos = signal.butter(6, self.CUTOFF, fs=FS/FIT_BOX, output="sos")

    def response(self, offset_hz):
        key = round(float(offset_hz), 1)
        cache = self.__dict__.setdefault("_cache", {})
        if key not in cache:
            cache[key] = self._response(key)
        return cache[key]

    def _response(self, offset_hz):
        f = abs(offset_hz)
        box = abs(math.sin(math.pi*FIT_BOX*f/FS)/(FIT_BOX*math.sin(math.pi*f/FS))) if f else 1.
        _, h = signal.sosfreqz(self.sos, worN=[f], fs=FS/FIT_BOX)
        return box * float(abs(h[0]))**2

    def __call__(self, z, base, nominal_hz):
        n = (len(z)//LOW_DECIMATION)*LOW_DECIMATION
        osc = mixer(nominal_hz, n).astype(np.complex64)
        x16 = (np.asarray(z[:n], dtype=np.complex64)*osc).reshape(-1, FIT_BOX).mean(axis=1, dtype=np.complex128)
        y = signal.sosfiltfilt(self.sos, x16)[::LOW_DECIMATION//FIT_BOX]
        positions = base + LOW_DECIMATION*np.arange(len(y)) + (FIT_BOX - 1)/2
        return y, positions


@dataclass
class SicCandidate:
    header_start: int           # physical start (symbol-0 centre) on this channel
    frequency_hz: float
    sync_correlation: float
    energy_ratio: float
    victim_power: float
    removed_power: float        # subtracted model power inside the candidate's validation band

    @property
    def isr(self):
        return self.removed_power / max(self.victim_power, 1e-30)


_CORRELATIONS = None


def _correlations():
    global _CORRELATIONS
    if _CORRELATIONS is None:
        from lrfhss_acquisition import SYNC_SIGNS
        from lrfhss_correlation import AcquisitionCorrelations
        low_spb = SPB//LOW_DECIMATION
        kernel = np.zeros(31*low_spb+1); kernel[::low_spb] = SYNC_SIGNS - SYNC_SIGNS.mean()
        ones = np.zeros(len(kernel)); ones[::low_spb] = 1
        _CORRELATIONS = (AcquisitionCorrelations(kernel, ones), float(np.sum(kernel**2)))
    return _CORRELATIONS


def find_candidates(x, positions, nominal_hz, models, cancelled_models=None, *, lo_start, hi_start,
                    max_peaks=4, peak_floor=0.02, correlation=0.55, window_s=0.25, step_s=0.1,
                    affected_hz=1000.):
    """Sync candidates on a (cleaned) channel whose physical start lies in [lo_start, hi_start).

    The acquisition's detector (FFT peaks -> narrow filter -> phase soft symbols ->
    Pearson correlation with the sync word), but only over the victims' sync
    region and only at CFOs within affected_hz of a subtracted model."""
    low_fs = FS/LOW_DECIMATION; low_spb = SPB//LOW_DECIMATION; half = low_spb//2
    cancelled = list(models) if cancelled_models is None else list(cancelled_models)
    if not cancelled or len(x) < 64*low_spb:
        return []
    # noise: 20th percentile of the band's periodogram (per-sample power)
    nfft = 1 << int(math.ceil(math.log2(len(x))))
    hw = np.hanning(len(x))
    P = np.abs(np.fft.fft(x*hw, nfft))**2 / np.sum(hw**2)
    freq = np.fft.fftfreq(nfft, 1/low_fs)
    noise = max(float(np.percentile(P[np.abs(freq) <= 4500], 20)), 1e-30)
    # sync symbols of a victim starting in [lo_start, hi_start) sit at +40.5..+72.5 symbols
    r0, r1 = np.searchsorted(positions, (lo_start + 40*SPB, hi_start + 73*SPB))
    win = int(window_s*low_fs); step = int(step_s*low_fs)
    hann = np.hanning(win); wfft = 1 << int(math.ceil(math.log2(win)))
    wfreq = np.fft.fftshift(np.fft.fftfreq(wfft, 1/low_fs))
    cfos = []
    a_lo = max(0, r0 - win//2); a_hi = max(a_lo + 1, min(len(x) - win + 1, r1 - win//2 + 1))
    for a in range(a_lo, a_hi, step):
        seg_lo, seg_hi = positions[a], positions[min(len(x)-1, a+win-1)]
        near = [m.freq_hz - nominal_hz for m in cancelled if m.body[0] < seg_hi and m.body[1] > seg_lo]
        if not near:
            continue
        power = np.abs(np.fft.fftshift(np.fft.fft(x[a:a+win]*hann, wfft)))**2
        peaks, _ = signal.find_peaks(power, distance=max(1, round(180/(low_fs/wfft))))
        peaks = peaks[np.abs(wfreq[peaks]) <= 4500]
        if not len(peaks):
            continue
        peaks = peaks[np.argsort(power[peaks])[::-1][:max_peaks]]
        top = power[peaks[0]]
        for p in peaks:
            f = float(wfreq[p])
            if power[p] >= peak_floor*top and min(abs(f - g) for g in near) <= affected_hz:
                cfos.append(f)
    merged = []
    for f in sorted(cfos):
        if merged and f - merged[-1][-1] <= 30.:
            merged[-1].append(f)
        else:
            merged.append([f])
    if not merged:
        return []
    # correlate only over the sync region (plus filter context)
    c0 = max(0, r0 - 4*low_spb); c1 = min(len(x), r1 + 36*low_spb)
    xs = x[c0:c1]; t = np.arange(len(xs))/low_fs
    narrow = signal.butter(4, 300, fs=low_fs, output="sos")
    cfo_rows = np.array([float(np.mean(g)) for g in merged])
    bb = signal.sosfiltfilt(narrow, xs[None, :]*np.exp(-2j*np.pi*cfo_rows[:, None]*t[None, :]), axis=-1)
    phase = np.angle(bb[:, half:]*np.conj(bb[:, :-half]))
    soft = np.clip(phase/(np.pi/4), -1, 1)
    soft[np.abs(phase) > np.pi/2] = 0
    soft = np.pad(soft, ((0, 0), (half//2, half-half//2)))
    correlations, kernel_energy = _correlations()
    num, summed, squared = correlations.batched(soft)
    variance = squared - summed**2/32
    corr_rows = num/np.sqrt(np.maximum(variance, 1e-12)*kernel_energy)
    found = []
    for row, cfo in enumerate(cfo_rows):
        corr = corr_rows[row]
        peaks, _ = signal.find_peaks(corr, height=correlation, distance=2*low_spb)
        for kk in peaks:
            k = kk + c0
            if k < low_spb or k + 32*low_spb > len(x) - low_spb:
                continue
            start = int(round(positions[k] - 41*SPB))
            if not lo_start <= start < hi_start:
                continue
            energy = float(np.mean(np.abs(x[k:k+32*low_spb])**2)/noise)
            if energy < 2:
                continue
            f_abs = nominal_hz + float(cfo)
            vic = float(np.mean(np.abs(bb[row, kk:kk+32*low_spb])**2)) - noise*600./low_fs
            body = (start - SPB/2, start + (SYMBOLS - 0.5)*SPB)
            removed = 0.
            for m in cancelled:
                mlo, mhi = m.body
                overlap = max(0., min(mhi, body[1]) - max(mlo, body[0]))/(body[1] - body[0])
                if overlap > 0:
                    removed += m.power*overlap*band_fraction(m.freq_hz - f_abs)
            found.append(SicCandidate(start, f_abs, float(corr[kk]), energy, max(vic, 0.), removed))
    # the acquisition's local non-maximum suppression (2 symbols, 250 Hz)
    found.sort(key=lambda c: -c.sync_correlation)
    kept = []
    for c in found:
        if all(abs(c.header_start - o.header_start) > 2*SPB or abs(c.frequency_hz - o.frequency_hz) > 250 for o in kept):
            kept.append(c)
    return kept


def search(z, base, copy_index, nominal_hz, models, known=(), *, lo_start, hi_start, isr_threshold=0.1,
           min_correlation=0.7, min_energy=3.,
           max_validations=8, batch_header=True, frontend="screened", decimation=64, grids=None,
           counters=None, candidate_id_base=0):
    """Find and validate PHDR copies on channel copy_index hidden under the subtracted models.

    known: (physical_start, frequency_hz) of packets already known on this channel; a
    candidate within 2,048 samples and 250 Hz of one is not validated again.
    Returns (validated, stats); validated entries are dicts with the Header,
    its detail, and the candidate fields for the owner's association."""
    from collections import Counter
    import lrfhss_live as phy
    from lrfhss_acquisition import HeaderCandidate
    from lrfhss_header_transaction import evaluate
    counts = Counter() if counters is None else counters
    wall, cpu = time.perf_counter(), time.process_time()
    z = np.asarray(z)
    models = [m for m in models if m.copy_index == copy_index]
    stats = dict(models=len(models), candidates=0, known_skipped=0, energy_skipped=0, isr_skipped=0, validated=0,
                 validations=0, valid=0)
    if not models:
        return [], dict(stats, wall_s=0., cpu_s=0.)
    chan = Channelizer()
    c_lo = max(0, (lo_start + 20*SPB - 65536 - base)//LOW_DECIMATION*LOW_DECIMATION)
    c_hi = min(len(z), hi_start + 94*SPB + 65536 - base)
    x, positions = chan(z[c_lo:c_hi], base + c_lo, nominal_hz)
    for m in models:        # cancel in the channel domain (15.6 kS/s) for the search
        x = x - synthesize(m, positions, mix_hz=nominal_hz, mix_ref=base + c_lo,
                           response=chan.response(m.freq_hz - nominal_hz))
    t1 = time.process_time()
    candidates = find_candidates(x, positions, nominal_hz, models, lo_start=lo_start, hi_start=hi_start,
                                 correlation=min_correlation)
    t2 = time.process_time()
    stats["candidates"] = len(candidates)
    todo = []
    for c in candidates:
        if any(abs(c.header_start - s) <= 2048 and abs(c.frequency_hz - f) <= 250 for s, f in known):
            stats["known_skipped"] += 1
            continue
        if c.energy_ratio < min_energy:
            stats["energy_skipped"] += 1
            continue
        if c.isr < isr_threshold:
            stats["isr_skipped"] += 1
            continue
        todo.append(c)
    todo.sort(key=lambda c: -c.sync_correlation)
    validated = []
    for i, c in enumerate(todo[:max_validations]):
        vbase = c.header_start - 35000; vstop = c.header_start + 290000
        if vbase < base or vstop > base + len(z):
            continue
        stats["validations"] += 1
        # full-rate cancellation only where the validator looks: this window, and
        # models close enough in frequency to pass its 250 Hz low-pass/refinement
        near = [m for m in models if abs(m.freq_hz - c.frequency_hz) <= VALIDATION_MODEL_SPAN_HZ
                and m.body[0] < vstop and m.body[1] > vbase]
        window = subtract_models(z[vbase-base:vstop-base], vbase, near)
        window.flags.writeable = False
        cand = HeaderCandidate(candidate_id_base + i, int(c.header_start), int(copy_index), float(nominal_hz),
                               float(c.frequency_hz), "sync", float(c.sync_correlation), float(c.energy_ratio),
                               int(c.header_start))
        attempts = evaluate(window, vbase, cand, batch_header=batch_header, grids=grids, counters=counts,
                            allow_refinements=True, frontend=frontend, decimation=decimation)
        hit = next((a for a in attempts if a["header"] is not None), None)
        if hit is None:
            continue
        stats["valid"] += 1
        detail = {k: v for k, v in hit["detail"].items() if k not in ("frequency_selection", "previous_validation")}
        validated.append(dict(header=hit["header"], detail=detail, physical_index=int(copy_index),
                              candidate_start=int(c.header_start), candidate_frequency_hz=float(c.frequency_hz),
                              sync_correlation=float(c.sync_correlation), energy_ratio=float(c.energy_ratio),
                              isr=float(c.isr), span_hz=hit["span_hz"], models_cancelled=len(near)))
    stats["validated"] = len(validated)
    t3 = time.process_time()
    stats.update(cpu_channel_s=t1-cpu, cpu_search_s=t2-t1, cpu_validate_s=t3-t2,
                 wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu)
    return validated, stats
