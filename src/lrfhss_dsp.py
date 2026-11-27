"""Batch 9.5 NumPy/SciPy kernels; no change to filters or search hypotheses.

Caches contain coordinates, coefficients and spectral indices, never input IQ.
The original function bodies remain executable references in lrfhss_live.
"""
from collections import OrderedDict
from functools import lru_cache
import math
import numpy as np
from scipy import signal, fft


class ArrayCache:
    """Byte AND entry bounded process-local immutable intermediate cache."""
    def __init__(self, max_bytes=8*1024*1024, max_entries=128):
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = self.hits = self.misses = 0

    def get(self, key, make):
        value = self.entries.pop(key, None)
        if value is not None:
            self.hits += 1; self.entries[key] = value
            return value
        self.misses += 1
        value = make()
        size = sum(v.nbytes for v in value if isinstance(v, np.ndarray))
        for v in value:
            if isinstance(v, np.ndarray): v.flags.writeable = False
        if size <= self.max_bytes:
            while self.entries and (self.bytes+size > self.max_bytes or len(self.entries) >= self.max_entries):
                _, old = self.entries.popitem(last=False)
                self.bytes -= sum(v.nbytes for v in old if isinstance(v, np.ndarray))
            self.entries[key] = value; self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        return value

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, bytes=self.bytes,
                    peak_bytes=self.peak_bytes, max_bytes=self.max_bytes,
                    entries=len(self.entries), max_entries=self.max_entries)


coordinates = ArrayCache()
spectral = ArrayCache(max_bytes=4*1024*1024, max_entries=1024)


@lru_cache(maxsize=8)
def _sos(fs, cutoff_hz):
    sos = signal.butter(6, cutoff_hz, btype='low', fs=fs, output='sos')
    sos.flags.writeable = False
    return sos


def mix_lowpass(z, freq_hz, cutoff_hz):
    from lrfhss_live import FS
    n, = coordinates.get(('time',len(z)), lambda: (np.arange(len(z),dtype=np.float64),))
    # Same phase expression and complex128 arithmetic. Real sin/cos ufuncs
    # avoid the generic complex exponential's redundant real-exponent path.
    # NumPy's complex / real-scalar loop multiplies by the reciprocal.
    # Preserve that rounding (real division here changes phase by ~6e-11).
    angle = ((-2*np.pi*freq_hz)*n)*(1.0/FS)
    phasor = np.empty(len(z), dtype=np.complex128)
    np.cos(angle, out=phasor.real); np.sin(angle, out=phasor.imag)
    phasor *= z
    # SciPy requires writable SOS storage; copy only 18 float64 coefficients.
    # Real SOS coefficients act independently on I and Q. SciPy's float64
    # kernel computes the same recurrences without generic complex products.
    paired=phasor.view(np.float64).reshape(-1,2).T
    filtered=signal.sosfiltfilt(_sos(FS,cutoff_hz).copy(),paired,axis=-1)
    # Match sosfiltfilt's reverse-strided output as well: NumPy angle uses
    # different scalar/SIMD rounding for positive and negative strides.
    result=np.empty(len(z),dtype=np.complex128)[::-1]
    result.real=filtered[0]; result.imag=filtered[1]
    return result


SOFT_TRIALS_IMPLEMENTATIONS = ("reference", "grouped")
_soft_trials_impl = "reference"


def set_soft_trials(implementation):
    """Select the per-process soft_trials implementation (Batch 9.10 P2).

    Both implementations are bit-identical for every input; "grouped" only
    removes Python/NumPy call overhead. Workers receive the choice per job.
    """
    global _soft_trials_impl
    if implementation not in SOFT_TRIALS_IMPLEMENTATIONS:
        raise ValueError(f"Unknown soft_trials implementation: {implementation!r}")
    _soft_trials_impl = implementation


def soft_trials(phase, centers):
    if _soft_trials_impl == "grouped":
        return _soft_trials_grouped(phase, centers)
    return _soft_trials_reference(phase, centers)


def _soft_trials_grouped(phase, centers):
    """Batch 9.10 P2: bit-identical to _soft_trials_reference, fewer NumPy calls.

    The masks and element counts for every timing trial are computed at once.
    Each row's two bias means still use a contiguous 1-D reduction over exactly
    the same selected elements in the same order (``row[mask].sum()/count`` is
    the arithmetic ``np.mean`` performs), counts are exact integers, and the
    adjustment is accumulated in the same order (negative side, then positive).
    Rows are independent, so no value depends on another row's processing.
    """
    from lrfhss_live import LOOK, PHASE_SLOPE
    centers = np.asarray(centers,dtype=np.intp)
    d = phase[centers+LOOK]-phase[centers-LOOK]
    wrapped = np.abs(d)>math.pi/2
    d[wrapped] -= np.copysign(2*math.pi,d[wrapped])
    d[np.abs(d)>math.pi/2] = 0.
    vals = (d/(2*LOOK))/PHASE_SLOPE
    p80 = np.percentile(np.abs(vals),80,axis=1)
    use = np.abs(vals) < p80[:, None]
    negative = use & (vals < 0)
    positive = use & (vals >= 0)
    denominators = np.count_nonzero(use, axis=1)
    negative_counts = np.count_nonzero(negative, axis=1)
    positive_counts = np.count_nonzero(positive, axis=1)
    for i in np.flatnonzero(denominators):
        row = vals[i]; denom = int(denominators[i]); adj = 0.
        count = int(negative_counts[i])
        if count:
            adj += (count/denom)*(float(row[negative[i]].sum()/count)+1.)
        count = int(positive_counts[i])
        if count:
            adj += (count/denom)*(float(row[positive[i]].sum()/count)-1.)
        row -= adj
    return np.clip(vals,-1.,1.)


def _soft_trials_reference(phase, centers):
    """Batch phase differences/percentiles; preserve each bias reduction order."""
    from lrfhss_live import LOOK, PHASE_SLOPE
    centers = np.asarray(centers,dtype=np.intp)
    d = phase[centers+LOOK]-phase[centers-LOOK]
    wrapped = np.abs(d)>math.pi/2
    d[wrapped] -= np.copysign(2*math.pi,d[wrapped])
    d[np.abs(d)>math.pi/2] = 0.
    vals = (d/(2*LOOK))/PHASE_SLOPE
    p80 = np.percentile(np.abs(vals),80,axis=1)
    # Rowwise reductions intentionally retain the reference summation order.
    # Padding masked values with zeros changes last-bit ties in timing scores.
    for row, threshold in zip(vals,p80):
        use = np.abs(row)<threshold
        if np.any(use):
            denom = int(np.sum(use)); adj = 0.
            for neg in (True,False):
                m = use & ((row<0) if neg else (row>=0))
                if np.any(m):
                    mean = float(np.mean(row[m]))
                    dev = mean+1. if neg else mean-1.
                    adj += (np.sum(m)/denom)*dev
            row -= adj
    return np.clip(vals,-1.,1.)


def soft_from_phase(phase, centers):
    return soft_trials(phase,np.asarray(centers)[None,:])[0]


class SampledPhase:
    """Compute exactly the phase samples the unchanged demodulator requests.

    A 325k-sample PHDR window only needs 1,582 phase values for the seven
    coarse trials. Avoid the unused full-window arctangent and its allocation.
    The negative input stride retains the reference np.angle rounding path.
    """
    def __init__(self,baseband):
        self.baseband=baseband

    def __getitem__(self,index):
        selected=np.asarray(self.baseband[index])
        phase=np.angle(selected.reshape(-1)[::-1])[::-1]
        return phase.reshape(selected.shape)


def spectral_power(raw, frequency_hz, fs):
    n = len(raw)
    if n<32: return 0.,0.
    nfft = 1<<(n-1).bit_length()
    def window_grid():
        window = np.hanning(n)
        return window, np.fft.fftfreq(nfft,1/fs), nfft*np.sum(window**2)
    window, freq, denominator = coordinates.get(('spectrum',n,fs),window_grid)
    def indices():
        delta = abs((freq-frequency_hz+fs/2)%fs-fs/2)
        return np.flatnonzero(delta<=350.),np.flatnonzero((delta>=800.)&(delta<=2500.))
    main, side = spectral.get((nfft,frequency_hz,fs),indices)
    # Same complex128 FFT and power normalization; only selected bins need
    # squaring. No decimation or replacement filter is involved.
    spectrum = fft.fft(raw*window,n=nfft,workers=1)
    main_power = abs(spectrum[main])**2/denominator
    side_power = abs(spectrum[side])**2/denominator
    return float(np.sum(main_power)),float(np.median(side_power)/math.log(2)*len(main))


def extract_payload_fragments(z, header, overlaps=()):
    import lrfhss_live as phy
    info = header.info
    _, frag_bits = phy.payload_layout(info['payload_len'])
    hops = phy.hop_frequencies(info['grid'],info['bw_code'],info['hop_seq_id'],
                               phy.DEFAULT_HEADER_COUNT,len(frag_bits))
    common_cfo = header.freq_hz-hops[0]
    payload_start = header.start+phy.DEFAULT_HEADER_COUNT*phy.STAY_HDR
    fragments = []
    for j,nbit in enumerate(frag_bits):
        nominal_start = payload_start+j*phy.STAY_DATA
        freq = hops[phy.DEFAULT_HEADER_COUNT+j]+common_cfo
        lo = max(0,nominal_start-2500); hi = min(len(z),nominal_start+phy.STAY_DATA+2500)
        result = phy.FragmentResult(j,float(freq),nominal_start,nbit); fragments.append(result)
        if hi-lo<phy.STAY_DATA:
            result.rejection_reason='fragment_window_unavailable'; continue
        bb = phy.mix_lowpass(z[lo:hi],float(freq),350.)
        phase = SampledPhase(bb)
        shifts = np.arange(-1024,1025,32)
        centers = nominal_start-lo+phy.LOOK+shifts[:,None]+np.arange(nbit+1)*phy.SPB
        valid = (centers[:,0]-phy.LOOK>=0)&(centers[:,-1]+phy.LOOK<len(bb))
        centers,shifts = centers[valid],shifts[valid]
        if not len(shifts):
            result.rejection_reason='no_valid_timing_trial'; continue
        softs = phy._soft_trials(phase,centers)
        scores = np.mean(np.abs(softs),axis=1)
        winner = int(np.argmax(scores))  # first maximum, as the strict > loop
        soft,center = softs[winner],centers[winner]
        result.timing_score = float(scores[winner])
        result.timing_offset_samples = int(shifts[winner])
        result.raw_soft_bits = soft[1:].copy()
        start = nominal_start+result.timing_offset_samples
        raw = z[max(0,start):min(len(z),start+(nbit+1)*phy.SPB)]
        phy.observe_quality(result,raw,phase,center,phy.FS,phy.LOOK,overlaps)
    return tuple(fragments)


def cache_stats():
    return dict(coordinates=coordinates.stats(), spectral_indices=spectral.stats(),
                filter_coefficients=dict(_sos.cache_info()._asdict()))
