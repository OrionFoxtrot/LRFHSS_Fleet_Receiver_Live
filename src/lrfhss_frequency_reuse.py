"""Compact exact PHDR FFT-mask reuse; no IQ or FFT cache crosses jobs.

A SHA-256 identity covers the entire immutable snapshot, shape, dtype, range,
base sample, candidate, sample rate, FFT backend and algorithm version. No
similarity matching. A mismatch recomputes; every attempt still validates.
"""
from dataclasses import dataclass, astuple
import hashlib
import math
import numpy as np
from scipy import fft

VERSION = 'frequency-masks-v1-hann-ascending-argmax'

@dataclass(frozen=True)
class FrequencySelection:
    identity: tuple
    frequencies: tuple


def identity(z, base, candidate, fs, grid_backend):
    anchor = candidate.header_start-base
    key = (VERSION, hashlib.sha256(memoryview(z).cast('B') if z.flags.c_contiguous else z.tobytes()).hexdigest(), z.shape, z.dtype.str,
        base, astuple(candidate), fs, anchor+2000, .20, bool(grid_backend))
    mode=getattr(grid_backend,'cfo_fft','float64')
    return key if mode=='float64' else key+('cfo_fft',mode,'cfo_tie_margin',
                                           getattr(grid_backend,'cfo_tie_margin',.01))


def select(z, base, candidate, fs, grid_backend, previous=None):
    key = identity(z, base, candidate, fs, grid_backend)
    if previous is not None and previous.identity == key:
        return previous, True
    s=max(0, candidate.header_start-base+2000)
    x=z[s:min(len(z), s+int(.20*fs))]
    nominal=candidate.frequency_hz
    if hasattr(grid_backend, 'estimate_masks'):
        values=tuple(grid_backend.estimate_masks(z,s,.20,nominal,(250.,100.,0.)).items())
    elif len(x)<1000:
        values=((250.,nominal),(100.,nominal),(0.,nominal))
    else:
        nfft=1 << int(math.ceil(math.log2(len(x))))
        window=np.hanning(len(x))
        f=np.fft.fftshift(np.fft.fftfreq(nfft,1/fs))
        if grid_backend:
            X=fft.fft(x*window,n=nfft,workers=1)
        else:
            X=np.fft.fftshift(np.fft.fft(x*window,n=nfft))
        values=[]
        for span in (250.,100.):
            m=(f>=nominal-span)&(f<=nominal+span)
            # Identical ascending frequency-bin tie order in both backends.
            indices=(np.flatnonzero(m)+nfft//2)%nfft if grid_backend else m
            values.append((span,float(f[m][np.argmax(np.abs(X[indices])**2)])))
        values.append((0.,nominal))
    return FrequencySelection(key,tuple(values)), False
