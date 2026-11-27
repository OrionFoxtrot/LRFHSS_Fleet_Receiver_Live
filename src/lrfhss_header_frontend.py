"""Exact oscillator reuse across PHDR candidates; filtering remains per input.

An oscillator depends on length, exact frequency, sample rate and arithmetic
expression, not IQ or cutoff. Cache keys retain scalar types as well as float64
bits (NumPy scalar promotion can differ by type). Cached arrays are immutable;
the multiply uses a fresh C-order copy and the unchanged in-place NumPy kernel.
"""
import struct
import time
import numpy as np
from scipy import signal
from lrfhss_dsp import ArrayCache, coordinates, _sos


class HeaderOscillators:
    def __init__(self, max_bytes=64*1024*1024):
        self.cache = ArrayCache(max_bytes=max_bytes, max_entries=128)

    def mix_lowpass(self, z, frequency, cutoff, counters):
        from lrfhss_live import FS
        def scalar(x):
            return (type(x).__module__,type(x).__qualname__,struct.pack('!d',float(x)))
        key=('phasor-v1-f64-reciprocal',len(z),scalar(frequency),scalar(FS))
        before=self.cache.hits
        def make():
            wall,cpu=time.perf_counter(),time.process_time()
            n,=coordinates.get(('time',len(z)),lambda:(np.arange(len(z),dtype=np.float64),))
            angle=((-2*np.pi*frequency)*n)*(1.0/FS)
            oscillator=np.empty(len(z),dtype=np.complex128)
            np.cos(angle,out=oscillator.real);np.sin(angle,out=oscillator.imag)
            return oscillator,time.perf_counter()-wall,time.process_time()-cpu
        oscillator,wall,cpu=self.cache.get(key,make)
        hit=self.cache.hits>before
        counters['phdr_oscillator_reuse_hits' if hit else 'phdr_oscillator_reuse_misses']+=1
        counters['phdr_oscillator_executions']+=int(not hit)
        if hit:
            counters['phdr_oscillator_reused_samples']+=len(z)
            counters['phdr_oscillator_estimated_avoided_cpu_s']+=cpu
            counters['phdr_oscillator_estimated_avoided_wall_s']+=wall
        else:
            counters['phdr_oscillator_cpu_s']+=cpu
            counters['phdr_oscillator_wall_s']+=wall
        # Identical dtype, layout and in-place multiply to lrfhss_dsp.mix_lowpass.
        mixed=oscillator.copy();mixed*=z
        paired=mixed.view(np.float64).reshape(-1,2).T
        filtered=signal.sosfiltfilt(_sos(FS,cutoff).copy(),paired,axis=-1)
        result=np.empty(len(z),dtype=np.complex128)[::-1]
        result.real=filtered[0];result.imag=filtered[1]
        return result

    def stats(self):
        return self.cache.stats()
