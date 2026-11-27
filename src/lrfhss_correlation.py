"""Exact reuse of the real FFTs used by the three acquisition correlations.

Uses the same padding, reversed kernels, FFT operations and valid-mode slice as
scipy.signal.correlate(..., method='fft'). No peak or threshold approximation.
Only fixed template spectra are cached; IQ and soft observations are not kept.
"""
from collections import OrderedDict
import numpy as np
from scipy import fft, signal


class AcquisitionCorrelations:
    def __init__(self, kernel, ones, max_bytes=2*1024*1024):
        self.kernel=np.asarray(kernel,dtype=np.float64).copy()
        self.ones=np.asarray(ones,dtype=np.float64).copy()
        self.entries=OrderedDict();self.bytes=0;self.max_bytes=max_bytes

    def __call__(self, soft):
        if len(soft)<len(self.kernel):
            return (signal.correlate(soft,self.kernel,'valid',method='fft'),
                    signal.correlate(soft,self.ones,'valid',method='fft'),
                    signal.correlate(soft**2,self.ones,'valid',method='fft'))
        n=fft.next_fast_len(len(soft)+len(self.kernel)-1,real=True)
        spectra=self.entries.pop(n,None)
        if spectra is None:
            spectra=(fft.rfftn(self.kernel[::-1],[n],axes=[0]),
                     fft.rfftn(self.ones[::-1],[n],axes=[0]))
            size=sum(x.nbytes for x in spectra)
            if size<=self.max_bytes:
                while self.entries and (self.bytes+size>self.max_bytes or len(self.entries)>=16):
                    _,old=self.entries.popitem(last=False);self.bytes-=sum(x.nbytes for x in old)
                for x in spectra:x.flags.writeable=False
                self.entries[n]=spectra;self.bytes+=size
        else:self.entries[n]=spectra
        transformed=fft.rfftn(soft,[n],axes=[0])
        squared=fft.rfftn(soft**2,[n],axes=[0])
        valid=slice(len(self.kernel)-1,len(soft))
        return (fft.irfftn(transformed*spectra[0],[n],axes=[0])[valid],
                fft.irfftn(transformed*spectra[1],[n],axes=[0])[valid],
                fft.irfftn(squared*spectra[1],[n],axes=[0])[valid])

    def batched(self, soft):
        """Identical FFT padding and valid slice, independent CFO rows."""
        soft = np.asarray(soft)
        if soft.ndim != 2:
            raise ValueError("Expected CFO-by-time matrix")
        length = soft.shape[-1]
        if length < len(self.kernel):
            return tuple(np.stack(rows) for rows in zip(*(self(row) for row in soft)))
        n = fft.next_fast_len(length+len(self.kernel)-1, real=True)
        spectra = self.entries.get(n)
        if spectra is None:
            # Populate the same bounded immutable template cache.
            self(soft[0])
            spectra = self.entries.get(n)
            if spectra is None:
                spectra = (fft.rfftn(self.kernel[::-1], [n], axes=[0]),
                           fft.rfftn(self.ones[::-1], [n], axes=[0]))
        transformed = fft.rfftn(soft, [n], axes=[-1])
        squared = fft.rfftn(soft**2, [n], axes=[-1])
        valid = slice(len(self.kernel)-1, length)
        return (fft.irfftn(transformed*spectra[0], [n], axes=[-1])[:, valid],
                fft.irfftn(transformed*spectra[1], [n], axes=[-1])[:, valid],
                fft.irfftn(squared*spectra[1], [n], axes=[-1])[:, valid])
