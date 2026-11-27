"""Byte-bounded owner-local FFT coordinate cache; never retains IQ or FFT output."""
from collections import OrderedDict
import math
import numpy as np
from scipy import fft

class HeaderSearchGrids:
    def __init__(self,fs,max_bytes=8*1024*1024,*,cfo_fft='float64',cfo_tie_margin=.01):
        self.fs=fs;self.max_bytes=max_bytes;self.entries=OrderedDict()
        self.bytes=0;self.peak_bytes=0;self.hits=0;self.misses=0
        if cfo_fft not in ('float64','float32_guarded'):
            raise ValueError('Unknown header CFO FFT mode')
        self.cfo_fft=cfo_fft
        if cfo_tie_margin != .01:
            raise ValueError('The qualified CFO near-tie guard is fixed at 1%')
        self.cfo_tie_margin=float(cfo_tie_margin)
        self.fast_windows=OrderedDict();self.fast_bytes=0;self.fast_peak_bytes=0
        self.fast_max_bytes=min(max_bytes,2*1024*1024)
        self.fast_ffts=0;self.fast_fallbacks=0;self.fast_unsupported=0

    def _guarded_fft(self,x,window,nfft,masks):
        """Mixed-precision peak search; subsequent validators are unchanged.

        Preserve the reference frequency grid and ascending argmax order. Use
        the original double FFT for near ties, nonfinite values or unsupported
        input precision. The conservative guard is not a proof of identical
        FFT arithmetic: offline payload-set qualification remains mandatory.
        """
        if self.cfo_fft!='float32_guarded':
            raise ValueError('Unknown header CFO FFT mode')
        if x.dtype!=np.complex64:
            self.fast_unsupported+=1
            return fft.fft(x*window,n=nfft,workers=1)
        cached=self.fast_windows.pop(len(window),None)
        if cached is None:
            cached=window.astype(np.float32);cached.flags.writeable=False
            if cached.nbytes<=self.fast_max_bytes:
                while self.fast_windows and self.fast_bytes+cached.nbytes>self.fast_max_bytes:
                    _,old=self.fast_windows.popitem(last=False);self.fast_bytes-=old.nbytes
                self.fast_windows[len(window)]=cached;self.fast_bytes+=cached.nbytes
                self.fast_peak_bytes=max(self.fast_peak_bytes,self.fast_bytes)
        else:self.fast_windows[len(window)]=cached
        self.fast_ffts+=1
        spectrum=fft.fft(x*cached,n=nfft,workers=1)
        peak=float(np.max(np.abs(spectrum)))
        guard=128*np.finfo(np.float32).eps*math.log2(nfft)*peak
        uncertain=not np.isfinite(peak)
        for indices in masks:
            if not len(indices):uncertain=True;break
            if len(indices)==1:continue
            amplitude=np.abs(spectrum[indices].astype(np.complex128))
            largest=np.partition(amplitude,-2)[-2:]
            if (not np.all(np.isfinite(amplitude)) or
                largest[-1]-largest[-2]<=max(guard,self.cfo_tie_margin*largest[-1])):
                uncertain=True;break
        if uncertain:
            self.fast_fallbacks+=1
            return fft.fft(x*window,n=nfft,workers=1)
        return spectrum
    def estimate(self,z,start,duration_s,nominal,span_hz):
        s=max(0,int(start));e=min(len(z),s+int(duration_s*self.fs));x=z[s:e]
        if len(x)<1000:return nominal
        nfft=1<<int(math.ceil(math.log2(len(x))))
        key=(len(x),nfft);pair=self.entries.pop(key,None)
        if pair is None:
            self.misses+=1
            pair=(np.hanning(len(x)),np.fft.fftshift(np.fft.fftfreq(nfft,1/self.fs)))
            size=sum(a.nbytes for a in pair)
            if size<=self.max_bytes:
                while self.entries and self.bytes+size>self.max_bytes:
                    _,old=self.entries.popitem(last=False);self.bytes-=sum(a.nbytes for a in old)
                for a in pair:a.flags.writeable=False
                self.entries[key]=pair;self.bytes+=size;self.peak_bytes=max(self.peak_bytes,self.bytes)
        else:self.hits+=1;self.entries[key]=pair
        window,f=pair
        # Preserve the original ascending shifted-bin order and argmax ties,
        # without copying/fftshifting the full spectrum on every request.
        lo=np.searchsorted(f,nominal-span_hz,side='left')
        hi=np.searchsorted(f,nominal+span_hz,side='right')
        indices=(np.arange(lo,hi)+nfft//2)%nfft
        X=(fft.fft(x*window,n=nfft,workers=1) if self.cfo_fft=='float64'
           else self._guarded_fft(x,window,nfft,[indices]))
        return float(f[lo:hi][np.argmax(np.abs(X[indices])**2)])
    def estimate_masks(self,z,start,duration_s,nominal,spans):
        """Same cached grid and FFT, select each exact ascending-bin mask."""
        s=max(0,int(start));e=min(len(z),s+int(duration_s*self.fs));x=z[s:e]
        if len(x)<1000:return {span:nominal for span in spans}
        nfft=1<<int(math.ceil(math.log2(len(x))))
        key=(len(x),nfft);pair=self.entries.pop(key,None)
        if pair is None:
            self.misses+=1
            pair=(np.hanning(len(x)),np.fft.fftshift(np.fft.fftfreq(nfft,1/self.fs)))
            size=sum(a.nbytes for a in pair)
            if size<=self.max_bytes:
                while self.entries and self.bytes+size>self.max_bytes:
                    _,old=self.entries.popitem(last=False);self.bytes-=sum(a.nbytes for a in old)
                for a in pair:a.flags.writeable=False
                self.entries[key]=pair;self.bytes+=size;self.peak_bytes=max(self.peak_bytes,self.bytes)
        else:self.hits+=1;self.entries[key]=pair
        window,f=pair
        if self.cfo_fft=='float64':
            X=fft.fft(x*window,n=nfft,workers=1)
        else:
            masks=[]
            for span in spans:
                if span==0:continue
                lo=np.searchsorted(f,nominal-span,side='left');hi=np.searchsorted(f,nominal+span,side='right')
                masks.append((np.arange(lo,hi)+nfft//2)%nfft)
            X=self._guarded_fft(x,window,nfft,masks)
        result={}
        for span in spans:
            if span==0:result[span]=nominal;continue
            lo=np.searchsorted(f,nominal-span,side='left')
            hi=np.searchsorted(f,nominal+span,side='right')
            indices=(np.arange(lo,hi)+nfft//2)%nfft
            result[span]=float(f[lo:hi][np.argmax(np.abs(X[indices])**2)])
        return result
    def clear(self):
        self.entries.clear();self.bytes=0;self.fast_windows.clear();self.fast_bytes=0
    def stats(self):
        result=dict(hits=self.hits,misses=self.misses,bytes=self.bytes,peak_bytes=self.peak_bytes,max_bytes=self.max_bytes)
        if self.cfo_fft!='float64':
            result.update(cfo_fft=self.cfo_fft,cfo_tie_margin=self.cfo_tie_margin,
                fast_ffts=self.fast_ffts,fast_fallbacks=self.fast_fallbacks,
                fast_unsupported=self.fast_unsupported,fast_window_bytes=self.fast_bytes,
                fast_window_peak_bytes=self.fast_peak_bytes,fast_window_max_bytes=self.fast_max_bytes)
        return result
