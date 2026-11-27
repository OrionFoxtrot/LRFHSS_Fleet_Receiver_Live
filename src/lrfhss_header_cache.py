"""Owner-local, byte-bounded oscillator cache. No IQ or packet state retained."""
from collections import OrderedDict
import numpy as np
from scipy import signal

class HeaderDSPCache:
    def __init__(self, fs, max_bytes=16*1024*1024):
        self.fs=fs;self.max_bytes=max_bytes;self.entries=OrderedDict()
        self.nbytes=0;self.peak_bytes=0;self.hits=0;self.misses=0
    def mix(self,z,freq,cutoff):
        key=(len(z),float(freq),float(cutoff))
        pair=self.entries.pop(key,None)
        if pair is None:
            self.misses+=1
            n=np.arange(len(z),dtype=np.float64)
            # Deliberately preserve the original expression and operation order.
            oscillator=np.exp(-1j*2*np.pi*freq*n/self.fs)
            sos=signal.butter(6,cutoff,btype='low',fs=self.fs,output='sos')
            pair=(oscillator,sos)
            size=oscillator.nbytes+sos.nbytes
            if size<=self.max_bytes:
                while self.entries and self.nbytes+size>self.max_bytes:
                    _,old=self.entries.popitem(last=False)
                    self.nbytes-=sum(x.nbytes for x in old)
                self.entries[key]=pair;self.nbytes+=size
                self.peak_bytes=max(self.peak_bytes,self.nbytes)
        else:
            self.hits+=1;self.entries[key]=pair
        return signal.sosfiltfilt(pair[1],z*pair[0])
    def clear(self):
        self.entries.clear();self.nbytes=0
    def stats(self):
        return dict(hits=self.hits,misses=self.misses,bytes=self.nbytes,peak_bytes=self.peak_bytes,max_bytes=self.max_bytes)
