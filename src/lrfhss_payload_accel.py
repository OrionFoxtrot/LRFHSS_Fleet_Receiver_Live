"""Optional narrow payload-trellis experiment; reference PHY stays unchanged.

NumPy compares the same two predecessors of each destination. A strict less-
than comparison chooses the lower numbered predecessor on ties, exactly as the
reference's ascending state loop. Native code uses the same arithmetic/order.
No backend changes payload CRC, dewhitening, tracking or worker ownership.
"""
from contextlib import contextmanager
from functools import lru_cache,wraps
from pathlib import Path
import ctypes
import time
import numpy as np

def _finish(cost,bits,phy):
    if len(bits)<22:return None
    body_crc=bits[:-6];data=body_crc[:-16];received=body_crc[-16:]
    return float(cost),data,bool(np.array_equal(phy.crc16_bits(data),received))

@lru_cache(maxsize=1)
def _trellis():
    from lrfhss_live import PAY_OUT_BITS
    destinations=np.arange(64);p0=destinations//2;p1=p0+32;bit=destinations&1
    return p0,p1,2*PAY_OUT_BITS[p0,bit,:2].astype(float)-1,2*PAY_OUT_BITS[p1,bit,:2].astype(float)-1

def viterbi_numpy(soft):
    import lrfhss_live as phy
    vals=np.asarray(soft,dtype=float);ninfo=int(np.floor(len(vals)/1.5))
    metric=np.full(64,np.inf);metric[0]=0.;prev=np.full((ninfo,64),-1,dtype=np.int16)
    p0,p1,e0,e1=_trellis();cursor=0
    for i in range(ninfo):
        if not i&1:
            r=vals[cursor:cursor+2];cursor+=2
            c0=metric[p0]+np.sum((e0-r)**2,axis=1)
            c1=metric[p1]+np.sum((e1-r)**2,axis=1)
        else:
            r=vals[cursor];cursor+=1
            c0=metric[p0]+(e0[:,1]-r)**2
            c1=metric[p1]+(e1[:,1]-r)**2
        # The reference never propagates a nonfinite predecessor or accepts
        # NaN/Inf candidate cost into its initially infinite destination.
        c0=np.where(np.isfinite(metric[p0])&np.isfinite(c0),c0,np.inf)
        c1=np.where(np.isfinite(metric[p1])&np.isfinite(c1),c1,np.inf)
        one=c1<c0;metric=np.where(one,c1,c0)
        prev[i]=np.where(np.isfinite(metric),np.where(one,p1,p0),-1)
    if not np.isfinite(metric[0]):return None
    state=0;bits=np.zeros(ninfo,dtype=np.uint8)
    for i in range(ninfo-1,-1,-1):
        bits[i]=state&1;state=int(prev[i,state])
        if state<0 and i:return None
    return _finish(metric[0],bits,phy)

@lru_cache(maxsize=4)
def _load_native(path):
    library=ctypes.CDLL(str(Path(path).resolve()))
    fn=library.lrfhss_viterbi_cr23
    fn.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,
                 ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_double)]
    fn.restype=ctypes.c_int
    return library,fn

def viterbi_native(soft,library_path):
    import lrfhss_live as phy
    vals=np.ascontiguousarray(soft,dtype=np.float64)
    bits=np.empty(int(np.floor(len(vals)/1.5)),dtype=np.uint8);cost=ctypes.c_double()
    table=np.ascontiguousarray(phy.PAY_OUT_BITS[:,:,:2],dtype=np.uint8)
    _,fn=_load_native(str(library_path))
    status=fn(vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(vals),
              table.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
              bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),ctypes.byref(cost))
    if status<0:raise RuntimeError('Native trellis allocation/argument failure')
    return _finish(cost.value,bits,phy) if status else None

@contextmanager
def kernel_context(backend='python',native_library=None,profile=False,capture=False):
    """Worker-local, sequential-job binding with unconditional restoration."""
    import lrfhss_live as phy
    reference=phy.viterbi_payload_cr23;originals={};records={};vectors=[]
    selected=backend;fallback=None
    if backend=='numpy':implementation=viterbi_numpy
    elif backend=='native':
        try:
            if native_library is None:raise OSError('No optional native library supplied')
            _load_native(str(native_library))
            implementation=lambda soft:viterbi_native(soft,native_library)
        except (OSError,AttributeError) as exc:
            implementation=reference;selected='python';fallback=str(exc)
    elif backend=='python':implementation=reference
    else:raise ValueError('Unknown payload FEC backend')
    info=dict(requested_backend=backend,selected_backend=selected,fallback_reason=fallback,
              kernels=records,fec_vectors=vectors)
    def bind(name,fn):
        originals.setdefault(name,getattr(phy,name))
        def wrapper(*args,**kwargs):
            wall,cpu=time.perf_counter(),time.process_time()
            try:
                result=fn(*args,**kwargs)
                if capture and name=='viterbi_payload_cr23':
                    vectors.append(dict(soft_f64_hex=np.asarray(args[0],dtype='<f8').tobytes().hex(),
                        cost=None if result is None else result[0],data_hex=None if result is None else result[1].tobytes().hex(),
                        crc_ok=None if result is None else bool(result[2])))
                return result
            finally:
                if profile:
                    r=records.setdefault(name,dict(calls=0,wall_s=0.,cpu_s=0.))
                    r['calls']+=1;r['wall_s']+=time.perf_counter()-wall;r['cpu_s']+=time.process_time()-cpu
        setattr(phy,name,wraps(fn)(wrapper) if profile or capture else fn)
    try:
        bind('viterbi_payload_cr23',implementation)
        if profile:
            for name in ('deinterleave_payload','crc16_bits','dewhiten','_soft_from_phase','_soft_trials','mix_lowpass','extract_payload_fragments'):
                bind(name,getattr(phy,name))
        yield info
    finally:
        for name,fn in originals.items():setattr(phy,name,fn)
