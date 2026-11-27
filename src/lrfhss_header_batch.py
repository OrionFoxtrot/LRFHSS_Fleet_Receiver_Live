"""Same exhaustive header Viterbi, batched over independent timing trials."""
import numpy as np
from contextlib import nullcontext

def viterbi_headers(soft80, phy, *, counters=None):
    obs=np.asarray(soft80,dtype=float).reshape(-1,40,2);n=len(obs)
    if not n:return []
    p0,p1,bit=phy._HDR_P0,phy._HDR_P1,phy._HDR_BIT
    metric=np.full((n,16,16),np.inf)
    metric[:,np.arange(16),np.arange(16)]=0.
    prev=np.empty((40,n,16,16),dtype=np.int8)
    e0=2*phy.HEADER_OUT_BITS[p0,bit].astype(float)-1
    e1=2*phy.HEADER_OUT_BITS[p1,bit].astype(float)-1
    for t in range(40):
        r=obs[:,t,None,:]
        c0=metric[:,:,p0]+np.sum((e0-r)**2,axis=2)[:,None,:]
        c1=metric[:,:,p1]+np.sum((e1-r)**2,axis=2)[:,None,:]
        choose1=c1<c0
        metric=np.where(choose1,c1,c0);prev[t]=np.where(choose1,p1,p0)
    batches=np.arange(n)[:,None,None];rows=np.arange(16)[None,:,None]
    states=np.broadcast_to(np.arange(16),(n,16,16)).copy()
    bits=np.empty((n,16,16,40),dtype=np.uint8)
    for t in range(39,-1,-1):
        bits[:,:,:,t]=states&1;states=prev[t,batches,rows,states]
    from lrfhss_metrics import measure
    with measure(counters, 'phdr_kernel_header_crc_batched') if counters is not None else nullcontext():
        crc=np.full((n,16,16),0xff,dtype=np.uint8)
        for t in range(32):
            top=(crc>>7)^bits[:,:,:,t];crc=(crc<<1)^(top*0x2f)
        received=np.zeros((n,16,16),dtype=np.uint8)
        for t in range(32,40):received=(received<<1)|bits[:,:,:,t]
        valid=crc==received
    out=[]
    for i in range(n):
        if not np.any(valid[i]):out.append(None);continue
        init,end=np.unravel_index(np.argmin(np.where(valid[i],metric[i],np.inf)),(16,16))
        out.append((float(metric[i,init,end]),bits[i,init,end].copy()))
    return out
