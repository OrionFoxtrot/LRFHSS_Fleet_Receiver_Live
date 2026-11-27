"""Instrumented PHDR physical kernels (payload processing is unchanged).

The reference frontend spells out SciPy's odd-padded forward/backward SOS
algorithm so each physical component has an inclusive CPU and wall counter.
No private SciPy API or process-global receiver configuration is required.
"""
from functools import lru_cache
import numpy as np
from scipy import signal
from lrfhss_dsp import coordinates, _sos
from lrfhss_metrics import measure


def stage(counters, name):
    return measure(counters, 'phdr_kernel_' + name)


@lru_cache(maxsize=8)
def _zi(fs, cutoff):
    value = signal.sosfilt_zi(_sos(fs, cutoff).copy())
    value.flags.writeable = False
    return value


def reference_mix_lowpass(z, frequency, cutoff, counters, *, factored=False):
    """The Batch 9.8 complex128 arithmetic, with separately timed stages."""
    from lrfhss_live import FS
    with stage(counters, 'mix_lowpass'):
        counters['phdr_kernel_fullrate_input_samples'] += len(z)
        with stage(counters, 'coordinates'):
            if not factored:
                n, = coordinates.get(('time', len(z)),
                    lambda: (np.arange(len(z), dtype=np.float64),))
        with stage(counters, 'oscillator'):
            if factored:
                # Exact complex-exponential identity with different float64
                # argument-rounding. The reference mode retains original math;
                # screened physical jobs use the separately validated identity.
                block = 1024
                within = ((-2*np.pi*frequency)*np.arange(block))*(1.0/FS)
                starts = ((-2*np.pi*frequency)*(np.arange((len(z)+block-1)//block)*block))*(1.0/FS)
                mixed = ((np.cos(starts)+1j*np.sin(starts))[:, None]*
                         (np.cos(within)+1j*np.sin(within))[None, :]).ravel()[:len(z)]
                counters['phdr_kernel_factored_oscillator_calls'] += 1
                counters['phdr_kernel_oscillator_trig_values'] += 2*(len(within)+len(starts))
            else:
                angle = ((-2*np.pi*frequency)*n)*(1.0/FS)
                mixed = np.empty(len(z), dtype=np.complex128)
                np.cos(angle, out=mixed.real)
                np.sin(angle, out=mixed.imag)
                counters['phdr_kernel_oscillator_trig_values'] += 2*len(z)
        with stage(counters, 'complex_multiply'):
            mixed *= z
        paired = mixed.view(np.float64).reshape(-1, 2).T
        sos = _sos(FS, cutoff).copy()
        edge = int(3*(2*len(sos)+1-min((sos[:, 2] == 0).sum(),
                                      (sos[:, 5] == 0).sum())))
        if len(z) <= edge:
            raise ValueError('IQ length must exceed forward/backward pad length')
        with stage(counters, 'filter_allocation_copy'):
            ext = np.concatenate((2*paired[:, :1]-paired[:, edge:0:-1], paired,
                                  2*paired[:, -1:]-paired[:, -2:-edge-2:-1]), axis=1)
            zi = _zi(FS, cutoff)[:, None, :]
        with stage(counters, 'filter_forward'):
            y, _ = signal.sosfilt(sos, ext, axis=-1, zi=zi*ext[:, :1])
        with stage(counters, 'filter_backward'):
            y, _ = signal.sosfilt(sos, y[:, ::-1], axis=-1, zi=zi*y[:, -1:])
        with stage(counters, 'output_allocation_copy'):
            filtered = y[:, ::-1][:, edge:-edge]
            result = np.empty(len(z), dtype=np.complex128)[::-1]
            result.real = filtered[0]
            result.imag = filtered[1]
        counters['phdr_kernel_sos_input_samples'] += 2*(len(z)+2*edge)
        counters['phdr_kernel_expensive_mix_filter_executions'] += 1
        return result


class WorkspaceBusy(RuntimeError):
    pass


class MixWorkspace:
    """One private synchronous validator lease; public mixers remain owned."""
    def __init__(self, length, fs=1_000_000, cutoff=250.):
        self.length, self.fs, self.cutoff = length, fs, cutoff
        from threading import Lock
        self.lease_lock = Lock()
        self.busy = False
        self.block = 1024
        self.blocks = (length+self.block-1)//self.block
        self.oscillator = np.empty((self.blocks,self.block),np.complex128)
        self.within = np.arange(self.block)
        self.starts = np.arange(self.blocks)*self.block
        self.sos = _sos(fs,cutoff).copy()
        self.edge = int(3*(2*len(self.sos)+1-min((self.sos[:,2]==0).sum(),(self.sos[:,5]==0).sum())))
        self.ext = np.empty((length+2*self.edge,2),np.float64).T
        self.zi = _zi(fs,cutoff)[:,None,:]
        self.output = np.empty(length,np.complex128)

    @property
    def retained_bytes(self):
        return sum(x.nbytes for x in (self.oscillator,self.within,self.starts,self.sos,self.ext,self.output))

    from contextlib import contextmanager
    @contextmanager
    def lease(self,z,frequency,counters):
        if len(z)!=self.length or len(z)<=self.edge or z.ndim!=1:
            raise ValueError('Unsupported workspace input')
        if not self.lease_lock.acquire(False):
            raise WorkspaceBusy('Header workspace already leased')
        self.busy = True
        try:
            with stage(counters,'mix_lowpass'):
                counters['phdr_kernel_fullrate_input_samples'] += len(z)
                with stage(counters,'oscillator'):
                    within = ((-2*np.pi*frequency)*self.within)*(1./self.fs)
                    starts = ((-2*np.pi*frequency)*self.starts)*(1./self.fs)
                    np.multiply((np.cos(starts)+1j*np.sin(starts))[:,None],
                                (np.cos(within)+1j*np.sin(within))[None,:],out=self.oscillator)
                    counters['phdr_kernel_factored_oscillator_calls'] += 1
                    counters['phdr_kernel_oscillator_trig_values'] += 2*(len(within)+len(starts))
                mixed = self.oscillator.ravel()[:self.length]
                with stage(counters,'complex_multiply'):
                    mixed *= z
                paired = mixed.view(np.float64).reshape(-1,2).T
                e = self.edge
                with stage(counters,'filter_allocation_copy'):
                    self.ext[:,:e] = 2*paired[:,:1]-paired[:,e:0:-1]
                    self.ext[:,e:-e] = paired
                    self.ext[:,-e:] = 2*paired[:,-1:]-paired[:,-2:-e-2:-1]
                with stage(counters,'filter_forward'):
                    y,_ = signal.sosfilt(self.sos,self.ext,axis=-1,zi=self.zi*self.ext[:,:1])
                with stage(counters,'filter_backward'):
                    y,_ = signal.sosfilt(self.sos,y[:,::-1],axis=-1,zi=self.zi*y[:,-1:])
                with stage(counters,'output_allocation_copy'):
                    filtered = y[:,::-1][:,e:-e]
                    result = self.output[::-1]
                    result.real = filtered[0]
                    result.imag = filtered[1]
                counters['phdr_kernel_sos_input_samples'] += 2*(len(z)+2*e)
                counters['phdr_kernel_expensive_mix_filter_executions'] += 1
                counters['phdr_workspace_reused_calls'] += 1
                counters['phdr_workspace_retained_bytes'] = self.retained_bytes
            yield result
        finally:
            self.busy = False
            self.lease_lock.release()


class HeaderMixWorkspace:
    """Bounded per-worker storage, active only within a physical validator."""
    def __init__(self):
        from contextvars import ContextVar
        from threading import Lock
        self.workspace = None
        self.allocation_lock = Lock()
        self.active = ContextVar('header_mix_scope',default=None)

    from contextlib import contextmanager
    @contextmanager
    def transaction(self):
        from contextlib import ExitStack
        with ExitStack() as scope:
            token = self.active.set(scope)
            try:
                yield
            finally:
                self.active.reset(token)

    def mix(self,z,frequency,cutoff,counters,*,factored=False):
        scope = self.active.get()
        if (scope is None or not factored or cutoff!=250.
                or len(z)!=325_000 or z.ndim!=1):
            counters['phdr_workspace_fallback_calls'] += 1
            return reference_mix_lowpass(z,frequency,cutoff,counters,factored=factored)
        if self.workspace is None:
            if not self.allocation_lock.acquire(False):
                counters['phdr_workspace_fallback_calls'] += 1
                return reference_mix_lowpass(z,frequency,cutoff,counters,factored=factored)
            try:
                if self.workspace is None:self.workspace = MixWorkspace(325_000)
            finally:self.allocation_lock.release()
        workspace = self.workspace
        if workspace.busy:
            counters['phdr_workspace_fallback_calls'] += 1
            return reference_mix_lowpass(z,frequency,cutoff,counters,factored=factored)
        try:
            return scope.enter_context(workspace.lease(z,frequency,counters))
        except WorkspaceBusy:
            counters['phdr_workspace_fallback_calls'] += 1
            return reference_mix_lowpass(z,frequency,cutoff,counters,factored=factored)
