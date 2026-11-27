"""Bounded PHDR process workers with immutable IQ and ordered owner-side commits.

Workers only evaluate DSP. The owner alone associates headers and mutates tracks.
The outstanding limit includes completed results waiting for an earlier job, so
out-of-order completion cannot make the reorder buffer grow without bound.
"""
from collections import OrderedDict, Counter, deque
from dataclasses import dataclass
import time
import traceback
import os
from multiprocessing.reduction import ForkingPickler
from lrfhss_payload_jobs import PayloadPool
from lrfhss_shared_iq import SharedIQ, SharedSnapshots, SharedReaders

@dataclass(frozen=True)
class HeaderJob:
    job_id: str
    iq: bytes | SharedIQ
    dtype: str
    base: int
    mode: str
    candidate: tuple
    onset: int = 0
    span: float | None = None
    batch_header: bool = False
    grid_cache: bool = False
    delay_s: float = 0.0
    inject_failure: bool = False
    diagnostic_noop: bool = False
    frequency_reuse: bool = False
    frequency_selection: object = None
    validation_reuse: bool = False
    previous_validation: object = None
    candidate_transaction: bool = False
    allow_refinements: bool = True
    retain_iq: bool = False
    oscillator_reuse: bool = False
    phdr_frontend: str = 'reference'
    phdr_decimation: int = 64
    soft_trials: str = 'reference'
    cfo_fft: str = 'float64'
    cfo_tie_margin: float = .01


def _worker(inbox, outbox, test_gate_flag=None):
    readers = SharedReaders()
    try:
        _worker_loop(inbox, outbox, test_gate_flag, readers)
    finally:
        readers.close()


def _worker_loop(inbox, outbox, test_gate_flag, readers):
    init_wall,init_cpu=time.perf_counter(),time.process_time()
    import numpy as np
    import lrfhss_live as phy
    from lrfhss_acquisition import HeaderCandidate
    from lrfhss_header_grids import HeaderSearchGrids
    from lrfhss_timing import Timings
    import lrfhss_header_batch as header_batch
    cache = HeaderSearchGrids(phy.FS)
    from lrfhss_header_frontend import HeaderOscillators
    oscillators = HeaderOscillators()
    use_oscillators = False
    use_factored = False
    from lrfhss_phdr_dsp import reference_mix_lowpass, stage, HeaderMixWorkspace
    mix_workspace = HeaderMixWorkspace()
    # A borrowed result stays alive through all phase, sync and FEC work in
    # this validator. The returned Header/diagnostic contains no IQ view.
    def scoped_validator(fn):
        def validate(*args, **kwargs):
            with mix_workspace.transaction():
                return fn(*args, **kwargs)
        return validate
    for validator_name in ('_validate_physical_frequency', '_validate_physical_frequency_batched'):
        setattr(phy, validator_name, scoped_validator(getattr(phy, validator_name)))
    # These wrappers only live in this PHDR worker; payload DSP is untouched.
    timings = Timings()
    for name in ('mix_lowpass','estimate_frequency','_soft_from_phase','_soft_trials',
                 'sync_score','viterbi_header','parse_header','crc8_bits'):
        original = getattr(phy, name)
        def wrap(fn, label):
            def measured(*args, **kwargs):
                with timings.measure(label):
                    if label == 'mix_lowpass' and use_oscillators:
                        return oscillators.mix_lowpass(*args, counters=counters, **kwargs)
                    if label == 'mix_lowpass':
                        return mix_workspace.mix(*args, counters=counters,
                                                     factored=use_factored, **kwargs)
                    with stage(counters, label):
                        return fn(*args, **kwargs)
            return measured
        setattr(phy, name, wrap(original, name))
    # Batch 8's batched trellis bypasses phy.viterbi_header. Measure that
    # separate entry point too, so the native-port assessment is not based
    # on the few remaining single-header calls alone.
    original_batch = header_batch.viterbi_headers
    def measured_batch(*args, **kwargs):
        with timings.measure('viterbi_headers_batch'):
            with stage(counters, 'viterbi_headers_batch'):
                return original_batch(*args, counters=counters, **kwargs)
    header_batch.viterbi_headers = measured_batch
    original_phase = phy._SampledPhase
    class MeasuredPhase(original_phase):
        def __getitem__(self, index):
            with stage(counters, 'phase'):
                result = super().__getitem__(index)
                counters['phdr_kernel_phase_samples'] += result.size
                return result
    phy._SampledPhase = MeasuredPhase
    init_wall,init_cpu=time.perf_counter()-init_wall,time.process_time()-init_cpu
    while True:
        wire=inbox.recv_bytes()
        dw,dc=time.perf_counter(),time.process_time()
        job=ForkingPickler.loads(wire)
        dw,dc=time.perf_counter()-dw,time.process_time()-dc
        del wire
        if job is None:
            return
        started, cpu = time.perf_counter(), time.process_time()
        outbox.send(('started',job.job_id,dict(worker_wall=started,worker_pid=os.getpid())))
        timings = Timings()
        counters = Counter()
        import lrfhss_dsp
        lrfhss_dsp.set_soft_trials(job.soft_trials)
        use_oscillators = job.oscillator_reuse
        cache.cfo_fft = job.cfo_fft
        cache.cfo_tie_margin = job.cfo_tie_margin
        use_factored = job.phdr_frontend == 'screened' and job.mode == 'physical'
        z = None
        attempts = None
        try:
            if test_gate_flag is not None:
                while not test_gate_flag.value:
                    time.sleep(.005)
            if job.delay_s:
                time.sleep(job.delay_s)
            if job.inject_failure:
                raise RuntimeError('injected PHDR worker failure')
            with timings.measure('array_reconstruction'):
                if isinstance(job.iq, SharedIQ):
                    z = readers.array(job.iq, job.dtype)
                else:
                    z = np.frombuffer(job.iq, dtype=job.dtype)
            if job.diagnostic_noop:
                h, detail = None, dict(reason='diagnostic_noop', diagnostic_only=True)
            elif job.mode == 'legacy':
                detail = {}
                h = phy.decode_first_header(z, job.onset-job.base, detail)
                if h is not None:
                    h.start += job.base
            elif job.candidate_transaction and job.mode == 'physical':
                from lrfhss_header_transaction import evaluate
                attempts = evaluate(z, job.base, HeaderCandidate(*job.candidate),
                    batch_header=job.batch_header, grids=cache if job.grid_cache else None,
                    counters=counters, allow_refinements=job.allow_refinements,
                    first_span=job.span, frequency_selection=job.frequency_selection,
                    previous_validation=job.previous_validation,
                    frontend=job.phdr_frontend, decimation=job.phdr_decimation)
                h, detail = attempts[0]['header'], attempts[0]['detail']
            else:
                candidate = HeaderCandidate(*job.candidate)
                h,detail = phy.decode_physical_header(z, job.base, candidate,
                    refinement_span=job.span, counters=counters,
                    batch_header=job.batch_header,
                    header_grids=cache if job.grid_cache else None,
                    frequency_reuse=job.frequency_reuse, frequency_selection=job.frequency_selection,
                    validation_reuse=job.validation_reuse, previous_validation=job.previous_validation)
            result = dict(header=h, detail=detail, counters=dict(counters),
                          timings={k:list(v) for k,v in timings.values.items()},
                          grid_cache=cache.stats() if job.grid_cache else None,
                          worker_pid=os.getpid(),
                          oscillator_cache=oscillators.stats() if use_oscillators else None)
            if attempts is not None:
                result['attempts'] = attempts
            kind = 'completed'
        except Exception as exc:
            result = dict(error_type=type(exc).__name__, error=str(exc),
                          traceback=traceback.format_exc())
            kind = 'failed'
        finally:
            # A terminal message authorizes owner slot reuse. Drop the active
            # view first; idle mappings persist until bounded LRU/shutdown.
            z = None
        result.update(worker_pid=os.getpid(),worker_entry_wall=started,worker_wall=time.perf_counter(),wall_s=time.perf_counter()-started,
                      cpu_s=time.process_time()-cpu,deserialize_wall_s=dw,deserialize_cpu_s=dc,
                      initialization_wall_s=init_wall,initialization_cpu_s=init_cpu)
        outbox.send((kind,job.job_id,result))
        init_wall=init_cpu=0.
        del job,result


class HeaderPool:
    def __init__(self, workers=2, capacity=16, test_gate=None, *, outstanding_capacity=None,
                 shared_iq=False, retained_capacity=0):
        if outstanding_capacity is not None and outstanding_capacity < workers+capacity:
            raise ValueError('Outstanding capacity must cover the bounded worker/queue slots')
        self.outstanding_capacity = workers+capacity if outstanding_capacity is None else outstanding_capacity
        self.pool = PayloadPool(workers,capacity,test_gate,
                               worker_target=_worker,thread_prefix='header')
        self.workers,self.capacity = workers,capacity
        self.order = OrderedDict()
        self.completed = {}
        self.started_messages = deque(maxlen=self.outstanding_capacity)
        self.peak_outstanding = self.peak_reorder = 0
        self.snapshots = SharedSnapshots(workers+capacity+retained_capacity) if shared_iq else None
        self.leases = {}
        self.retained = {}
        self.retain_on_completion = set()

    @property
    def available(self):
        if self.pool.closed:
            return 0
        # Completed headers need only small result records, not retained IQ.
        # Bound these separately from the unchanged active/waiting snapshot cap.
        live=sum(s['job'] is not None for s in self.pool.slots)+len(self.pool.waiting)
        return max(0,min(self.outstanding_capacity-len(self.order),self.workers+self.capacity-live))

    def submit(self,job):
        if not self.available or not self.pool.can_admit():
            return False
        if not self.pool.submit(job):
            return False
        if isinstance(job.iq, SharedIQ):
            self.leases[job.job_id] = job.iq
            self.retained.pop((job.iq.name, job.iq.generation), None)
            if job.retain_iq:
                self.retain_on_completion.add(job.job_id)
        self.order[job.job_id] = None
        self.peak_outstanding = max(self.peak_outstanding,len(self.order))
        return True

    def submit_cached(self, job_id, result):
        """Reserve the SAME bounded global result order without worker DSP/IQ.

        A cached logical retry must not jump over earlier submitted headers.
        This consumes a small ordered-result slot, never a worker queue slot.
        """
        if self.pool.closed or len(self.order) >= self.outstanding_capacity:
            return False
        if job_id in self.order or job_id in self.completed:
            raise RuntimeError('Duplicate cached header result ID')
        self.order[job_id] = None
        self.completed[job_id] = ('completed', job_id, result)
        self.peak_outstanding = max(self.peak_outstanding, len(self.order))
        self.peak_reorder = max(self.peak_reorder, len(self.completed))
        return True

    def pump(self):
        """Owner-only transport progress; no commits or TrackManager mutation.

        Restart already-admitted waiting jobs while acquisition is between
        channels. All messages remain bounded by the unchanged outstanding
        limit and terminal results still commit strictly in submission order.
        """
        for kind,job_id,result in self.pool.poll():
            if kind == 'started':
                if len(self.started_messages) == self.started_messages.maxlen:
                    raise RuntimeError('Header start-message accounting exceeded outstanding bound')
                self.started_messages.append((kind,job_id,result))
            else:
                descriptor = self.leases.pop(job_id, None)
                if descriptor is not None:
                    if job_id in self.retain_on_completion and kind == 'completed':
                        self.retained[(descriptor.name, descriptor.generation)] = descriptor
                    else:
                        self.snapshots.release(descriptor)
                self.retain_on_completion.discard(job_id)
                self.completed[job_id] = (kind,job_id,result)
        self.peak_reorder = max(self.peak_reorder,len(self.completed))

    def release_retained(self, descriptor):
        key = (descriptor.name, descriptor.generation)
        if self.retained.pop(key, None) is None:
            raise RuntimeError('Candidate IQ lease is not retained by the owner')
        self.snapshots.release(descriptor)

    def poll(self):
        self.pump()
        messages = list(self.started_messages)
        self.started_messages.clear()
        while self.order and next(iter(self.order)) in self.completed:
            job_id,_ = self.order.popitem(last=False)
            messages.append(self.completed.pop(job_id))
        return messages

    def close(self, *, graceful=False):
        self.pool.close(graceful=graceful)
        if self.snapshots is not None:
            self.snapshots.close()
        self.leases.clear()
        self.retained.clear()
        self.retain_on_completion.clear()
        self.order.clear()
        self.completed.clear()
        self.started_messages.clear()

    def stats(self):
        return {**{k.replace('payload_','header_'):v for k,v in self.pool.stats().items()},
                'header_jobs_outstanding':len(self.order),
                'header_ordered_result_capacity':self.outstanding_capacity,
                'header_outstanding_high_watermark':self.peak_outstanding,
                'header_reorder_high_watermark':self.peak_reorder,
                'header_retained_candidate_leases':len(self.retained),
                'header_shared_iq':self.snapshots.stats() if self.snapshots is not None else None}
