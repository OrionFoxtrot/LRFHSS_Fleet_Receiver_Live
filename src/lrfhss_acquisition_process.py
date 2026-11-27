"""Bounded spawn acquisition. Only numerical proposals cross the process boundary.

Order is (block_id, channel traversal, retained-CFO power rank, peak sample).
That is the reference traversal, including its local suppression tie breaks.
One worker and FIFO queues restore it without an unbounded reorder buffer.
"""
from collections import Counter, deque
import multiprocessing as mp
import os
import pickle
import queue
import time
import traceback
import threading


def _worker(inbox, outbox, fs, spb, nominal, trigger, options):
    from lrfhss_acquisition import HeaderAcquisition
    from lrfhss_live import u8_iq_to_complex
    a = HeaderAcquisition(fs, spb, nominal, trigger, lambda *args, **kw: None, **options)
    # The options carry the owner's channel_map. Proposals retain raw channel
    # indices; the owner maps each to (allowed IDs, physical copy) exactly once.
    # Proposals precede owner-side local suppression and scheduling. No pending
    # candidate queue or application state exists in the numerical worker.
    offers = []
    max_offers = len(nominal)*(a.MAX_CFO_PEAKS*(int(a.CONTEXT_S*a.low_fs)//(2*a.low_spb)+3)+1)
    if a.wide_search is not None:
        w = a.wide_search
        max_offers += len(nominal)*(w.MAX_CFO_PEAKS*(int(w.CONTEXT_S*w.low_fs)//(2*w.low_spb)+3)+1)
    def offer(*fields):
        if len(offers) >= max_offers:
            raise RuntimeError("Acquisition result exceeded derived peak bound")
        offers.append(fields)
    a._offer = offer
    # Exercise filter/search numerics before readiness, then discard warm state.
    import numpy as np
    a.feed(np.zeros(50000, np.complex64), 0)
    warm=np.exp(2j*np.pi*1000*np.arange(7031)/a.low_fs)
    a._search(warm, 0, 0, 1., 50000)
    a = HeaderAcquisition(fs, spb, nominal, trigger, lambda *args, **kw: None, **options)
    a._offer = offer
    outbox.send_bytes(pickle.dumps(dict(kind="ready", pid=os.getpid(), max_offers=max_offers), protocol=5))
    expected = 0
    next_block = 0
    while True:
        wire = inbox.recv_bytes()
        if not wire:
            return
        try:
            block_id, start, raw = pickle.loads(wire)
            if block_id != next_block or start != expected:
                raise RuntimeError("Noncontiguous acquisition input")
            offers.clear();a.counts.clear();a.processing.clear()
            t, cpu = time.perf_counter(), time.process_time()
            z = u8_iq_to_complex(raw)
            a.feed(z, start)
            expected = start+len(z);next_block += 1
            result = dict(kind="block", block_id=block_id, start=start, end=expected,
                offers=offers, counts=dict(a.counts), processing=dict(a.processing),
                noise=list(a.noise), energy_references=list(a.energy_references),
                wall_s=time.perf_counter()-t, cpu_s=time.process_time()-cpu,
                completed_wall=time.perf_counter())
            outbox.send_bytes(pickle.dumps(result, protocol=5))
        except BaseException:
            outbox.send_bytes(pickle.dumps(dict(kind="error", traceback=traceback.format_exc()), protocol=5))
            return


class AcquisitionProcess:
    def __init__(self, fs, spb, nominal, trigger, options, *, capacity=8, timeout=30.):
        if capacity < 1 or capacity > 64 or timeout <= 0:
            raise ValueError("Acquisition capacity must be 1..64 and timeout positive")
        self.capacity, self.timeout, self.fs = capacity, timeout, fs
        self.ctx = mp.get_context("spawn")
        # Use local bounded mailboxes and one-way pipes. A killed process can
        # corrupt multiprocessing.Queue's receive semaphore while holding it;
        # one-way pipe ownership instead guarantees EOF wakes the I/O threads.
        self.inbox = queue.Queue(capacity)
        self.outbox = queue.Queue(capacity)
        self.stopped = threading.Event()
        self.transport_errors = []
        self.transport_counters = Counter(pipe_send_wall_s=0., pipe_send_messages=0, pipe_send_bytes=0)
        self.sender = self.reader = None
        child_in, self.parent_in = self.ctx.Pipe(duplex=False)
        self.parent_out, child_out = self.ctx.Pipe(duplex=False)
        self.child_ends = (child_in, child_out)
        self.pending = deque()
        self.counters = Counter()
        self.next_block = self.watermark = self.submitted_sample = 0
        self.closed = False
        self.ready = None
        self.process = self.ctx.Process(target=_worker,
            args=(child_in, child_out, fs, spb, nominal, trigger, options), name="lrfhss-acquisition")
        self.last_progress = time.monotonic()
        self.started = time.perf_counter()
        try:
            self.process.start()
            child_in.close();child_out.close()
            self.sender=threading.Thread(target=self._send, name='acquisition-input', daemon=True)
            self.reader=threading.Thread(target=self._receive, name='acquisition-output', daemon=True)
            self.sender.start();self.reader.start()
            while self.ready is None:
                if time.perf_counter()-self.started > timeout:
                    raise TimeoutError("Acquisition readiness timeout")
                try:
                    self.ready = pickle.loads(self.outbox.get(timeout=.025))
                except queue.Empty:
                    self._health()
            if self.ready.get("kind") != "ready":
                raise RuntimeError(str(self.ready))
            self.counters['readiness_wall_s'] = time.perf_counter()-self.started
        except BaseException:
            self.close()
            raise

    def _send(self):
        try:
            while not self.stopped.is_set():
                try:wire=self.inbox.get(timeout=.025)
                except queue.Empty:continue
                if wire is None:
                    self.parent_in.send_bytes(b'')
                    return
                start=time.perf_counter()
                self.parent_in.send_bytes(wire)
                self.transport_counters['pipe_send_wall_s']+=time.perf_counter()-start
                self.transport_counters['pipe_send_messages']+=1
                self.transport_counters['pipe_send_bytes']+=len(wire)
        except (OSError,EOFError) as exc:
            if not self.stopped.is_set():self.transport_errors.append(repr(exc))
        finally:self.parent_in.close()

    def _receive(self):
        try:
            while not self.stopped.is_set():
                wire=self.parent_out.recv_bytes()
                while not self.stopped.is_set():
                    try:self.outbox.put(wire,timeout=.025);break
                    except queue.Full:continue
        except (OSError,EOFError) as exc:
            if not self.stopped.is_set():self.transport_errors.append(repr(exc))
        finally:self.parent_out.close()

    def is_alive(self):
        return False if self.closed else self.process.is_alive()

    @property
    def outstanding(self):
        return len(self.pending)

    def _health(self):
        if self.transport_errors and self.outbox.empty():
            raise RuntimeError("Acquisition pipe failed: "+str(self.transport_errors))
        if not self.process.is_alive():
            self.counters['worker_failures'] += 1
            raise RuntimeError(f"Acquisition worker exited: {self.process.exitcode}")
        if self.pending and time.monotonic()-self.last_progress > self.timeout:
            self.counters['progress_timeouts'] += 1
            raise TimeoutError("Acquisition stopped making progress")

    def submit(self, raw, start):
        self._health()
        if self.closed or self.outstanding >= self.capacity:
            raise RuntimeError("Acquisition admission bound exceeded")
        if start != self.submitted_sample or len(raw)%2 or len(raw)>100000:
            raise ValueError("Acquisition block must be contiguous and at most 50 ms")
        wire = pickle.dumps((self.next_block, start, raw), protocol=5)
        before = time.perf_counter()
        # Serialization failures are synchronous and visible. End-to-end
        # credits bound both local mailboxes, pipes, worker and received results.
        try:
            self.inbox.put(wire, timeout=self.timeout)
        except queue.Full as exc:
            raise TimeoutError("Acquisition input queue blocked") from exc
        self.counters['queue_send_wall_s'] += time.perf_counter()-before
        if not self.pending:
            self.last_progress = time.monotonic()
        self.pending.append((self.next_block, start, start+len(raw)//2, before))
        self.next_block += 1;self.submitted_sample += len(raw)//2
        self.counters['input_messages'] += 1
        self.counters['input_bytes'] += len(wire)
        self.counters['outstanding_high_watermark'] = max(self.counters['outstanding_high_watermark'], self.outstanding)
        self.counters['lag_high_watermark_samples'] = max(self.counters['lag_high_watermark_samples'], self.submitted_sample-self.watermark)

    def collect(self, wait=False):
        if self.closed:
            return []
        self._health()
        results = []
        while self.pending:
            try:
                wire = self.outbox.get(timeout=.025) if wait and not results else self.outbox.get_nowait()
            except queue.Empty:
                self._health()
                if wait and not results:
                    continue
                break
            r = pickle.loads(wire)
            if r['kind'] != 'block':
                self.counters['worker_failures'] += 1
                raise RuntimeError("Acquisition failure: "+r.get('traceback',str(r)))
            block_id, start, end, submitted = self.pending.popleft()
            if (r['block_id'], r['start'], r['end']) != (block_id, start, end) or start != self.watermark:
                raise RuntimeError("Acquisition output coordinates/order corrupted")
            self.watermark = end;self.last_progress = time.monotonic()
            self.counters['output_messages'] += 1;self.counters['output_bytes'] += len(wire)
            self.counters['worker_wall_s'] += r['wall_s'];self.counters['worker_cpu_s'] += r['cpu_s']
            self.counters['result_latency_max_s'] = max(self.counters['result_latency_max_s'], time.perf_counter()-submitted)
            self.counters['offers'] += len(r['offers'])
            results.append(r)
        return results

    def stats(self):
        input_depth=0 if self.closed else self.inbox.qsize()
        output_depth=0 if self.closed else self.outbox.qsize()
        self.counters['sampled_input_queue_high_watermark']=max(self.counters['sampled_input_queue_high_watermark'],input_depth)
        self.counters['sampled_output_queue_high_watermark']=max(self.counters['sampled_output_queue_high_watermark'],output_depth)
        wall = self.counters['worker_wall_s']
        return dict(self.counters, **self.transport_counters, capacity=self.capacity, input_capacity=self.capacity,
            output_capacity=self.capacity, input_queue_depth=input_depth, output_queue_depth=output_depth,
            outstanding=self.outstanding,
            processed_through_sample=self.watermark, submitted_sample=self.submitted_sample,
            lag_samples=self.submitted_sample-self.watermark,
            worker_rf_per_busy_wall_s=self.watermark/self.fs/wall if wall else None,
            ready=self.ready, closed=self.closed, pid=self.ready.get('pid') if self.ready else None)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.counters['cancelled_blocks'] += self.outstanding
        self.pending.clear()
        if self.process.pid is not None:
            if self.process.is_alive():
                try:
                    self.inbox.put_nowait(None)
                except queue.Full:
                    pass
                self.process.join(timeout=.5)
            if self.process.is_alive():
                self.process.terminate();self.process.join(timeout=2.)
            if self.process.is_alive():
                self.process.kill();self.process.join(timeout=2.)
            if self.process.is_alive():
                raise RuntimeError("Acquisition process could not be terminated")
            self.counters['exitcode'] = self.process.exitcode
        self.stopped.set()
        for thread in (self.sender,self.reader):
            if thread is not None:thread.join(timeout=2.)
        alive=[t.name for t in (self.sender,self.reader) if t is not None and t.is_alive()]
        for conn in (*self.child_ends,self.parent_in,self.parent_out):conn.close()
        self.process.close()
        for mailbox in (self.inbox,self.outbox):
            while True:
                try:mailbox.get_nowait()
                except queue.Empty:break
        if alive:raise RuntimeError("Acquisition I/O threads did not stop: "+str(alive))
