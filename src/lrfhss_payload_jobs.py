"""Bounded spawn workers; all pool methods belong to the receiver owner.

Each worker has one in-flight job, including any uncollected result. A bounded
owner deque holds waiting snapshots. Dedicated pipe I/O threads serialize IQ away
from acquisition. Workers never receive a ring, live track, callback or widget.
"""
from collections import deque, Counter
from dataclasses import dataclass
import multiprocessing as mp
import queue
import time
import traceback
import threading
import os
from multiprocessing.reduction import ForkingPickler

# One deadline for the whole pool, independent of worker count. These are
# failure bounds, not sleeps: successful shutdown returns as soon as reaped.
SHUTDOWN_TIMEOUT_S = 2.0
GRACEFUL_SHUTDOWN_TIMEOUT_S = 3.0

if os.name == 'nt':
    import ctypes
    from ctypes import wintypes
    _cancel_io_ex = ctypes.WinDLL('kernel32', use_last_error=True).CancelIoEx
    _cancel_io_ex.argtypes = (wintypes.HANDLE, ctypes.c_void_p)
    _cancel_io_ex.restype = wintypes.BOOL


def _close_connection(connection, lock):
    # Serialize final handle release with the owner's cancellation request.
    # Never hold this lock across a blocking send/receive.
    with lock:
        connection.close()


def _interrupt_io(slot):
    if os.name != 'nt':
        return  # Closing the child's peer endpoints wakes POSIX pipe I/O.
    with slot['io_lock']:
        for connection in (slot['input_connection'], slot['output_connection']):
            if not connection.closed:
                if not _cancel_io_ex(connection.fileno(), None):
                    error = ctypes.get_last_error()
                    if error != 1168:  # ERROR_NOT_FOUND: no pending operation.
                        raise ctypes.WinError(error)


def _worker_entry(target, inbox, outbox, test_gate_flag):
    """Pipe closure is expected during cancellation, outside any DSP job."""
    try:
        target(inbox, outbox, test_gate_flag)
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    finally:
        inbox.close()
        outbox.close()

@dataclass(frozen=True)
class PayloadJob:
    job_id: str
    receiver_id: str
    track_id: int
    packet_start: int
    base: int
    header_start: int
    frequency_hz: float
    header_cost: float
    profile: tuple
    iq: bytes
    dtype: str
    overlaps: tuple
    hops_hz: tuple
    cfo_hz: float
    delay_s: float = 0.0
    inject_failure: bool = False
    fec_backend: str = "python"
    native_library: str | None = None
    profile_kernels: bool = False
    capture_fec_vectors: bool = False
    diagnostic_noop: bool = False
    soft_trials: str = "reference"
    # Batch 10.2: ordered (dt, df) retry offsets used only after a PHY CRC
    # failure; empty = unchanged behaviour. Acceptance also requires the
    # DeSCENT application CRC unless retry_app_crc is False.
    payload_retry: tuple = ()
    retry_app_crc: bool = True

def execute_job(job):
    import_wall,import_cpu=time.perf_counter(),time.process_time()
    import numpy as np
    from lrfhss_live import Header, decode_payload, retry_payload, u8_iq_to_complex_fast, U8_IQ_DTYPE
    import_wall,import_cpu=time.perf_counter()-import_wall,time.process_time()-import_cpu
    import lrfhss_dsp
    lrfhss_dsp.set_soft_trials(job.soft_trials)
    if job.diagnostic_noop:
        # The immutable real IQ bytes have crossed the normal pickle/pipe path.
        np.frombuffer(job.iq, dtype=np.uint8 if job.dtype == U8_IQ_DTYPE else job.dtype)
        return None, {"stage": "diagnostic_noop", "diagnostic_only": True}
    if job.delay_s:
        time.sleep(job.delay_s)
    if job.inject_failure:
        raise RuntimeError("injected payload worker failure")
    rw,rc=time.perf_counter(),time.process_time()
    if job.dtype == U8_IQ_DTYPE:
        # Batch 9.10: exact table form of the owner-ring conversion (bit-identical
        # for all 65,536 byte pairs; see u8_iq_to_complex_fast).
        iq = u8_iq_to_complex_fast(job.iq)
    else:
        iq = np.frombuffer(job.iq, dtype=job.dtype)
    rw,rc=time.perf_counter()-rw,time.process_time()-rc
    header = Header(job.header_start, job.frequency_hz, job.header_cost, dict(job.profile))
    detail = dict(payload_import_wall_s=import_wall,payload_import_cpu_s=import_cpu,
                  payload_array_reconstruction_wall_s=rw,payload_array_reconstruction_cpu_s=rc)
    def decode():
        packet = decode_payload(iq, header, detail, job.overlaps)
        if job.payload_retry and (packet is None or not packet.crc_ok):
            retried = retry_payload(iq, header, detail, job.overlaps, job.payload_retry, job.retry_app_crc)
            if retried is not None:
                packet = retried
        return packet
    if job.fec_backend != "python" or job.profile_kernels or job.capture_fec_vectors:
        from lrfhss_payload_accel import kernel_context
        with kernel_context(job.fec_backend,job.native_library,job.profile_kernels,job.capture_fec_vectors) as kernel_info:
            packet = decode()
        detail["kernel_profile"] = kernel_info
    else:
        packet = decode()
    return packet, detail

def _worker(inbox, outbox, test_gate_flag=None):
    while True:
        wire = inbox.recv_bytes()
        deserialize_wall, deserialize_cpu = time.perf_counter(), time.process_time()
        job = ForkingPickler.loads(wire)
        deserialize_wall = time.perf_counter()-deserialize_wall
        deserialize_cpu = time.process_time()-deserialize_cpu
        del wire
        if job is None:
            return
        started, cpu = time.perf_counter(), time.process_time()
        outbox.send(("started", job.job_id, {"worker_wall": started, "worker_pid": os.getpid()}))
        try:
            if test_gate_flag is not None:
                while not test_gate_flag.value:
                    time.sleep(.005)
            packet, detail = execute_job(job)
            result = {"packet": packet, "detail": detail}
            kind = "completed"
        except Exception as exc:
            kind = "failed"
            result = {"error_type": type(exc).__name__, "error": str(exc),
                      "traceback": traceback.format_exc()}
        result.update(worker_pid=os.getpid(), worker_entry_wall=started, worker_wall=time.perf_counter(), wall_s=time.perf_counter()-started,
                      cpu_s=time.process_time()-cpu,
                      deserialize_wall_s=deserialize_wall,deserialize_cpu_s=deserialize_cpu)
        outbox.send((kind, job.job_id, result))
        del job, result

def _send_jobs(connection, mailbox, stopped, ipc=None, observer=None, io_lock=None):
    io_lock = io_lock or threading.Lock()
    try:
        while not stopped.is_set():
            try:
                job = mailbox.get(timeout=.05)
            except queue.Empty:
                continue
            if stopped.is_set():
                break
            wall,cpu=time.perf_counter(),time.thread_time()
            wire=ForkingPickler.dumps(job)
            serialized=time.perf_counter()
            if ipc is not None and job is not None:
                ipc['serialization_wall_s']+=serialized-wall
                ipc['serialization_thread_cpu_s']+=time.thread_time()-cpu
                ipc['messages']+=1
                ipc['bytes']+=len(wire)
            if stopped.is_set():
                break
            connection.send_bytes(wire)
            if observer is not None and job is not None:
                observer("transport_sent", job.job_id, dict(serialize_start_wall=wall,
                    serialize_end_wall=serialized, send_end_wall=time.perf_counter(), wire_bytes=len(wire)))
            if ipc is not None and job is not None:
                ipc['pipe_send_wall_s']+=time.perf_counter()-serialized
            del wire
            del job
    except (EOFError, OSError):
        pass  # process death is classified by the owner, never by an I/O thread
    except Exception:
        if not stopped.is_set():
            raise
    finally:
        _close_connection(connection, io_lock)


def _receive_results(connection, mailbox, stopped, observer=None, io_lock=None):
    io_lock = io_lock or threading.Lock()
    try:
        while not stopped.is_set():
            # An idle pipe cannot hide the stop event indefinitely. poll alone
            # is insufficient for a partial large message; shutdown also cancels
            # active Windows I/O and closes peer endpoints by reaping the child.
            if not connection.poll(.05) or stopped.is_set():
                continue
            message = connection.recv()
            if observer is not None:
                message[2]["result_received_wall"] = time.perf_counter()
                observer("transport_received", message[1], dict(kind=message[0], **{k:v for k,v in message[2].items() if k in ("worker_pid","worker_entry_wall","worker_wall","cpu_s","wall_s","deserialize_wall_s","deserialize_cpu_s","result_received_wall")}))
            while not stopped.is_set():
                try:
                    mailbox.put(message, timeout=.05)
                    break
                except queue.Full:
                    continue
    except (EOFError, OSError):
        pass
    except Exception:
        # CPython's Windows partial-message path can report an assertion or
        # RuntimeError when overlapped I/O is cancelled. Only shutdown permits it.
        if not stopped.is_set():
            raise
    finally:
        _close_connection(connection, io_lock)


class PayloadPool:
    def __init__(self, workers=1, capacity=4, test_gate=None, *, worker_target=None,
                 thread_prefix="payload"):
        if workers < 1 or capacity < 1:
            raise ValueError("payload workers and queue capacity must be positive")
        self.workers, self.capacity = workers, capacity
        self.observer = None
        self.job_transform = None
        self.readiness = []
        self.worker_target = _worker if worker_target is None else worker_target
        self.thread_prefix = thread_prefix
        self.test_gate = test_gate
        self.context = mp.get_context("spawn")
        # Test-only worker holds must not block a spawned worker on a
        # multiprocessing Event/Condition that may be left corrupted if the
        # worker is deliberately terminated. Mirror any parent-side gate into
        # a lock-free shared byte instead. Production operation leaves this None.
        self._test_gate_flag = self.context.RawValue("b", 0) if test_gate is not None else None
        self.waiting, self.slots = deque(), []
        self.closed = False
        self.closed_ipc = Counter()
        self.spawn_wall_s = 0.
        self.high_watermark = self.max_active = self.max_retained_iq_bytes = 0
        # Batch 9.10 publication invariants. Messages already removed from a
        # worker outbox are owner-held here until poll() returns them, so an
        # exception during dead-slot disposal can never discard a result.
        self._undelivered = deque()
        self.poll_disposal_errors = 0
        self.dead_worker_results_recovered = 0
        self.dead_idle_workers_replaced = 0
        # Terminal results still sitting in an outbox when a slot is shut down.
        # Bounded identity sample plus exact counts; never unbounded history.
        self.shutdown_discarded = Counter()
        self.shutdown_discarded_samples = deque(maxlen=64)

    def _new_slot(self):
        spawn_started=time.perf_counter()
        child_in, parent_in = self.context.Pipe(duplex=False)
        parent_out, child_out = self.context.Pipe(duplex=False)
        self._sync_test_gate()
        process = self.context.Process(target=_worker_entry,
            args=(self.worker_target, child_in, child_out, self._test_gate_flag), daemon=True)
        try:
            process.start()
        except Exception:
            for conn in (child_in, parent_in, parent_out, child_out): conn.close()
            process.close()
            raise
        # Release parent copies of the child endpoints immediately. Peer death
        # wakes POSIX pipe I/O; Windows shutdown also explicitly cancels pending
        # overlapped I/O, and idle receivers poll their stop event.
        child_in.close(); child_out.close()
        inbox, outbox = queue.Queue(1), queue.Queue(2)
        stopped = threading.Event()
        io_lock = threading.Lock()
        ipc=Counter()
        sender = threading.Thread(target=_send_jobs,
            args=(parent_in, inbox, stopped, ipc, self.observer, io_lock), daemon=True, name=self.thread_prefix+"-input")
        reader = threading.Thread(target=_receive_results,
            args=(parent_out, outbox, stopped, self.observer, io_lock), daemon=True, name=self.thread_prefix+"-output")
        sender.start(); reader.start()
        self.spawn_wall_s+=time.perf_counter()-spawn_started
        return dict(process=process, inbox=inbox, outbox=outbox, job=None, iq_bytes=0,
                    stopped=stopped, sender=sender, reader=reader, ipc=ipc,
                    input_connection=parent_in, output_connection=parent_out, io_lock=io_lock)

    def warmup(self, job, timeout=60., stop_event=None):
        """Owner-only readiness barrier, before RF admission; no pool job counters.

        A real representative job is dispatched directly to EVERY slot. A PID is
        ready only after successful numerical decode and result receipt. Failure,
        cancellation or timeout closes every slot and cannot admit RF accidentally.
        """
        from dataclasses import replace
        if self.closed or self.waiting or any(s["job"] is not None for s in self.slots):
            raise RuntimeError("Warmup requires an open, idle pool before RF admission")
        if timeout <= 0:
            raise ValueError("Warmup timeout must be positive")
        begin = time.perf_counter()
        expected = {}
        records = []
        try:
            while len(self.slots) < self.workers:
                self.slots.append(self._new_slot())
            for i, slot in enumerate(self.slots):
                jid = f"warmup:{self.thread_prefix}:{i}"
                expected[jid] = slot["process"].pid
                self._dispatch(slot, replace(job, job_id=jid, diagnostic_noop=False))
            while len(records) < self.workers:
                if stop_event is not None and stop_event.is_set():
                    raise InterruptedError("Worker readiness cancelled")
                if time.perf_counter()-begin > timeout:
                    raise TimeoutError("Not every distinct worker acknowledged readiness")
                for kind, jid, result in self.poll():
                    if kind == "started":
                        continue
                    if kind != "completed" or result.get("worker_pid") != expected.get(jid):
                        raise RuntimeError(f"Worker warmup failed: {jid}: {result}")
                    value = result.get("header") if self.thread_prefix == "header" else result.get("packet")
                    if value is None or (self.thread_prefix != "header" and not value.crc_ok):
                        raise RuntimeError("Warmup did not exercise a successful real numerical decode")
                    records.append(dict(worker_pid=result["worker_pid"], job_id=jid,
                        worker_entry_wall=result["worker_entry_wall"], worker_complete_wall=result["worker_wall"],
                        acknowledged_wall=time.perf_counter(), numerical_wall_s=result["wall_s"],
                        numerical_cpu_s=result["cpu_s"], valid=True))
                if len(records) < self.workers:
                    time.sleep(.002)
            if len({r["worker_pid"] for r in records}) != self.workers:
                raise RuntimeError("Readiness did not cover all distinct worker PIDs")
            self.readiness = records
            return dict(workers=self.workers, wall_s=time.perf_counter()-begin, acknowledgements=records)
        except BaseException:
            self.close()
            raise

    def _sync_test_gate(self):
        if self._test_gate_flag is not None:
            self._test_gate_flag.value = int(bool(self.test_gate.is_set()))

    def can_admit(self):
        return not self.closed and (len(self.slots) < self.workers or
            any(s["job"] is None for s in self.slots) or len(self.waiting) < self.capacity)

    def submit(self, job):
        if self.job_transform is not None:
            job = self.job_transform(job)
        if self.observer is not None:
            self.observer("job_submit", job.job_id, dict(wall=time.perf_counter(), iq_bytes=len(job.iq)))
        self._sync_test_gate()
        if not self.can_admit():
            return False
        self._replace_dead_idle_slots()
        while len(self.slots) < self.workers:
            self.slots.append(self._new_slot())
        idle = next((s for s in self.slots if s["job"] is None), None)
        if idle is not None:
            self._dispatch(idle, job)
        else:
            self.waiting.append(job)
            self.high_watermark = max(self.high_watermark, len(self.waiting))
        retained = sum(len(j.iq) for j in self.waiting)+sum(s["iq_bytes"] for s in self.slots)
        self.max_retained_iq_bytes = max(self.max_retained_iq_bytes, retained)
        return True

    def try_submit_idle(self, job):
        """Borrow an existing healthy idle worker without queuing or spawning.

        The owner calls this only after serving ready primary payloads. A later
        primary arrival cannot preempt a job already dispatched here.
        """
        if self.closed or self.waiting:
            return False
        idle = next((s for s in self.slots
                     if s["job"] is None and s["process"].is_alive()), None)
        if idle is None:
            return False
        if self.job_transform is not None:
            job = self.job_transform(job)
        if self.observer is not None:
            self.observer("job_submit", job.job_id, dict(wall=time.perf_counter(), iq_bytes=len(job.iq)))
        self._sync_test_gate()
        self._dispatch(idle, job)
        retained = sum(len(j.iq) for j in self.waiting)+sum(s["iq_bytes"] for s in self.slots)
        self.max_retained_iq_bytes = max(self.max_retained_iq_bytes, retained)
        return True

    def _replace_dead_idle_slots(self):
        """Audit 02 B12: drop idle slots whose worker died while idle.

        Without this, submit() could hand a healthy job to a dead worker and
        the next poll() would fail it as WorkerExit. Busy dead slots are left
        to poll(), which recovers or fails their job. Anything a dead idle
        worker still left in its outbox is kept for the next poll().
        """
        survivors = []
        for slot in self.slots:
            if slot["job"] is None and not slot["process"].is_alive():
                self._drain_dead_slot(slot, self._undelivered)
                try:
                    self._dispose(slot, terminate=True)
                except Exception:
                    self.poll_disposal_errors += 1
                    survivors.append(slot)   # keep it for diagnosis; never chosen while dead
                    continue
                self.closed_ipc.update(slot["ipc"])
                self.dead_idle_workers_replaced += 1
            else:
                survivors.append(slot)
        self.slots = survivors

    def _dispatch(self, slot, job):
        if self.observer is not None:
            self.observer("job_dispatch", job.job_id, dict(wall=time.perf_counter(), worker_pid=slot["process"].pid))
        slot["job"], slot["iq_bytes"] = job.job_id, len(job.iq)
        slot["inbox"].put_nowait(job)
        self.max_active = max(self.max_active, sum(s["job"] is not None for s in self.slots))

    DEAD_WORKER_DRAIN_S = 0.5

    def _drain_outbox(self, slot, messages):
        while True:
            try:
                message = slot["outbox"].get_nowait()
            except queue.Empty:
                return
            messages.append(message)
            if message[0] != "started":
                slot["job"], slot["iq_bytes"] = None, 0

    def _drain_dead_slot(self, slot, messages):
        """A dead child's final result may still be in its pipe or reader thread.

        The reader exits at pipe EOF after forwarding everything the child wrote.
        Collect that before synthesizing a failure, bounded by a short deadline.
        """
        had_job = slot["job"]
        deadline = time.monotonic()+self.DEAD_WORKER_DRAIN_S
        reader = slot["reader"]
        while True:
            self._drain_outbox(slot, messages)
            if slot["job"] is None or not reader.is_alive() or time.monotonic() >= deadline:
                break
            reader.join(.01)
        self._drain_outbox(slot, messages)
        if had_job is not None and slot["job"] is None:
            self.dead_worker_results_recovered += 1

    def poll(self):
        self._sync_test_gate()
        messages = self._undelivered
        survivors = []
        for index, slot in enumerate(self.slots):
            self._drain_outbox(slot, messages)
            if not slot["process"].is_alive():
                self._drain_dead_slot(slot, messages)
                if slot["job"] is not None:
                    messages.append(("failed", slot["job"], dict(error_type="WorkerExit",
                        error=f"worker exited with code {slot['process'].exitcode}")))
                    slot["job"], slot["iq_bytes"] = None, 0
                try:
                    self._dispose(slot, terminate=True)
                except Exception:
                    # Keep the undisposed slot for diagnosis/retry and keep every
                    # message already collected; the next poll returns them.
                    self.poll_disposal_errors += 1
                    self.slots = survivors+self.slots[index:]
                    raise
                self.closed_ipc.update(slot['ipc'])
            else:
                survivors.append(slot)
        self.slots = survivors
        # Restart only when work needs a slot. Spawn failure must not escape
        # into acquisition or strand already admitted waiting jobs.
        while self.waiting:
            idle = next((s for s in self.slots if s["job"] is None), None)
            if idle is None and len(self.slots) < self.workers:
                try:
                    idle = self._new_slot()
                    self.slots.append(idle)
                except Exception as exc:
                    job = self.waiting.popleft()
                    messages.append(("failed", job.job_id, dict(error_type=type(exc).__name__,
                        error=str(exc), traceback=traceback.format_exc())))
                    continue
            if idle is None:
                break
            self._dispatch(idle, self.waiting.popleft())
        out = list(messages)
        messages.clear()
        return out

    def _dispose(self, slot, terminate=False, graceful=False):
        self._record_discarded(PayloadPool._shutdown_slots([slot], graceful=graceful and not terminate))

    def _record_discarded(self, discarded):
        for kind, job_id, crc_ok in discarded:
            self.shutdown_discarded[kind] += 1
            if crc_ok:
                self.shutdown_discarded['completed_crc_ok'] += 1
            self.shutdown_discarded_samples.append(dict(kind=kind, job_id=job_id, crc_ok=crc_ok))

    @staticmethod
    def _shutdown_slots(slots, *, graceful=False):
        started = time.monotonic()
        deadline = started + (GRACEFUL_SHUTDOWN_TIMEOUT_S if graceful else SHUTDOWN_TIMEOUT_S)
        remaining = lambda: max(0., deadline-time.monotonic())
        if graceful:
            # Send all idle-worker sentinels before waiting for any one child.
            for slot in slots:
                if slot['job'] is None and slot['process'].is_alive():
                    try:
                        slot['inbox'].put_nowait(None)
                    except queue.Full:
                        pass
            grace_deadline = min(deadline, started+.5)
            for slot in slots:
                if slot['job'] is None:
                    slot['process'].join(max(0., grace_deadline-time.monotonic()))
        # Stop every producer first. Wake mailbox waiters; queued work is
        # discarded, never dispatched after close. Threads retain their handles
        # until their final I/O completes, avoiding close/reused-handle races.
        for slot in slots:
            slot['stopped'].set()
            try:
                slot['inbox'].put_nowait(None)
            except queue.Full:
                pass
        for slot in slots:
            _interrupt_io(slot)
            if slot['process'].is_alive():
                slot['process'].terminate()
        terminate_deadline = min(deadline, time.monotonic()+.5)
        for slot in slots:
            slot['process'].join(max(0., terminate_deadline-time.monotonic()))
        for slot in slots:
            if slot['process'].is_alive():
                slot['process'].kill()
        # Repeat cancellation to cover I/O begun just after the first request.
        # Bounded joins wait for completion, with no arbitrary sleep delay.
        while remaining():
            live = False
            for slot in slots:
                _interrupt_io(slot)
                for thread in (slot['sender'], slot['reader']):
                    if thread.is_alive():
                        thread.join(min(.01, remaining()))
                        live |= thread.is_alive()
                process = slot['process']
                process.join(min(.01, remaining()))
                live |= process.is_alive()
            if not live:
                break
        incomplete = []
        for slot in slots:
            process = slot['process']
            live_threads = [t.name for t in (slot['sender'], slot['reader']) if t.is_alive()]
            if process.is_alive() or live_threads:
                incomplete.append(dict(pid=process.pid, threads=live_threads))
            else:
                _close_connection(slot['input_connection'], slot['io_lock'])
                _close_connection(slot['output_connection'], slot['io_lock'])
        if incomplete:
            # Keep slot references for diagnosis/retry; never report a clean
            # close while daemon threads or child processes remain alive.
            raise RuntimeError(f'Worker shutdown deadline exceeded: {incomplete}')
        discarded = []
        for slot in slots:
            slot['process'].close()
            for mailbox in (slot['inbox'], slot['outbox']):
                while True:
                    try:
                        message = mailbox.get_nowait()
                    except queue.Empty:
                        break
                    # Report (never silently drop) uncollected terminal results.
                    if mailbox is slot['outbox'] and isinstance(message, tuple) and message[0] != 'started':
                        payload = message[2] if len(message) > 2 and isinstance(message[2], dict) else {}
                        packet = payload.get('packet')
                        discarded.append((message[0], message[1], bool(getattr(packet, 'crc_ok', False))))
        return discarded

    def close(self, *, graceful=False):
        if self.closed and not self.slots:
            return
        self.closed = True
        self.waiting.clear()
        self._record_discarded(self._shutdown_slots(self.slots, graceful=graceful))
        for slot in self.slots:
            self.closed_ipc.update(slot['ipc'])
        self.slots.clear()

    def stats(self):
        ipc=self.closed_ipc.copy()
        for slot in self.slots:ipc.update(slot['ipc'])
        return dict(payload_queue_capacity=self.capacity, payload_worker_count=self.workers,
            payload_queue_depth=len(self.waiting), payload_queue_high_watermark=self.high_watermark,
            payload_worker_slots_busy=sum(s["job"] is not None for s in self.slots),
            payload_max_worker_slots_busy=self.max_active,
            payload_max_retained_iq_bytes=self.max_retained_iq_bytes,
            payload_ipc=dict(ipc),payload_process_spawn_owner_wall_s=self.spawn_wall_s,
            payload_undelivered_messages=len(self._undelivered),
            payload_poll_disposal_errors=self.poll_disposal_errors,
            payload_dead_worker_results_recovered=self.dead_worker_results_recovered,
            payload_dead_idle_workers_replaced=self.dead_idle_workers_replaced,
            payload_shutdown_discarded=dict(self.shutdown_discarded),
            payload_shutdown_discarded_samples=list(self.shutdown_discarded_samples))
