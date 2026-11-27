"""Batch 10.4: SIC worker processes (PHDR rebuild and cancelled-channel search).

Separate from the PHDR and payload pools so SIC can never take a queue slot
from primary work, and started at below-normal OS priority so it only uses CPU
the primary path leaves idle. Transport and lifetime reuse PayloadPool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
import traceback

from lrfhss_payload_jobs import PayloadPool

SIC_JOB_KINDS = ("rebuild", "tile", "payload")


@dataclass(frozen=True)
class SicJob:
    job_id: str
    kind: str                      # "rebuild" | "tile"
    iq: bytes                      # immutable snapshot (u8 I/Q or complex64 bytes)
    dtype: str                     # "u8iq" or a numpy dtype string
    base: int                      # absolute sample of iq[0]
    params: dict = field(default_factory=dict)
    diagnostic_noop: bool = False


def lower_priority():
    """Best effort: below-normal scheduling priority for this worker process.

    Returns (lowered, detail). Windows: psutil when installed, otherwise
    SetPriorityClass with explicit HANDLE types (the 10.4 version passed
    GetCurrentProcess()'s pseudo-handle through ctypes' default int and failed
    silently on 64-bit Windows, so the workers ran at normal priority)."""
    try:
        if os.name == "nt":
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            try:
                import psutil
                proc = psutil.Process()
                proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                return proc.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS, "psutil"
            except ImportError:
                pass
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            k32.SetPriorityClass.restype = wintypes.BOOL
            k32.GetPriorityClass.argtypes = (wintypes.HANDLE,)
            k32.GetPriorityClass.restype = wintypes.DWORD
            handle = k32.GetCurrentProcess()
            if not k32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS):
                return False, f"SetPriorityClass error {ctypes.get_last_error()}"
            return k32.GetPriorityClass(handle) == BELOW_NORMAL_PRIORITY_CLASS, "SetPriorityClass"
        os.nice(10)
        return os.nice(0) >= 10, "nice"
    except Exception as exc:
        return False, repr(exc)


def _json_safe(value):
    """Plain JSON value (strict: no NaN) for a descriptor field."""
    import dataclasses
    import math
    import numpy as np
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_safe(getattr(value, f.name)) for f in dataclasses.fields(value)
                if f.name not in ("iq", "diagnostic_noop")}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        return [_json_safe(value.real), _json_safe(value.imag)]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return dict(bytes=len(value))
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return repr(value)


def describe_job(job):
    """Batch 10.4e: JSON-serializable descriptor fields of a SIC job (diagnostics).

    Tile jobs carry PhdrModel objects and payload jobs a PayloadJob; both are
    expanded field by field (a payload job's descriptor matches a primary one)."""
    fields = dict(kind=job.kind, dtype=job.dtype, base=int(job.base))
    params = dict(job.params)
    if job.kind == "payload" and "job" in params:
        fields["payload_job"] = _json_safe(params.pop("job"))
    fields["params"] = _json_safe(params)
    return fields


def _array(job):
    import numpy as np
    import lrfhss_live as phy
    if job.dtype == phy.U8_IQ_DTYPE:
        return phy.u8_iq_to_complex_fast(job.iq)
    return np.frombuffer(job.iq, dtype=job.dtype)


def run_job(job):
    """Execute one SIC job in the current process (the worker body; also used by tests)."""
    import lrfhss_sic as sic
    p = job.params
    if job.kind == "payload":
        z = None
    else:
        z = _array(job)
    if job.kind == "rebuild":
        models = sic.rebuild_packet(z, job.base, p["header_start"], p["header_frequency_hz"], p["info"],
                                    p.get("observations", ()), bt=p.get("bt", sic.DEFAULT_BT),
                                    copies=p.get("copies"))
        return dict(models=models)
    if job.kind == "tile":
        import lrfhss_dsp
        lrfhss_dsp.set_soft_trials(p.get("soft_trials", "reference"))
        validated, stats = sic.search(z, job.base, p["copy_index"], p["nominal_hz"], p["models"], p.get("known", ()),
                                      lo_start=p["lo_start"], hi_start=p["hi_start"],
                                      batch_header=p.get("batch_header", True), frontend=p.get("frontend", "screened"),
                                      decimation=p.get("decimation", 64), candidate_id_base=p.get("candidate_id_base", 0),
                                      **p.get("options", {}))
        return dict(validated=validated, stats=stats)
    if job.kind == "payload":
        # Batch 10.4c: a SIC-dependent packet's payload decodes here, isolated from
        # the primary payload pool (the unchanged payload decoder, retry included).
        from dataclasses import replace
        from lrfhss_payload_jobs import execute_job
        packet, detail = execute_job(replace(p["job"], iq=job.iq))
        return dict(packet=packet, detail=detail)
    raise ValueError(f"unknown SIC job kind {job.kind!r}")


def _sic_worker(inbox, outbox, test_gate_flag=None):
    from multiprocessing.reduction import ForkingPickler
    priority_lowered, priority_detail = lower_priority()
    init_wall, init_cpu = time.perf_counter(), time.process_time()
    import numpy  # noqa: F401  (import cost counted as initialization)
    import lrfhss_sic  # noqa: F401
    init_wall, init_cpu = time.perf_counter()-init_wall, time.process_time()-init_cpu
    while True:
        wire = inbox.recv_bytes()
        job = ForkingPickler.loads(wire)
        del wire
        if job is None:
            return
        started, cpu = time.perf_counter(), time.process_time()
        outbox.send(("started", job.job_id, dict(worker_wall=started, worker_pid=os.getpid())))
        try:
            if test_gate_flag is not None:
                while not test_gate_flag.value:
                    time.sleep(.005)
            result = dict(diagnostic_only=True) if job.diagnostic_noop else run_job(job)
            kind = "completed"
        except Exception as exc:
            result = dict(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            kind = "failed"
        result.update(worker_pid=os.getpid(), worker_entry_wall=started, worker_wall=time.perf_counter(),
                      wall_s=time.perf_counter()-started, cpu_s=time.process_time()-cpu, job_kind=job.kind,
                      priority_lowered=priority_lowered, priority_detail=priority_detail,
                      initialization_wall_s=init_wall,
                      initialization_cpu_s=init_cpu)
        init_wall = init_cpu = 0.
        outbox.send((kind, job.job_id, result))
        del job, result


class SicPool(PayloadPool):
    def __init__(self, workers=2, capacity=4, test_gate=None):
        super().__init__(workers, capacity, test_gate, worker_target=_sic_worker, thread_prefix="sic")

    @property
    def outstanding(self):
        return sum(s["job"] is not None for s in self.slots) + len(self.waiting)

    @property
    def idle(self):
        return not self.closed and self.outstanding < self.workers
