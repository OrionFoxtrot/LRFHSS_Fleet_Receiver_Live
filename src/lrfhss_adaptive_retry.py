"""Schedule live payload retries on spare workers, one offset at a time."""
from collections import Counter, deque
from dataclasses import dataclass, replace

from lrfhss_payload_jobs import PayloadJob, PayloadPool


@dataclass
class DeferredPayload:
    job: PayloadJob
    original_result: dict
    remaining: list[tuple[int, float]]
    trials: int = 0
    rejected_app_crc: int = 0


class AdaptivePayloadPool(PayloadPool):
    """Expose one logical completion per packet, including when retries exhaust."""

    MAX_DEFERRED = 32
    MAX_DEFERRED_IQ_BYTES = 128 * 1024 * 1024

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_pass = {}
        self.deferred = deque()
        self.active_retry = None
        self.last_successful_offset = None
        self.deferred_bytes = 0
        self.counts = Counter()

    def _admit_first_pass(self, job, admit):
        if not job.payload_retry:
            return admit(job)
        admitted = admit(replace(job, payload_retry=()))
        if admitted and self.workers >= 2:
            self.first_pass[job.job_id] = job
        return admitted

    def submit(self, job):
        return self._admit_first_pass(job, super().submit)

    def try_submit_idle(self, job):
        return self._admit_first_pass(job, super().try_submit_idle)

    def service_retries(self):
        """Called by the owner after admitting ready packets. Reserve one idle slot."""
        if not self.deferred or self.active_retry or self.waiting or self.closed:
            return
        if sum(s["job"] is None and s["process"].is_alive() for s in self.slots) < 2:
            return
        pending = self.deferred[0]
        offset = self.last_successful_offset
        if offset not in pending.remaining:
            offset = pending.remaining[0]
        dt, df = offset
        job = pending.job
        trial_id = f"{job.job_id}:adaptive:{pending.trials + 1}"
        trial = replace(job, job_id=trial_id, header_start=job.header_start + dt,
                        frequency_hz=job.frequency_hz + df, payload_retry=())
        if super().try_submit_idle(trial):
            pending.remaining.remove(offset)
            pending.trials += 1
            self.active_retry = (trial_id, self.deferred.popleft(), offset)
            self.counts["trials_started"] += 1

    def _complete_trial(self, kind, result):
        from lrfhss_live import _descent_application_crc_ok
        _, pending, offset = self.active_retry
        self.active_retry = None
        job = pending.job
        original = pending.original_result
        for key in ("cpu_s", "wall_s"):
            original[key] = original.get(key, 0.) + result.get(key, 0.)
        packet = result.get("packet")
        valid = packet is not None and packet.crc_ok
        if valid and job.retry_app_crc and not _descent_application_crc_ok(packet.payload):
            pending.rejected_app_crc += 1
            valid = False
        if valid:
            # Report the original RF coordinates, as with inline retry_payload().
            dt, _ = offset
            packet.header.start = job.header_start
            packet.header.freq_hz = job.frequency_hz
            packet.payload_start -= dt
            original["packet"] = packet
            original["detail"] = result["detail"]
            self.last_successful_offset = offset
            self.counts["rescued"] += 1
        elif kind == "failed":
            self.counts["trial_errors"] += 1
        elif pending.remaining:
            self.deferred.append(pending)
            return None
        else:
            self.counts["exhausted"] += 1
        original["detail"]["payload_retry"] = dict(mode="adaptive", trials=pending.trials,
            accepted=list(offset) if valid else None, rejected_app_crc=pending.rejected_app_crc)
        self.deferred_bytes -= len(job.iq)
        return "completed", job.job_id, original

    def poll(self):
        messages = []
        for kind, job_id, result in super().poll():
            if self.active_retry and job_id == self.active_retry[0]:
                if kind != "started":
                    terminal = self._complete_trial(kind, result)
                    if terminal is not None:
                        messages.append(terminal)
                continue
            job = None if kind == "started" else self.first_pass.pop(job_id, None)
            packet = result.get("packet")
            if job and kind == "completed" and (packet is None or not packet.crc_ok):
                depth = len(self.deferred) + int(self.active_retry is not None)
                if (depth < self.MAX_DEFERRED and
                        self.deferred_bytes + len(job.iq) <= self.MAX_DEFERRED_IQ_BYTES):
                    self.deferred.append(DeferredPayload(job, result, list(job.payload_retry)))
                    self.deferred_bytes += len(job.iq)
                    self.counts["deferred"] += 1
                    self.counts["high_watermark"] = max(self.counts["high_watermark"], depth + 1)
                    messages.append(("deferred", job_id, {}))
                    continue
                self.counts["capacity_skips"] += 1
            messages.append((kind, job_id, result))
        return messages

    def retry_stats(self):
        return dict(self.counts, waiting=len(self.deferred), active=int(self.active_retry is not None),
                    retained_iq_bytes=self.deferred_bytes)

    def stats(self):
        return dict(super().stats(), adaptive_retry=self.retry_stats())

    def close(self, *, graceful=False):
        try:
            super().close(graceful=graceful)
        finally:
            self.counts["abandoned"] += len(self.deferred) + int(self.active_retry is not None)
            self.first_pass.clear()
            self.deferred.clear()
            self.active_retry = None
            self.deferred_bytes = 0
