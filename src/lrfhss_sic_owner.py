"""Batch 10.4: owner-side scheduling of PHDR successive interference cancellation.

Runs on the IQ owner thread only, like the header pipeline. SIC is optional
work: it never mutates the ring, never takes a PHDR queue slot, and yields to
primary reception.

* Every packet whose payload passes its PHY CRC is rebuilt ("rebuild" jobs
  return small PhdrModels) in two stages (10.4f): copy 0 first, copies 1 and 2
  ranked with the searches of those copies.
* The stream is cut into tiles of victim packet start times [P, P + tile).
  When every packet that could overlap a tile has resolved, header channel 0,
  then 1, then 2 is searched with all models cancelled ("tile" jobs); a
  channel sees the headers the previous channel found.
* Validated headers enter the ordinary TrackManager association (source
  "sic"). A track whose header SIC *established* (a new track, or an existing
  headerless energy track) is SIC-dependent: its payload is admitted only
  within the SIC payload allowance, and it is published only if its payload
  passes the PHY CRC.

Overload control (Batch 10.4b):

* PressureGate: optional SIC work (new rebuild and tile jobs, and payload
  decoding of SIC-dependent packets) is suspended while primary reception is
  under pressure (input lag, PHDR outstanding, payload waiting, acquisition
  lead, candidate backlog; smoothed), and resumes only after every signal has
  stayed below its lower threshold for a hold time.
* Every SIC backlog has a hard size limit and a deadline (its IQ leaving the
  ring): queued rebuilds, open tiles, retained models. Expiry runs on every
  service call, also while the pool is full or SIC is suspended, and every
  skipped item is counted.

Tile scheduling (Batch 10.4d/10.4e/10.4f): earliest search deadline first,
with header copies 1 and 2 deferred behind every tile's copy 0 and second
passes behind first passes (a CPU-starved host spends its SIC time where the
yield is); a tile waits for its first model until its search deadline; model
coverage is per header copy, so a second pass searches only copies with a
model they have not been searched with; whether a tile gets a second pass is
decided after its last search's results are in; reopened tiles respect the
open-tile limit; the EOF flush follows the same deadlines.
"""
from __future__ import annotations

from collections import Counter, deque
import itertools
import time

from lrfhss_sic_jobs import SicJob, SicPool

FS = 1_000_000
STAY_HDR = 233_472
HEADER_COUNT = 3
PACKET_DECODE_S = 2.27        # packet start -> last payload sample (3 PHDRs + 15 fragments + 30 ms)
REBUILD_MARGIN = 16_384       # IQ beyond the three PHDR copies for the fit's timing search
TILE_MARGIN = 300_000         # IQ after a tile channel's victim-start interval (validation window: +290,000)
TILE_LEAD = 40_000            # IQ before it (validation window: -35,000)
OVERLAP = STAY_HDR + 40_000   # packet-start distance within which PHDR copies can overlap
EXPIRY_GUARD = 200_000        # expire work this far before its IQ would leave the ring

SEARCH_MARGIN_S = 1.5         # a search must start this long before its victims' payload deadline
SEARCH_WAIT_S = 1.5           # after a tile is ready, wait at most this long for overlapping packets
MAX_TILE_PASSES = 2           # a tile searched with unresolved overlaps is searched once more
SECOND_PASS_DEFERRAL_S = 3.0  # scheduling rank of a second pass relative to first passes
COPY_DEFERRAL_S = 7.0         # scheduling rank of header copies 1 and 2 relative to copy 0 (10.4f):
                              # longer than a tile's search window (ready ~3.9-5.4 s, deadline 10.5 s),
                              # so every eligible tile's copy 0 ranks ahead of any copy 1 or 2

MAX_QUEUED_REBUILDS = 128     # hard limits (two stages per packet: about 8 s of CRC-valid packets / 12 s of tiles / 32 s of models)
MAX_OPEN_TILES = 12
MAX_REVISIT_TILES = 48        # tiles awaiting a model or a second pass (also bounded by the search deadline)
MAX_MODEL_PACKETS = 256

PRESSURE_SUSPEND = dict(input_lag_s=0.20, header_fill=0.60, payload_waiting=2.0, acquisition_fill=0.40,
                        candidates=96.)
PRESSURE_RESUME = dict(input_lag_s=0.05, header_fill=0.35, payload_waiting=0.5, acquisition_fill=0.25,
                       candidates=48.)
PRESSURE_RESUME_HOLD_S = 2.0
PRESSURE_TIME_CONSTANT_S = 1.0

LATENESS_SUSPEND = dict(input_lag_s=PRESSURE_SUSPEND['input_lag_s'],
                        acquisition_fill=PRESSURE_SUSPEND['acquisition_fill'],
                        payload_waiting=PRESSURE_SUSPEND['payload_waiting'],
                        initial_wait_fraction=.25, due_refinements=96., recent_primary_loss=.5)
LATENESS_RESUME = dict(input_lag_s=PRESSURE_RESUME['input_lag_s'],
                       acquisition_fill=PRESSURE_RESUME['acquisition_fill'],
                       payload_waiting=PRESSURE_RESUME['payload_waiting'],
                       initial_wait_fraction=.10, due_refinements=48., recent_primary_loss=0.)
PRIMARY_LOSS_HOLD_S = 5.
LATENESS_DUTY = (.075, .20)
PRIMARY_LOSS_REASONS = {
    'capacity_rejected': {None},
    'header_candidate_late': {'candidate_age_limit', 'header_iq_unavailable'},
    'header_candidate_rejected': {'candidate_capacity', 'track_capacity'},
    'header_candidate_enqueue_rejected': {'header_queue_capacity'},
    'header_refinement_rejected': {'refinement_capacity'},
    'track_queue_rejected': {'processing_capacity'},
    'track_expired': {'primary_admission_deadline'},
}

# Batch 10.4h: proportional SIC duty below the suspend gate. Full duty while the slowly
# smoothed PHDR outstanding / capacity is at or below the first value, none at the second,
# linear between (Windows: 7H/4P control at 1.25x averages 0.42; SIC 1.10x 0.36 to 0.40).
SIC_DUTY_FILL = (0.42, 0.52)
SIC_DUTY_TIME_CONSTANT_S = 5.0
SIC_DUTY_MAX_COOLDOWN_S = 5.0


class PressureGate:
    """Hysteresis on smoothed primary-receiver pressure signals.

    Suspends when ANY smoothed signal exceeds its suspend threshold (input lag
    is used unsmoothed: it is already an integral). Resumes only when EVERY
    signal has been at or below its resume threshold for ``hold_s`` seconds.
    """
    def __init__(self, suspend=None, resume=None, hold_s=PRESSURE_RESUME_HOLD_S,
                 time_constant_s=PRESSURE_TIME_CONSTANT_S, clock=time.perf_counter, *, lateness=False):
        self.suspend_at = dict(LATENESS_SUSPEND if lateness else PRESSURE_SUSPEND, **(suspend or {}))
        self.resume_at = dict(LATENESS_RESUME if lateness else PRESSURE_RESUME, **(resume or {}))
        if set(self.suspend_at) != set(self.resume_at) or any(
                self.resume_at[k] > self.suspend_at[k] for k in self.suspend_at):
            raise ValueError("resume thresholds must not exceed suspend thresholds")
        self.hold_s, self.tau, self.clock = hold_s, time_constant_s, clock
        self.smoothed = {k: 0. for k in self.suspend_at}
        self.suspended = False
        self.calm_since = None
        self.last = None
        self.suspended_since = None
        self.suspended_s = 0.
        self.transitions = Counter()
        self.reasons = Counter()
        self.peaks = {k: 0. for k in self.suspend_at}

    def _suspend(self, now, reasons):
        self.suspended = True
        self.suspended_since = now
        self.calm_since = None
        self.transitions['suspend'] += 1
        self.reasons.update(reasons)

    def primary_loss(self, now):
        self.peaks['recent_primary_loss'] = self.smoothed['recent_primary_loss'] = 1.
        self.calm_since = None
        if self.suspended:
            return False
        self._suspend(now, ['recent_primary_loss'])
        return True

    def update(self, metrics):
        now = self.clock()
        first = self.last is None
        dt = 0. if first else max(0., now - self.last)
        self.last = now
        alpha = 1. if (first or self.tau <= 0) else min(1., dt/self.tau)
        for key, value in metrics.items():
            value = float(value)
            self.peaks[key] = max(self.peaks[key], value)
            if key in ("input_lag_s", "recent_primary_loss"):
                self.smoothed[key] = value
            else:
                self.smoothed[key] += alpha*(value - self.smoothed[key])
        self.transitions["updates"] += 1
        over = [k for k, v in self.smoothed.items() if v > self.suspend_at[k]]
        if not self.suspended:
            if over:
                self._suspend(now, over)
        else:
            calm = all(v <= self.resume_at[k] for k, v in self.smoothed.items())
            if not calm:
                self.calm_since = None
            elif self.calm_since is None:
                self.calm_since = now
            elif now - self.calm_since >= self.hold_s:
                self.suspended = False
                self.suspended_s += now - self.suspended_since
                self.suspended_since = None
                self.transitions["resume"] += 1
        return self.suspended

    def stats(self):
        running = self.suspended_s + (self.clock() - self.suspended_since if self.suspended else 0.)
        return dict(suspended=self.suspended, suspended_s=running, transitions=dict(self.transitions),
                    suspend_reasons=dict(self.reasons), smoothed=dict(self.smoothed), peaks=dict(self.peaks),
                    suspend_thresholds=dict(self.suspend_at), resume_thresholds=dict(self.resume_at),
                    resume_hold_s=self.hold_s)


class DutyLimiter:
    """Batch 10.4h: proportional SIC duty from primary PHDR load.

    The suspend gate is all-or-nothing: SIC ran flat out until PHDR outstanding reached 60%
    of capacity, was suspended, and ran flat out again after a 2 s calm, so at 1.25x the
    primary spent half the run near the gate limit with a growing backlog (Windows sweeps
    #18 to #21: PHDR outstanding 13.6 to 14.9 while SIC ran, 10.1 to 10.3 for the control).
    Below the gate, SIC now gets a duty fraction from a slow average of the PHDR fill: 1 at or
    below ``fill[0]``, 0 at or above ``fill[1]``. Below full duty, at most one SIC job is
    outstanding and each finished job is followed by a cool-down of wall * (1 - duty) / duty."""
    def __init__(self, fill=SIC_DUTY_FILL, time_constant_s=SIC_DUTY_TIME_CONSTANT_S,
                 max_cooldown_s=SIC_DUTY_MAX_COOLDOWN_S, clock=time.perf_counter):
        self.lo, self.hi = float(fill[0]), float(fill[1])
        if not 0. <= self.lo < self.hi:
            raise ValueError("duty fill thresholds must satisfy 0 <= low < high")
        self.tau, self.max_cooldown, self.clock = time_constant_s, max_cooldown_s, clock
        self.slow = None
        self.duty = 1.
        self.cooldown_until = 0.
        self.last = None
        self.elapsed = self.duty_time = self.throttled_s = self.blocked_s = self.cooldown_s = 0.
        self.min_duty = 1.
        self.peak_slow = 0.

    def update(self, header_fill):
        # Keep the public keyword. The scheduler supplies its configured pressure signal.
        pressure_value = header_fill
        now = self.clock()
        dt = 0. if self.last is None else max(0., now - self.last)
        self.last = now
        if self.slow is None:
            self.slow = float(pressure_value)
        else:
            self.slow += min(1., dt/self.tau if self.tau > 0 else 1.)*(float(pressure_value) - self.slow)
        self.elapsed += dt
        self.duty_time += self.duty*dt
        if self.duty < 1.:
            self.throttled_s += dt
        if self.duty <= 0.:
            self.blocked_s += dt
        self.peak_slow = max(self.peak_slow, self.slow)
        if self.slow <= self.lo:
            self.duty = 1.
        elif self.slow >= self.hi:
            self.duty = 0.
        else:
            self.duty = (self.hi - self.slow)/(self.hi - self.lo)
        self.min_duty = min(self.min_duty, self.duty)
        return self.duty

    def allows(self, outstanding):
        if self.duty >= 1.:
            return True
        if self.duty <= 0.:
            return False
        return outstanding == 0 and self.clock() >= self.cooldown_until

    def job_finished(self, wall_s):
        if self.duty >= 1. or not wall_s or wall_s <= 0:
            return
        d = max(self.duty, .05)
        cool = min(self.max_cooldown, wall_s*(1. - d)/d)
        self.cooldown_until = max(self.cooldown_until, self.clock() + cool)
        self.cooldown_s += cool

    def stats(self):
        return dict(duty=self.duty, mean_duty=(self.duty_time/self.elapsed) if self.elapsed else 1.,
                    min_duty=self.min_duty, throttled_s=self.throttled_s, blocked_s=self.blocked_s,
                    cooldown_s=self.cooldown_s, slow_header_fill=self.slow, peak_slow_header_fill=self.peak_slow,
                    fill_thresholds=(self.lo, self.hi), time_constant_s=self.tau)


class SicScheduler:
    def __init__(self, decoder, *, workers=2, capacity=4, tile_s=1.0, safety_s=2.0, slack_s=0.3,
                 max_inflight=None, payload_allowance=2, pressure=None, options=None, test_gate=None):
        if workers < 1 or capacity < 1 or tile_s <= 0 or safety_s < 0 or payload_allowance < 0:
            raise ValueError("invalid SIC scheduler settings")
        self.d = decoder
        self.pool = SicPool(workers, capacity, test_gate)
        self.tile = int(round(tile_s*FS))
        self.safety = int(round(safety_s*FS))
        self.slack = int(round(slack_s*FS))
        self.max_inflight = workers if max_inflight is None else max(1, int(max_inflight))
        self.payload_allowance = int(payload_allowance)
        pressure = dict(pressure or {})
        mode = pressure.get('mode', 'level')
        if mode not in ('level', 'lateness'):
            raise ValueError('unknown SIC pressure mode')
        self.lateness_pressure = mode == 'lateness'
        self.primary_loss_events = PRIMARY_LOSS_REASONS if self.lateness_pressure else ()
        self.primary_loss_until = float('-inf')
        self.gate = PressureGate(pressure.get("suspend"), pressure.get("resume"),
                                 pressure.get("hold_s", PRESSURE_RESUME_HOLD_S),
                                 pressure.get("time_constant_s", PRESSURE_TIME_CONSTANT_S),
                                 lateness=self.lateness_pressure,
                                 **({"clock": pressure["clock"]} if "clock" in pressure else {}))
        duty = pressure.get("duty", {})            # None disables the proportional duty (10.4h)
        if self.lateness_pressure and duty is not None:
            duty = {'fill': LATENESS_DUTY, **duty}
        self.duty = None if duty is None else DutyLimiter(clock=self.gate.clock, **dict(duty))
        self.options = dict(options or {})
        self.models = {}                  # track_id -> (packet_start, [PhdrModel])
        # Batch 10.4f: a packet is rebuilt in two stages. Stage 0 fits copy 0 (what the
        # copy-0 searches need); stage 1 fits copies 1 and 2 and ranks with the searches
        # of those copies, so a CPU-starved host does not fit models it will never use.
        self.rebuilds = deque()           # (track, stage) awaiting a rebuild job
        self.rebuilding = {}              # track id -> packet start, stage 0 queued or in flight
        self.rebuilding_late = {}         # track id -> packet start, stage 1 queued or in flight
        self.pending = {}                 # job_id -> context
        self.tiles = {}                   # tile start P -> next channel (0..2), absent when done
        self.next_tile = 0
        self.tile_in_flight = set()
        # Batch 10.4e: one state per open tile (keys always a subset of self.tiles):
        #   passes     passes completed that searched at least one channel
        #   covered    {copy index: model track ids that copy was actually searched with}
        #   searched   this pass submitted a search
        #   unresolved during this pass, more overlapping models could still arrive
        self.tile_state = {}
        self.revisit = {}                 # tile P -> state: awaiting its first model or a second pass
        self.eof = False
        self.disabled = None              # reason, once SIC was switched off (e.g. worker priority not lowered)
        self.counts = Counter()
        self.processing = Counter()
        self._sequence = itertools.count(1)
        self._candidate_ids = itertools.count(2_000_000_000)
        import lrfhss_live as phy
        self.hops = phy.hop_frequencies(phy.DEFAULT_GRID, phy.DEFAULT_BW_CODE, phy.DEFAULT_HOPSEQ,
                                        HEADER_COUNT, 0)

    # ------------------------------------------------------------------ hooks
    def on_packet(self, track):
        """A payload passed its PHY CRC: model its three PHDR copies (bounded queue)."""
        tid = track.track_id
        if (self.pool.closed or track.header is None or tid in self.models or tid in self.rebuilding
                or tid in self.rebuilding_late):
            return
        self.rebuilds.append((track, 0))
        self.rebuilds.append((track, 1))
        self.rebuilding[tid] = track.header.start
        self.rebuilding_late[tid] = track.header.start
        self.counts["rebuilds_requested"] += 1
        while len(self.rebuilds) > MAX_QUEUED_REBUILDS:
            self._drop_rebuild(self.rebuilds.popleft(), "dropped_capacity")
        self.counts["max_rebuilds_queued"] = max(self.counts["max_rebuilds_queued"], len(self.rebuilds))

    def _drop_rebuild(self, entry, reason):
        track, stage = entry
        (self.rebuilding if stage == 0 else self.rebuilding_late).pop(track.track_id, None)
        self.counts[("rebuilds_" if stage == 0 else "late_copy_rebuilds_") + reason] += 1

    @staticmethod
    def _rebuild_window(start, stage):
        """IQ a rebuild stage needs: copy 0, or copies 1 and 2, plus the fit's timing margin."""
        if stage == 0:
            return start - REBUILD_MARGIN, start + STAY_HDR + REBUILD_MARGIN
        return start + STAY_HDR - REBUILD_MARGIN, start + HEADER_COUNT*STAY_HDR + REBUILD_MARGIN

    def busy(self):
        """Work remains (used by finish() to keep draining)."""
        if self.pool.closed:
            return False
        return bool(self.pending or self.rebuilds or self.tiles or self.revisit
                    or (self.eof and self.next_tile < self.d.abs_sample))

    @property
    def suspended(self):
        return self.gate.suspended and not self.eof

    def payload_jobs_in_flight(self):
        return sum(1 for c in self.pending.values() if c["kind"] == "payload")

    def payload_slot_free(self):
        """Shared SIC payload allowance and unchanged admission/duty gates."""
        return (not self.pool.closed and not self.suspended and self.pool.can_admit()
                and self.payload_jobs_in_flight() < self.payload_allowance and self._duty_allows())

    def _duty_allows(self):
        """Below the suspend gate, SIC runs at a duty set by primary PHDR load (10.4h)."""
        return self.eof or self.duty is None or self.duty.allows(len(self.pending))

    def submit_payload(self, payload_job, *, primary_pool=None):
        """Use an idle primary slot when offered; otherwise retain the SIC path.

        Both routes consume the same logical SIC payload allowance and duty.
        Exceptions from primary dispatch propagate: an uncertain send must not
        be retried on a second pool under the same identity.
        """
        if not self.payload_slot_free():
            return False
        if primary_pool is not None and primary_pool.try_submit_idle(payload_job):
            self.pending[payload_job.job_id] = dict(kind="payload", primary_pool=True,
                submitted_wall=time.perf_counter(), iq_bytes=len(payload_job.iq))
            self.counts["payload_jobs_submitted"] += 1
            self.counts["payload_jobs_on_primary"] += 1
            return True
        from dataclasses import replace
        job = SicJob(payload_job.job_id, "payload", payload_job.iq, payload_job.dtype, payload_job.base,
                     dict(job=replace(payload_job, iq=b"")))
        try:
            admitted = self.pool.submit(job)
        except Exception as exc:
            self.counts["job_start_failures"] += 1
            self.d._acquisition_event("sic_job_start_failed", self.d.abs_sample, kind="payload", error=repr(exc))
            return False
        if admitted:
            self.pending[job.job_id] = dict(kind="payload", submitted_wall=time.perf_counter(), iq_bytes=len(job.iq))
            self.counts["payload_jobs_submitted"] += 1
        return admitted

    def collect_primary_payload(self, kind, job_id, result):
        """Account once for a borrowed job; the owner already holds its message."""
        context = self.pending.get(job_id)
        if context is None or not context.get("primary_pool") or kind == "started":
            return
        del self.pending[job_id]
        if self.duty is not None:
            self.duty.job_finished(result.get("wall_s", 0.))
        for clock in ("wall_s", "cpu_s"):
            self.processing["payload_"+clock] += result.get(clock, 0.)
            self.processing["primary_payload_"+clock] += result.get(clock, 0.)
        self.counts["payload_jobs_"+kind] += 1
        self.counts["primary_payload_jobs_"+kind] += 1

    # ------------------------------------------------------------------ pressure
    def observe_primary_loss(self, event):
        """Latch primary loss immediately, including between normal owner service calls."""
        allowed = PRIMARY_LOSS_REASONS.get(event['event'])
        if not allowed or event.get('reason') not in allowed or self.pool.closed:
            return False
        if event.get('source') == 'sic' or not self.lateness_pressure:
            return False
        now = self.gate.clock()
        self.primary_loss_until = now + PRIMARY_LOSS_HOLD_S
        self.counts['primary_pressure_loss_events'] += 1
        return self.gate.primary_loss(now)

    def pressure_metrics(self):
        d = self.d
        acq = d.acquisition_process
        metrics = dict(input_lag_s=d.input_lag_s(), payload_waiting=float(len(d.payload_pool.waiting)),
                       acquisition_fill=(acq.outstanding/acq.capacity) if acq is not None else 0.)
        if self.lateness_pressure:
            now = self.gate.clock()
            wait, due = d.header_pipeline.sic_lateness(now) if d.header_pipeline is not None else (0., 0)
            metrics.update(initial_wait_fraction=wait, due_refinements=due, recent_primary_loss=float(now < self.primary_loss_until))
        else:
            hp = d.header_pipeline.pool if d.header_pipeline is not None else None
            metrics.update(header_fill=(len(hp.order)/hp.outstanding_capacity) if hp is not None else 0.,
                           candidates=float(len(d.acquisition.pending) + len(d._refinement_pending)))
        return metrics

    def _update_pressure(self):
        was = self.gate.suspended
        metrics = self.pressure_metrics()
        now = self.gate.update(metrics)
        if self.duty is not None:
            self.duty.update(metrics["initial_wait_fraction" if self.lateness_pressure else "header_fill"])
        if now != was:
            self.d._acquisition_event("sic_suspended" if now else "sic_resumed", self.d.abs_sample,
                                      smoothed=dict(self.gate.smoothed))

    # ------------------------------------------------------------------ snapshots
    def _snapshot(self, lo, hi):
        d = self.d
        lo = max(0, int(lo)); hi = min(int(hi), d.ring.end)
        if hi <= lo:
            return None
        raw = d.ring.raw_snapshot(lo, hi)
        if raw is not None:
            import lrfhss_live as phy
            return lo, raw, phy.U8_IQ_DTYPE
        snap = d.ring.snapshot_bytes(lo, hi)
        return None if snap is None else (lo, snap[0], snap[1])

    def _submit(self, kind, snapshot, params, context):
        base, iq, dtype = snapshot
        job = SicJob(f"{self.d.receiver_id}:s{next(self._sequence)}", kind, iq, dtype, base, params)
        try:
            admitted = self.pool.submit(job)
        except Exception as exc:
            self.counts["job_start_failures"] += 1
            self.d._acquisition_event("sic_job_start_failed", self.d.abs_sample, kind=kind, error=repr(exc))
            return False
        if not admitted:
            return False
        self.pending[job.job_id] = dict(context, kind=kind, submitted_wall=time.perf_counter(), iq_bytes=len(iq))
        self.counts[kind + "_jobs_submitted"] += 1
        self.counts["max_iq_bytes_per_job"] = max(self.counts["max_iq_bytes_per_job"], len(iq))
        return True

    def _can_start(self):
        """Room for a rebuild or tile job (SIC payloads have their own allowance)."""
        others = len(self.pending) - self.payload_jobs_in_flight()
        return (not self.pool.closed and others < self.max_inflight and self.pool.can_admit()
                and self._duty_allows())

    # ------------------------------------------------------------------ service
    def service(self):
        if self.pool.closed:
            return
        self._update_pressure()
        self._open_tiles()
        self._expire()                    # always, also when suspended or the pool is full
        if self.suspended:
            self.counts["service_calls_suspended"] += 1
            return
        self._dispatch()

    # ------------------------------------------------------------------ deadlines
    def payload_deadline(self):
        import lrfhss_live as phy
        return int(phy.SIC_PAYLOAD_DEADLINE_S*FS)

    def search_deadline(self, P):
        """Last sample at which tile P may still be searched: its earliest victim (start P)
        must still reach payload admission before SIC_PAYLOAD_DEADLINE_S (Batch 10.4d)."""
        return P + self.payload_deadline() - int(SEARCH_MARGIN_S*FS)

    def rebuild_deadline(self, start):
        """A rebuild is useful until the last tile it overlaps (P <= start + OVERLAP) is past its search deadline."""
        return self.search_deadline(start + OVERLAP)

    def _rebuild_priority(self, start):
        """Earliest search deadline among the tiles this packet's models feed."""
        return self.search_deadline(start - OVERLAP - self.tile)

    def _ring_start(self):
        d = self.d
        return d.ring.chunks[0][0] if d.ring.chunks else d.abs_sample

    def _eviction_floor(self):
        """Work needing IQ before this sample expires: the ring's eviction point plus a guard.

        Batch 10.4c: measured from the ring's retention horizon (end - keep), not from
        its oldest chunk, so nothing expires near the capture start (where the oldest
        chunk is sample 0 and nothing has been evicted yet)."""
        d = self.d
        return d.ring.end - d.ring.keep + EXPIRY_GUARD

    def _expire(self):
        """Deadlines (IQ about to leave the ring) and hard limits for every SIC backlog."""
        floor = self._eviction_floor()
        # queued rebuilds
        kept = deque()
        for entry in self.rebuilds:
            if self._rebuild_window(entry[0].header.start, entry[1])[0] < floor:
                self._drop_rebuild(entry, "expired")
            else:
                kept.append(entry)
        self.rebuilds = kept
        # open tiles: expire channel by channel (the next channel's window is one
        # PHDR later, so a later header copy may still be recoverable)
        for P in [P for P in self.tiles if P not in self.tile_in_flight]:
            while P in self.tiles and P + self.tiles[P]*STAY_HDR - TILE_LEAD < floor:
                if self.tiles[P] + 1 >= HEADER_COUNT:
                    self._remove_tile(P, "tile_channels_expired")
                else:
                    self.counts["tile_channels_expired"] += 1
                    self.tiles[P] += 1
        for P in [P for P in self.revisit if P + (HEADER_COUNT - 1)*STAY_HDR - TILE_LEAD < floor]:
            del self.revisit[P]
            self.counts["tile_revisits_expired"] += 1
        # searches (and the rebuilds feeding them) that can no longer reach payload admission.
        # Batch 10.4e: also during the EOF flush, where payload admission applies the same clock
        now = self.d.abs_sample
        for P in [P for P in self.tiles if P not in self.tile_in_flight and now > self.search_deadline(P)]:
            self._remove_tile(P, "tile_channels_past_search_deadline")
        for P in [P for P in self.revisit if now > self.search_deadline(P)]:
            del self.revisit[P]
            self.counts["tile_revisits_past_deadline"] += 1
        kept = deque()
        for entry in self.rebuilds:
            if now > self.rebuild_deadline(entry[0].header.start):
                self._drop_rebuild(entry, "past_search_deadline")
            else:
                kept.append(entry)
        self.rebuilds = kept
        while len(self.tiles) > MAX_OPEN_TILES:
            idle = [P for P in self.tiles if P not in self.tile_in_flight]
            if not idle:
                break
            self._remove_tile(min(idle), "tile_channels_dropped_capacity")
        while len(self.revisit) > MAX_REVISIT_TILES:
            del self.revisit[min(self.revisit)]
            self.counts["tile_revisits_dropped_capacity"] += 1
        # retained models: only what an open or future tile can still use
        needed_from = min(list(self.tiles) + list(self.revisit) + [self.next_tile]) - OVERLAP
        needed_from = max(needed_from, floor - EXPIRY_GUARD - OVERLAP)
        for tid in [tid for tid, (start, _) in self.models.items()
                    if start + HEADER_COUNT*STAY_HDR + REBUILD_MARGIN < needed_from]:
            del self.models[tid]
            self.counts["models_released"] += 1
        if len(self.models) > MAX_MODEL_PACKETS:
            for tid in sorted(self.models, key=lambda t: self.models[t][0])[:len(self.models) - MAX_MODEL_PACKETS]:
                del self.models[tid]
                self.counts["models_dropped_capacity"] += 1
        self.counts["max_tiles_open"] = max(self.counts["max_tiles_open"], len(self.tiles))
        self.counts["max_model_packets"] = max(self.counts["max_model_packets"], len(self.models))

    def _submit_rebuild(self, track, stage=0):
        """True when the entry left the queue (submitted or IQ gone), False when the pool refused."""
        h = track.header
        lo, hi = self._rebuild_window(h.start, stage)
        snap = self._snapshot(lo, hi)
        if snap is None or snap[0] > max(0, lo):
            self._drop_rebuild((track, stage), "iq_unavailable")
            return True
        observations = tuple((k, obs.physical_start, obs.frequency_hz) for k, obs in sorted(track.headers.items()))
        params = dict(header_start=int(h.start), header_frequency_hz=float(h.freq_hz), info=dict(h.info),
                      observations=observations, copies=(0,) if stage == 0 else tuple(range(1, HEADER_COUNT)))
        return self._submit("rebuild", snap, params, dict(track_id=track.track_id, packet_start=h.start, stage=stage))

    def _dispatch(self):
        """Earliest search deadline first, across rebuilds and eligible tile channels (Batch 10.4d).

        A rebuild's deadline is the earliest search deadline of the tiles its models
        feed, so a tile near its payload deadline is never starved by rebuilds."""
        self._reopen_tiles()
        # stage 1 (copies 1 and 2) ranks with the searches of those copies (10.4f)
        work = [(self._rebuild_priority(t.header.start) + int(COPY_DEFERRAL_S*FS)*stage, 0, "rebuild", (t, stage))
                for t, stage in self.rebuilds]
        work += [(self._tile_rank(P), 1, "tile", (P, self.tiles[P])) for P in self._eligible_tiles()]
        work.sort(key=lambda w: (w[0], w[1]))
        for _, _, kind, item in work:
            if not self._can_start():
                return
            if kind == "rebuild":
                if self._submit_rebuild(*item):
                    self.rebuilds.remove(item)
            else:
                P, k = item
                if self.tiles.get(P) == k:
                    # only the ranked copy: if it has nothing to search, the tile moves on
                    # and its next copy is ranked on the next call (10.4f)
                    self._submit_tile(P, only=k)

    def _tile_rank(self, P):
        """Scheduling key of tile P's next channel: its search deadline, deferred per header copy
        (10.4f) and per completed pass (10.4d).

        On a CPU-starved host this searches copy 0 of every tile before any copy 1 or 2 (on
        Linux, 210 of 250 SIC finds came from the first copy searched; copies 1 and 2 only find
        what it missed), and first passes before second passes. On an idle host everything is
        searched anyway; each channel still has its tile's search deadline."""
        return (self.search_deadline(P) + int(COPY_DEFERRAL_S*FS)*self.tiles[P]
                + int(SECOND_PASS_DEFERRAL_S*FS)*self._state(P)["passes"])

    def _eligible_tiles(self):
        """Tiles whose victims' data is complete and whose overlapping packets have resolved,
        or whose bounded wait for them (SEARCH_WAIT_S) is over: search with the models available."""
        d = self.d
        eligible = []
        for P in sorted(self.tiles):
            if P in self.tile_in_flight:
                continue
            ready_at = P + self.tile + TILE_MARGIN + int(PACKET_DECODE_S*FS) + self.slack
            if not self.eof and d.abs_sample < ready_at:
                break                                   # later tiles are not ready either
            k = self.tiles[P]
            if k and self._late_rebuild_pending(P):
                continue      # copies 1/2: their models' rebuilds rank ahead of this search (10.4f)
            if self._unresolved(P, k):
                if self.eof or d.abs_sample < ready_at + int(SEARCH_WAIT_S*FS):
                    continue
            eligible.append(P)
        return eligible

    def _late_rebuild_pending(self, P):
        lo, hi = P - OVERLAP, P + self.tile + OVERLAP
        return any(lo <= start < hi for start in self.rebuilding_late.values())

    def _may_gain_models(self, P):
        """More verified models overlapping tile P may still arrive: unresolved packets, SIC-found
        packets whose payload has not completed yet, or a neighbouring tile not yet searched
        (it can find packets that overlap this one). Such a tile gets a second pass."""
        if self._unresolved(P):
            return True
        lo, hi = P - OVERLAP, P + self.tile + OVERLAP
        manager = self.d.track_manager
        for t in list(manager.active_tracks.values()) + list(manager.payload_tracks.values()):
            if getattr(t, "sic_origin", False) and t.header is not None and lo <= t.header.start < hi:
                if t.track_id in manager.active_tracks and self.payload_allowance <= 0:
                    continue      # can never be admitted, so it will never yield a model
                return True
        # only tiles still searching (a tile waiting for its own second pass must not keep
        # its neighbour's revisit alive, or two such tiles would wait on each other forever)
        neighbours = set(self.tiles) | {c["tile"] for c in self.pending.values() if "tile" in c}
        return any(Q != P and P - OVERLAP - self.tile <= Q < P + self.tile + OVERLAP for Q in neighbours)

    def _channel_models(self, P, k):
        """(track id, model) pairs usable on channel k of tile P: copy k models overlapping its window."""
        lo = P + k*STAY_HDR - TILE_LEAD
        hi = P + k*STAY_HDR + self.tile + TILE_MARGIN
        return [(tid, m) for tid, (_, ms) in self.models.items() for m in ms
                if m.copy_index == k and m.body[0] < hi and m.body[1] > lo]

    def _uncovered(self, P, state):
        """Channels with a model that channel has not been searched with yet (per header copy)."""
        return [k for k in range(HEADER_COUNT)
                if {tid for tid, _ in self._channel_models(P, k)} - state["covered"][k]]

    def _reopen_tiles(self):
        """Resume a tile that is awaiting models (Batch 10.4d/10.4e).

        * A tile whose attempt found no model on any channel (passes == 0) is still
          waiting for its first search: it resumes as soon as a model arrives.
        * A tile searched while more models could still arrive gets a second pass once
          they have arrived (nothing overlapping can still gain one), or when its search
          deadline is near.
        Coverage is per header copy, so a model that arrived after copy 0 was searched
        still makes copy 0 eligible. A waiting tile is dropped only when no channel has
        a new model and none can arrive any more."""
        now = self.d.abs_sample
        # 10.4f: admission is bounded by MAX_OPEN_TILES; the earliest search deadline goes first,
        # and a tile that does not fit stays waiting (until its deadline) instead of being
        # reopened and then discarded by the capacity limit
        for P, state in sorted(self.revisit.items()):
            new = self._uncovered(P, state)
            pending = self._may_gain_models(P)
            if not new:
                if not pending:
                    del self.revisit[P]
                    self.counts["tile_second_passes_not_needed" if state["passes"] else
                                "tiles_without_models"] += 1
                continue
            late = now >= self.search_deadline(P) - int(1.5*FS)
            if state["passes"] and pending and not late and not self.eof:
                continue
            if len(self.tiles) >= MAX_OPEN_TILES:
                self.counts["tile_reopens_deferred_capacity"] += 1
                continue
            del self.revisit[P]
            state.update(searched=False, unresolved=False)
            self.tiles[P] = min(new)          # channels before the first one with a new model have nothing to add
            self.tile_state[P] = state
            self.counts["tile_second_passes" if state["passes"] else "tile_first_passes_resumed"] += 1
        self.counts["max_tiles_open"] = max(self.counts["max_tiles_open"], len(self.tiles))

    def _open_tiles(self):
        d = self.d
        while True:
            P = self.next_tile
            if self.eof:
                if P >= d.abs_sample:
                    return
            elif d.abs_sample < P + self.tile:
                return
            self.tiles[P] = 0
            self.next_tile = P + self.tile

    def _unresolved(self, P, k=None):
        """A packet that could overlap tile P is still awaiting its payload result or rebuild
        (for header copy k: its copy-k model; None: any copy)."""
        lo, hi = P - OVERLAP, P + self.tile + OVERLAP
        manager = self.d.track_manager
        for t in list(manager.active_tracks.values()) + list(manager.payload_tracks.values()):
            if (getattr(t, "sic_origin", False) and t.track_id in manager.active_tracks
                    and t.header is not None and self.d.abs_sample >= t.expected_end_sample):
                continue          # a SIC packet waiting for payload admission is not about to resolve
            if self.eof and t.track_id in manager.active_tracks and self.d.abs_sample < t.expected_end_sample:
                continue          # needs samples after the end of the stream: it can never resolve (10.4e)
            start = t.header.start if t.header is not None else t.detected_sample + 8704
            if lo <= start < hi:
                return True
        # header candidates still awaiting validation (a fast offline feed runs ahead of them).
        # At EOF the header phase has already ended (finish() sets eof after it): a candidate
        # left waiting (e.g. a refinement for a header copy that never arrives) can no longer
        # resolve and must not hold the flush until the drain timeout (10.4h shutdown check).
        if self.eof:
            return any(lo <= start < hi for start in self.rebuilding.values()) or (
                (k is None or k >= 1) and any(lo <= start < hi for start in self.rebuilding_late.values()))
        pipeline = self.d.header_pipeline
        candidates = list(self.d.acquisition.pending) + list(self.d._refinement_pending)
        if pipeline is not None:
            candidates += [v["candidate"] for v in pipeline.pending.values() if v["candidate"] is not None]
            candidates += [e["candidate"] for e in pipeline.parked.values()] + list(pipeline.released)
        for c in candidates:
            if lo <= c.header_start - c.physical_index*STAY_HDR < hi:
                return True
        starts = list(self.rebuilding.values())
        if k is None or k >= 1:
            starts += list(self.rebuilding_late.values())
        return any(lo <= start < hi for start in starts)

    def _state(self, P):
        if P not in self.tile_state:
            self.tile_state[P] = self._blank_state()
        return self.tile_state[P]

    def _remove_tile(self, P, count_key):
        """The one way an open tile is dropped (expiry, deadline, capacity): its state goes with it."""
        self.counts[count_key] += HEADER_COUNT - self.tiles.pop(P)
        self.tile_state.pop(P, None)

    def _submit_tile(self, P, only=None):
        """Submit the next channel of tile P with a model it has not been searched with; True when submitted.

        ``only``: submit only that channel (skipping it, when it has nothing new, is allowed);
        a later channel waits to be ranked on its own (10.4f)."""
        d = self.d
        while P in self.tiles:
            k = self.tiles[P]
            if only is not None and k != only:
                return False
            lo_start = P + k*STAY_HDR
            hi_start = lo_start + self.tile
            lo, hi = lo_start - TILE_LEAD, hi_start + TILE_MARGIN
            state = self._state(P)
            if self._may_gain_models(P):
                state["unresolved"] = True     # searched or skipped while more models could arrive
            pairs = self._channel_models(P, k)
            tids = {tid for tid, _ in pairs}
            if not tids - state["covered"][k]:
                self.counts["tile_channels_without_models" if not tids else "tile_channels_already_covered"] += 1
                self._advance(P)
                continue
            models = [m for _, m in pairs]
            snap = self._snapshot(lo, hi)
            if snap is None or snap[0] > max(0, lo_start - 40_000):
                self.counts["tile_channels_iq_unavailable"] += 1
                self._advance(P)
                continue
            known = []
            for t in d.track_manager._known():
                if t.header is None or t.cfo_hz is None or len(t.expected_hops_hz) <= k:
                    continue
                s = t.header.start + k*STAY_HDR
                if lo_start - 4096 <= s < hi_start + 4096:
                    known.append((int(s), float(t.expected_hops_hz[k] + t.cfo_hz)))
            params = dict(copy_index=k, nominal_hz=float(self.hops[k]), models=models, known=tuple(known),
                          lo_start=int(lo_start), hi_start=int(hi_start), batch_header=d.batch_header,
                          frontend=d.phdr_frontend, decimation=d.phdr_decimation, soft_trials=d.dsp_soft_trials,
                          options=self.options)
            if not self._submit("tile", snap, params, dict(tile=P, copy_index=k, model_tracks=sorted(tids))):
                return False
            self.tile_in_flight.add(P)
            state["covered"][k] |= tids        # exactly the models this job cancels
            state["searched"] = True
            return True
        return False

    def _advance(self, P):
        """Next channel; at the end of a pass keep the tile waiting when models may still arrive."""
        k = self.tiles[P] + 1
        if k < HEADER_COUNT:
            self.tiles[P] = k
            return
        del self.tiles[P]
        state = self.tile_state.pop(P, None) or self._blank_state()
        # 10.4f: judged after this pass's own results were incorporated. A packet the final
        # copy just found (its model comes after its payload), or a model that arrived during
        # the pass, keeps the tile for a second pass; the pass limit still applies.
        if self._may_gain_models(P) or self._uncovered(P, state):
            state["unresolved"] = True
        if state["searched"]:
            state["passes"] += 1
            self.counts["tiles_completed"] += 1
        else:
            self.counts["tile_passes_without_search"] += 1
        # an attempt that searched nothing never completes a tile (Batch 10.4e): it waits
        # for its first model until the search deadline, like a tile awaiting a second pass
        if state["unresolved"] and state["passes"] < MAX_TILE_PASSES:
            self.revisit[P] = state

    def _blank_state(self):
        return dict(passes=0, covered={k: set() for k in range(HEADER_COUNT)}, searched=False, unresolved=False)

    # ------------------------------------------------------------------ results
    def collect(self):
        d = self.d
        if self.pool.closed:
            return
        for kind, job_id, r in self.pool.poll():
            context = self.pending.get(job_id)
            if kind != "started" and "priority_lowered" in r:
                # every job kind, payloads included, is checked before it is used (10.4d)
                self.counts["job_results_priority_lowered" if r["priority_lowered"] else
                            "job_results_priority_normal"] += 1
                if not r["priority_lowered"]:
                    self.pending.pop(job_id, None)
                    self.disable("worker_priority_not_lowered")
                    return
            if kind != "started" and self.duty is not None:
                self.duty.job_finished(r.get("wall_s", 0.))
            if context is not None and context["kind"] == "payload":
                # SIC-dependent payloads: the decoder's ordinary result path
                d._payload_inbox.append((kind, job_id, r))
                if kind != "started":
                    del self.pending[job_id]
                    self.processing["payload_wall_s"] += r.get("wall_s", 0.)
                    self.processing["payload_cpu_s"] += r.get("cpu_s", 0.)
                    self.counts["payload_jobs_" + kind] += 1
                continue
            if kind == "started":
                continue
            context = self.pending.pop(job_id, None)
            if context is None:
                continue
            self.processing[context["kind"] + "_wall_s"] += r.get("wall_s", 0.)
            self.processing[context["kind"] + "_cpu_s"] += r.get("cpu_s", 0.)
            self.processing["worker_initialization_cpu_s"] += r.get("initialization_cpu_s", 0.)
            if context["kind"] == "rebuild":
                stage = context.get("stage", 0)
                (self.rebuilding if stage == 0 else self.rebuilding_late).pop(context["track_id"], None)
                if kind != "completed":
                    self.counts["rebuild_jobs_failed"] += 1
                    d._acquisition_event("sic_job_failed", d.abs_sample, job_id=job_id, **{k: v for k, v in r.items() if k in ("error_type", "error")})
                    continue
                self.counts["rebuild_jobs_completed"] += 1
                self.counts["late_copy_rebuild_jobs_completed" if stage else "copy0_rebuild_jobs_completed"] += 1
                self.counts["copies_modelled"] += len(r["models"])
                start, held = self.models.get(context["track_id"], (context["packet_start"], []))
                self.models[context["track_id"]] = (start, list(held) + list(r["models"]))
                continue
            P = context["tile"]
            self.tile_in_flight.discard(P)
            if kind != "completed":
                self.counts["tile_jobs_failed"] += 1
                d._acquisition_event("sic_job_failed", d.abs_sample, job_id=job_id, **{k: v for k, v in r.items() if k in ("error_type", "error")})
            else:
                self.counts["tile_jobs_completed"] += 1
                for key, value in r["stats"].items():
                    if isinstance(value, (int, float)):
                        self.processing["tile_search_" + key] += value
                for v in r["validated"]:
                    self._accept(v, job_id)
            if P in self.tiles:
                self._advance(P)

    def _accept(self, v, job_id):
        """Associate a SIC-validated header. SIC dependence = SIC established the track's header."""
        d = self.d
        manager = d.track_manager
        created_before = manager.counts["tracks_created"]
        accepted_before = manager.counts["headers_accepted"]
        candidate_id = next(self._candidate_ids)
        track = d._accept_physical(v["header"], v["physical_index"], candidate_id=candidate_id, source="sic",
                                   score=v["sync_correlation"], energy=v["energy_ratio"])
        created = manager.counts["tracks_created"] > created_before
        established = track is not None and manager.counts["headers_accepted"] > accepted_before
        if established:
            track.sic_origin = True          # depends on SIC (cleared if the primary path validates it)
            track.sic_discovered = True      # provenance, never cleared
            self.counts["headers_established"] += 1
            self.counts["new_tracks" if created else "headerless_tracks_established"] += 1
        else:
            self.counts["headers_associated_existing" if track is not None else "headers_not_associated"] += 1
        beyond_eof = bool(self.eof and track is not None and track.track_id in manager.active_tracks
                          and track.expected_end_sample > d.abs_sample)
        if beyond_eof:
            # found during the EOF flush but its payload ends after the last sample: it can
            # never complete, so it is retired at once instead of holding the drain (10.4e)
            manager.retire(track, "expired", d.abs_sample, "end_of_stream", stage="sic_eof")
            self.counts["tracks_retired_beyond_eof"] += 1
        self.counts["headers_validated"] += 1
        d._acquisition_event("sic_header_validated", d.abs_sample, job_id=job_id,
                             physical_header=v["physical_index"]+1, candidate_sample=v["candidate_start"],
                             frequency_hz=v["header"].freq_hz, sync_correlation=v["sync_correlation"],
                             isr=v["isr"], new_track=bool(created), header_established=bool(established),
                             track_id=getattr(track, "track_id", None), retired_beyond_eof=beyond_eof)

    # ------------------------------------------------------------------ lifetime
    def warmup(self, timeout=60., stop_event=None):
        """Spawn every worker and run a real rebuild on each before RF admission."""
        import numpy as np
        import lrfhss_sic as sic
        info = dict(payload_len=55, modulation=0, cr=1, grid=1, hop=1, bw_code=6, hop_seq_id=256, sync_index=2)
        n = HEADER_COUNT*STAY_HDR + 2*REBUILD_MARGIN
        start = REBUILD_MARGIN
        copies = sic.packet_copies(start, float(self.hops[0]), info)
        t = np.arange(n, dtype=float)
        z = np.zeros(n, dtype=np.complex64)
        for k, (c, f) in enumerate(copies):
            model = sic.PhdrModel(k, float(c), sic.phdr_symbols(info, k).astype(np.int8), f, c,
                                  np.array([c - 2048., c + 240000.]), np.array([0.3+0j, 0.3+0j]), sic.DEFAULT_BT)
            z += sic.synthesize(model, t).astype(np.complex64)
        began = time.perf_counter()
        while len(self.pool.slots) < self.pool.workers:
            self.pool.slots.append(self.pool._new_slot())
        expected = {}
        for i, slot in enumerate(self.pool.slots):
            job = SicJob(f"warmup:sic:{i}", "rebuild", z.tobytes(), z.dtype.str, 0,
                         dict(header_start=start, header_frequency_hz=float(self.hops[0]), info=info))
            expected[job.job_id] = slot["process"].pid
            self.pool._dispatch(slot, job)
        records = []
        try:
            while len(records) < len(expected):
                if stop_event is not None and stop_event.is_set():
                    raise InterruptedError("SIC worker readiness cancelled")
                if time.perf_counter() - began > timeout:
                    raise TimeoutError("Not every SIC worker acknowledged readiness")
                for kind, jid, r in self.pool.poll():
                    if kind == "started":
                        continue
                    if kind != "completed" or len(r.get("models", ())) != HEADER_COUNT or r.get("worker_pid") != expected.get(jid):
                        raise RuntimeError(f"SIC worker warmup failed: {jid}: {r.get('error', r)}")
                    records.append(dict(worker_pid=r["worker_pid"], wall_s=r["wall_s"], cpu_s=r["cpu_s"],
                                        priority_lowered=r.get("priority_lowered"),
                                        priority_detail=r.get("priority_detail")))
                time.sleep(.002)
        except BaseException:
            self.pool.close()
            raise
        self.counts["workers_priority_lowered"] = sum(bool(r["priority_lowered"]) for r in records)
        qualified = all(r["priority_lowered"] for r in records)
        if not qualified:
            # SIC must only use CPU the primary receiver leaves idle; without the
            # lowered priority it is switched off (the receiver is BATCH10 without SIC).
            self.disable("worker_priority_not_lowered")
        return dict(workers=self.pool.workers, wall_s=time.perf_counter()-began, acknowledgements=records,
                    qualified=qualified, disabled=self.disabled)

    def disable(self, reason):
        """Switch SIC off for the rest of the run; the receiver continues without it."""
        if self.disabled is None:
            self.disabled = reason
            self.counts["disabled"] = 1
            self.d._acquisition_event("sic_disabled", self.d.abs_sample, reason=reason)
        self.d._abandon_sic_payloads("sic_disabled", handoff=True)   # confirmed packets move to the primary pool (10.4k)
        self.close("sic_disabled")

    def close(self, reason):
        if self.pool.closed and not self.pending:
            return
        # EOF/disable may stop optional SIC workers while an admitted borrowed
        # payload is still running on a primary worker. Its owner drains it.
        borrowed = {j: c for j, c in self.pending.items()
                    if c.get("primary_pool") and j in self.d._payload_pending}
        cancelled = len(self.pending)-len(borrowed)
        self.pool.close(graceful=(reason == "finished"))
        self.counts["jobs_cancelled"] += cancelled
        self.counts["tiles_abandoned"] += len(self.tiles)
        self.counts["rebuilds_abandoned"] += len(self.rebuilds)
        self.pending.clear(); self.pending.update(borrowed)
        self.rebuilds.clear(); self.rebuilding.clear(); self.rebuilding_late.clear()
        self.tiles.clear(); self.tile_in_flight.clear(); self.models.clear()
        self.tile_state.clear(); self.revisit.clear()

    def stats(self):
        pool = {k.replace("payload_", "sic_"): v for k, v in self.pool.stats().items()}
        duty = None if self.duty is None else self.duty.stats()
        if self.lateness_pressure and duty is not None:
            duty['slow_initial_wait_fraction'] = duty.pop('slow_header_fill')
            duty['peak_slow_initial_wait_fraction'] = duty.pop('peak_slow_header_fill')
            duty['initial_wait_thresholds'] = duty.pop('fill_thresholds')
        return dict(counts=dict(self.counts), processing=dict(self.processing), pool=pool, disabled=self.disabled,
                    pressure=self.gate.stats(), duty=duty,
                    models_held=sum(len(ms) for _, ms in self.models.values()),
                    model_packets=len(self.models), rebuilds_queued=len(self.rebuilds),
                    tiles_open=len(self.tiles), tiles_awaiting_second_pass=len(self.revisit),
                    jobs_outstanding=len(self.pending),
                    max_inflight=self.max_inflight, payload_allowance=self.payload_allowance,
                    tile_s=self.tile/FS, options=self.options)
