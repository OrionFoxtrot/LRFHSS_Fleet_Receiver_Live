"""File-only backpressure and the existing CRC-gated failed-payload retry grid.

Settle runnable owner/worker work at each 50 ms input boundary. Work needing
future samples is left alone. A 2 s guard (or 25% of a shorter horizon) protects
both candidate age and the ring's IQ horizon. No recording is buffered wholesale.
Header eligibility and validation stay intact; failed payloads may use the
ordinary validated-header alternatives and fixed 19-offset retry grid.
"""
from __future__ import annotations

import time

FS = 1_000_000


class ExhaustiveFile:
    def __init__(self, decoder, *, warn=print, watchdog_s=120., clock=time.monotonic,
                 sleep=time.sleep):
        if watchdog_s <= 0:
            raise ValueError('Exhaustive watchdog must be positive')
        self.decoder, self.warn = decoder, warn
        self.watchdog_s, self.clock, self.sleep = float(watchdog_s), clock, sleep
        self.started = clock()
        self.wait_wall_s = 0.
        self.wait_count = self.watchdog_trips = 0
        self.max_candidate_age_s = 0.
        self.min_iq_margin_s = None
        self._waiting_since = None
        self._last_progress = self.started
        self._last_signature = None
        horizon = min(decoder.ring.keep / FS,
                      decoder.header_pipeline.max_age_s if decoder.header_pipeline is not None else decoder.ring.keep / FS)
        self.margin_s = min(2., horizon / 4)
        self._horizon = horizon
        # The only admission difference is to defer a complete primary payload
        # on a full pool, keeping its ordinary job/validation for the next service.
        decoder._exhaustive_file = True

    def _candidates(self):
        d, hp = self.decoder, self.decoder.header_pipeline
        candidates = list(d.acquisition.pending) + list(d._refinement_pending)
        if hp is not None:
            candidates += list(hp.released)
            candidates += [entry['candidate'] for entry in hp.parked.values()]
        return candidates

    def state(self, next_samples=50_000):
        """Read owner state; distinguish runnable backlog from future-IQ waits."""
        d, hp = self.decoder, self.decoder.header_pipeline
        now = d.abs_sample
        acquisition = d.acquisition_process
        pending = list(d.acquisition.pending)
        refinements = list(d._refinement_pending)
        tracks = list(d.track_manager.active_tracks.values())
        candidates = self._candidates()
        ages = [(now-c.observed_sample)/FS for c in candidates]
        oldest = max(ages, default=0.)
        self.max_candidate_age_s = max(self.max_candidate_age_s, oldest)
        anchors = [max(0,c.header_start-35000) for c in candidates
                   if getattr(c,'transaction_snapshot',None) is None]
        anchors += [max(0,t.detected_sample-50000) for t in tracks if t.header is None]
        anchors += [max(0,t.header.start) for t in tracks if t.header is not None]
        # ring.end, not abs_sample, accounts for uncommitted acquisition input.
        margin = (min(anchors) - (d.ring.end + next_samples - d.ring.keep))/FS if anchors else self._horizon
        self.min_iq_margin_s = margin if self.min_iq_margin_s is None else min(self.min_iq_margin_s,margin)
        reasons=[]
        if acquisition is not None and acquisition.outstanding: reasons.append('acquisition')
        if hp is not None and hp.pending: reasons.append('header_workers')
        if d._payload_pending: reasons.append('payload_workers')
        if any(now >= c.header_start+335000 for c in pending): reasons.append('ready_headers')
        if hp is not None:
            if hp.released: reasons.append('released_headers')
            if refinements:
                index=hp._refinement_index()
                if any(hp._refinement_state(c,*index)!='waiting' for c in refinements): reasons.append('ready_refinements')
        elif any(now-c.observed_sample>1500000 for c in refinements):
            reasons.append('ready_refinements')
        if any(t.header is None and now>=t.detected_sample+340000 for t in tracks): reasons.append('legacy_headers')
        if any(t.header is not None and now>=t.expected_end_sample for t in tracks): reasons.append('ready_payloads')
        if not d.payload_pool.can_admit(): reasons.append('payload_capacity')
        if hp is not None and not hp.pool.available: reasons.append('header_capacity')
        if oldest+next_samples/FS >= self._horizon-self.margin_s: reasons.append('candidate_age_guard')
        if anchors and margin <= self.margin_s: reasons.append('iq_retention_guard')
        sic=d.sic
        if sic is not None and not sic.pool.closed:
            if sic.pending or sic.rebuilds or sic._eligible_tiles(): reasons.append('sic_work')
            if not sic.pool.can_admit(): reasons.append('sic_capacity')
        signature=(now, d.input_abs_sample, tuple(c.candidate_id for c in candidates),
                   tuple(sorted(hp.pending)) if hp is not None else (), tuple(sorted(d._payload_pending)),
                   tuple((t.track_id,t.status) for t in tracks),
                   d.runtime_counts['header_jobs_completed'], d.runtime_counts['payload_jobs_completed'],
                   tuple(sorted(sic.pending)) if sic is not None else (),
                   tuple(sorted(sic.tiles.items())) if sic is not None else ())
        return dict(reasons=reasons,signature=signature,oldest_candidate_s=oldest,iq_margin_s=margin)

    def _progress(self, state):
        now=self.clock()
        if state['signature'] != self._last_signature:
            self._last_signature=state['signature'];self._last_progress=now
        return now-self._last_progress

    def _trip(self, reasons):
        self.watchdog_trips += 1
        self.warn('Exhaustive decode watchdog: no owner/worker progress for '
                  f'{self.watchdog_s:g} s ({", ".join(reasons)}); leaving this wait and continuing.')
        self._last_progress=self.clock()

    def before_feed(self, service, *, next_samples=50_000, stop_event=None, on_wait=None):
        """Service callback must publish packets through the caller's normal path."""
        began=self.clock()
        waiting=False
        self._last_signature=None
        try:
            while stop_event is None or not stop_event.is_set():
                service()
                state=self.state(next_samples)
                if not state['reasons']: return True
                if not waiting:
                    waiting=True;self.wait_count+=1;self._waiting_since=began
                if on_wait is not None:on_wait()
                if self._progress(state) >= self.watchdog_s:
                    self._trip(state['reasons'])
                    pipeline=self.decoder.acquisition_process
                    if pipeline is not None and pipeline.outstanding>=pipeline.capacity:
                        # feed_raw itself would block on a wedged acquisition
                        # pipe. Surface that hard worker fault instead of hanging.
                        raise RuntimeError('Exhaustive watchdog: acquisition cannot admit another block')
                    return True
                self.sleep(.001)
            return False
        finally:
            if waiting:self.wait_wall_s += self.clock()-began
            self._waiting_since=None

    def finish(self, stop_event=None):
        """Use ordinary EOF rules without a total wall-time deadline.

        The no-progress watchdog remains active and cancellation is still
        immediate. A stalled EOF drain uses the existing safe stop/close path.
        """
        controller=self
        class Stop:
            def is_set(self):
                if stop_event is not None and stop_event.is_set():return True
                state=controller.state(0)
                if controller._progress(state)>=controller.watchdog_s:
                    controller._trip(['end_of_stream'])
                    return True
                return False
        self._last_signature=None
        original=self.decoder.payload_finish_timeout
        self.decoder.payload_finish_timeout=float('inf')
        try:return self.decoder.finish(stop_event=Stop())
        finally:self.decoder.payload_finish_timeout=original

    def snapshot(self, progress=None):
        elapsed=max(0.,self.clock()-self.started)
        waiting=0. if self._waiting_since is None else self.clock()-self._waiting_since
        eta=None if progress is None or progress<=0 else max(0.,elapsed*(1-progress)/progress)
        return dict(enabled=True,wait_wall_s=self.wait_wall_s+waiting,wait_count=self.wait_count,
                    payload_retry="grid",
                    watchdog_trips=self.watchdog_trips,watchdog_s=self.watchdog_s,
                    deadline_margin_s=self.margin_s,max_candidate_age_s=self.max_candidate_age_s,
                    min_iq_margin_s=self.min_iq_margin_s,wall_s=elapsed,eta_s=eta)
