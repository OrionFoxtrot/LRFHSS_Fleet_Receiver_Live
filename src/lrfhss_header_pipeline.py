"""Owner-side scheduling/commit logic for the optional asynchronous PHDR path."""
from collections import deque
from bisect import bisect_left
from dataclasses import astuple
import time
from lrfhss_header_jobs import HeaderJob, HeaderPool

# A replica prediction must match an established packet's expected header
# replica in time AND frequency. The window is never broader than the track
# manager's own "same packet" association bounds (TrackManager
# header_time_tolerance=2048 samples, observation_cfo_tolerance_hz=250 Hz):
# a transmission the tracker would keep as a distinct packet is never skipped.
# (Change 09: the earlier 4096-sample / 500-Hz window admitted an independent
# same-channel packet 2.5 ms / 400 Hz away in an adversarial collision test.)
REPLICA_TIME_TOLERANCE_SAMPLES = 2048
REPLICA_FREQUENCY_TOLERANCE_HZ = 250.

# Change 10, deferred skip (mode "deferred"). A matched candidate is PARKED,
# never discarded on geometry alone: an established packet's first header can
# be corrupted yet pass its 8-bit CRC (e.g. a wrong payload length), and its
# own replicas are exactly what recovers the packet (Audit 02, A1). A parked
# candidate is dropped only once the matched packet's payload passed its PHY
# CRC-16. It is validated as usual when that packet fails CRC, is rejected,
# expires, fails or is cancelled, when it has not resolved within
# REPLICA_PARK_MAX_AGE_S of the candidate's observation (never more than half
# the candidate age limit), and at end of stream. Parked candidates hold no
# IQ and no worker lease; their number never exceeds the acquisition's own
# pending-candidate bound. Mode "on" (immediate skip, Changes 06-09) remains
# only as a research comparison arm: it can lose that packet.
REPLICA_PARK_MAX_AGE_S = 6.
REPLICA_SKIP_MODES = ('off', 'shadow', 'on', 'deferred')


class HeaderPipeline:
    def __init__(self, decoder, workers, capacity, *, max_age_s=12., test_gate=None,
                 test_delay=0., outstanding_capacity=None):
        if max_age_s <= 0:
            raise ValueError('header maximum age must be positive')
        self.d = decoder
        self.pool = HeaderPool(workers,capacity,test_gate,outstanding_capacity=outstanding_capacity,
                               shared_iq=decoder.header_shared_iq,
                               retained_capacity=decoder.acquisition.MAX_PENDING if decoder.candidate_transactions else 0)
        self.max_age_s = max_age_s
        self.test_delay = test_delay
        self._refinement_cache = None
        self._lateness_cache = None
        self.pending = {}
        self.inbox = deque()             # polled results held by the owner until incorporated (10.4k)
        self.legacy_pending = set()
        self.sequence = 0
        self.worker_grid_caches = {}
        self.worker_oscillator_caches = {}
        self.parked = {}                  # candidate_id -> park record (insertion ordered)
        self.eof = False                  # finish() started: park nothing more (Change 11)
        self.released = deque()           # released parked candidates awaiting a worker
        self.park_capacity = decoder.acquisition.MAX_PENDING
        self.park_max_age_samples = int(min(REPLICA_PARK_MAX_AGE_S, max_age_s/2)*1e6)

    def fields(self, c):
        return dict(candidate_id=c.candidate_id,candidate_sample=c.header_start,
                    inferred_packet_start=c.header_start-c.physical_index*233472,
                    physical_header=c.physical_index+1,source=c.source,
                    candidate_cfo_hz=c.frequency_hz-c.nominal_hz)

    def replica_match(self, c):
        """Established packet whose header replica ``c`` is expected to be, else None.

        Pure owner-side geometry from validated tracks: the replica's expected
        physical start (packet start + index*STAY_HDR) and RF frequency (that
        packet's hop for this replica + its CFO). No schedule or identity.
        """
        return self._replica_match(c)[1]

    def _replica_match(self, c):
        """(matched track, match fields), or (None, None)."""
        from lrfhss_live import STAY_HDR, DEFAULT_HEADER_COUNT
        if c is None or c.source != 'sync' or not 0 <= c.physical_index < DEFAULT_HEADER_COUNT:
            return None, None
        best = best_track = None
        manager = self.d.track_manager
        associable = getattr(manager, "_associable", lambda t, source: True)
        for t in manager._known():
            h = t.header
            if h is None or t.cfo_hz is None or len(t.expected_hops_hz) <= c.physical_index:
                continue
            if not associable(t, c.source):
                continue          # an unpublished failed SIC-only packet: its replicas are needed (10.4k)
            dt = c.header_start-(h.start+c.physical_index*STAY_HDR)
            if abs(dt) > REPLICA_TIME_TOLERANCE_SAMPLES:
                continue
            df = c.frequency_hz-(t.expected_hops_hz[c.physical_index]+t.cfo_hz)
            if abs(df) > REPLICA_FREQUENCY_TOLERANCE_HZ:
                continue
            if best is None or abs(dt) < abs(best['dt_samples']):
                best = dict(track_id=t.track_id, dt_samples=int(dt), df_hz=float(df))
                best_track = t
        return best_track, best

    def _replica_gate(self, c):
        """True when the candidate must not be submitted now (on/deferred modes)."""
        d = self.d
        if d.header_replica_skip == 'off':
            return False
        track, match = self._replica_match(c)
        if match is None:
            return False
        d.runtime_counts['phdr_replica_predicted'] += 1
        if d.header_replica_skip == 'shadow':
            c.replica_prediction = match
            return False
        if d.header_replica_skip == 'deferred':
            if getattr(self, 'eof', False):
                d.runtime_counts['phdr_replica_eof_validated'] += 1
                return False
            return self._park(c, track, match)
        d.runtime_counts['phdr_replica_skipped'] += 1
        self._clear_transaction(c)
        d._acquisition_event('header_candidate_skipped', d.abs_sample, **self.fields(c),
                             reason='established_packet_replica', **match)
        return True

    def _park(self, c, track, match):
        d = self.d
        if len(self.parked) >= self.park_capacity:
            # Bounded like the acquisition's pending list: validate instead.
            d.runtime_counts['phdr_replica_park_capacity_validated'] += 1
            return False
        self._clear_transaction(c)
        self.parked[c.candidate_id] = dict(candidate=c, track=track, match=match, parked_sample=d.abs_sample)
        self._geometry_changed()
        d.runtime_counts['phdr_replica_parked'] += 1
        d.runtime_counts['phdr_replica_parked_high_watermark'] = max(
            d.runtime_counts['phdr_replica_parked_high_watermark'], len(self.parked))
        d._acquisition_event('header_candidate_parked', d.abs_sample, **self.fields(c),
                             reason='established_packet_replica', **match)
        return True

    def _release(self, candidate_id, reason):
        d = self.d
        entry = self.parked.pop(candidate_id)
        self._geometry_changed()
        c = entry['candidate']
        d.runtime_counts['phdr_replica_released'] += 1
        d.runtime_counts['phdr_replica_released_' + reason] += 1
        d._acquisition_event('header_candidate_released', d.abs_sample, **self.fields(c), reason=reason,
                             parked_sample=entry['parked_sample'], **entry['match'])
        c.sic_release_wall = time.perf_counter()
        self.released.append(c)
        self._geometry_changed()

    def release_parked(self, reason):
        """Release every parked candidate for ordinary validation."""
        for candidate_id in list(self.parked):
            self._release(candidate_id, reason)

    def end_of_stream(self):
        """Finalization: release every parked candidate and park nothing more.

        Change 11: finish() keeps servicing header work after it starts, and
        that service could park new replicas that nothing would ever resolve
        (only poll() runs during the final payload drain). From here on the
        deferred skip validates every candidate, exactly as the skip-off
        receiver does at end of stream.
        """
        self.eof = True
        self.release_parked('end_of_stream')

    def _resolve_parked(self):
        """Drop parked candidates whose packet passed CRC; release the others."""
        if not self.parked:
            return
        d = self.d
        manager = d.track_manager
        for candidate_id, entry in list(self.parked.items()):
            c, track = entry['candidate'], entry['track']
            if track.track_id in manager.active_tracks or track.track_id in manager.payload_tracks:
                if d.abs_sample-c.observed_sample > self.park_max_age_samples:
                    self._release(candidate_id, 'park_deadline')
                continue
            if track.status == 'completed' and track.reason == 'payload_decoded':
                del self.parked[candidate_id]
                self._geometry_changed()
                d.runtime_counts['phdr_replica_skipped'] += 1
                d._acquisition_event('header_candidate_skipped', d.abs_sample, **self.fields(c),
                                     reason='established_packet_replica', confirmation='payload_crc_ok',
                                     parked_sample=entry['parked_sample'], **entry['match'])
            else:
                self._release(candidate_id, 'phy_crc_failed' if track.status == 'completed'
                              else 'track_' + track.status)

    def _shadow_outcome(self, c, span, h, track, r):
        """Record what a predicted-skippable validation actually produced."""
        prediction = getattr(c, 'replica_prediction', None) if c is not None else None
        if prediction is None:
            return
        d = self.d
        if h is None:
            outcome = 'invalid'
        elif track is None:
            outcome = 'valid_not_associated'
        elif getattr(track, 'duplicate_of', None) is None and getattr(c, '_replica_new_track', False):
            outcome = 'valid_new_track'
        elif track.track_id == prediction['track_id']:
            outcome = 'valid_same_track'
        else:
            outcome = 'valid_other_track'
        d.runtime_counts['phdr_replica_shadow_' + outcome] += 1
        d.processing['phdr_replica_shadow_cpu_s'] += r.get('cpu_s') or 0.
        d._acquisition_event('header_replica_shadow', d.abs_sample, **self.fields(c),
                             span_hz=span, outcome=outcome, worker_cpu_s=r.get('cpu_s'), **prediction)

    def _expired(self,c,phase='initial'):
        d = self.d
        if d.abs_sample-c.observed_sample <= self.max_age_s*1e6:
            return False
        d._acquisition_event('header_candidate_late',d.abs_sample,**self.fields(c),
                             reason='candidate_age_limit',phase=phase,
                             observed_sample=c.observed_sample,
                             candidate_age_s=(d.abs_sample-c.observed_sample)/1e6)
        d.runtime_counts['header_candidates_expired'] += 1
        d.runtime_counts['header_candidates_expired_'+phase] += 1
        self._clear_transaction(c)
        return True

    def _submit(self,c=None,track=None,span=None):
        d = self.d
        legacy = track is not None
        if not legacy and span is not None and d.candidate_transactions and getattr(c, 'transaction_attempts', None):
            attempts = getattr(c, 'transaction_attempts', None)
            if not attempts or span not in attempts:
                raise RuntimeError('Missing candidate transaction retry outcome')
            attempt = attempts[span]
            virtual_id = c.transaction_job_id + ':r' + str(span)
            result = dict(header=attempt['header'], detail=attempt['detail'],
                wall_s=attempt['wall_s'], cpu_s=attempt['cpu_s'],
                worker_wall=c.transaction_complete_wall,
                cached_transaction_attempt=True, coalesced=attempt['coalesced'])
            if not self.pool.submit_cached(virtual_id, result):
                raise RuntimeError('Cached PHDR result exceeded ordered-result capacity')
            del attempts[span]
            self.pending[virtual_id] = dict(candidate=c, track=None, span=span,
                fields=self.fields(c), submitted_wall=time.perf_counter(), submitted_sample=d.abs_sample)
            d._consume_header_token()
            d.runtime_counts['phdr_cached_refinements_enqueued'] += 1
            return True
        if legacy:
            base=max(0,track.detected_sample-50000);stop=track.detected_sample+340000
            fields=dict(candidate_id=-track.track_id, candidate_sample=track.detected_sample+8704,
                        inferred_packet_start=track.detected_sample+8704,physical_header=1,
                        source='legacy_energy',track_id=track.track_id)
        else:
            base=max(0,c.header_start-35000);stop=c.header_start+290000
            fields=self.fields(c)
        snapshot_start = time.perf_counter()
        retained = getattr(c, 'transaction_snapshot', None)
        if retained is not None:
            old_base, old_stop, snapshot = retained
            if (old_base, old_stop) != (base, stop):
                raise RuntimeError('Candidate transaction changed its retained IQ window')
            d.runtime_counts['phdr_iq_snapshot_lease_reuse'] += 1
        else:
            snapshot=(self.pool.snapshots.snapshot(d.ring,base,stop) if self.pool.snapshots is not None
                      else d._snapshot('header',base,stop))
        snapshot_complete = time.perf_counter()
        if snapshot is None:
            d.runtime_counts['header_candidates_expired'] += 1
            d._acquisition_event('header_candidate_late',d.abs_sample,**fields,
                                 reason='header_iq_unavailable')
            if legacy and track.header is None:
                d.track_manager.retire(track,'expired',d.abs_sample,'header_iq_unavailable')
            return False
        self.sequence += 1
        job_id=f'{d.receiver_id}:h{self.sequence}'
        iq_bytes,dtype=snapshot
        job=HeaderJob(job_id,iq_bytes,dtype,base,
                      'legacy' if legacy else 'physical',() if legacy else astuple(c),
                      onset=track.detected_sample if legacy else 0, span=span,
                      batch_header=d.batch_header,grid_cache=d.header_grids is not None,
                      delay_s=self.test_delay, frequency_reuse=d.phdr_frequency_reuse,
                      frequency_selection=getattr(c,"frequency_selection",None),
                      validation_reuse=d.header_validation_reuse,
                      previous_validation=getattr(c,"previous_validation",None),
                      candidate_transaction=d.candidate_transactions and not legacy,
                      oscillator_reuse=d.header_oscillator_reuse,
                      cfo_fft=getattr(d,'header_cfo_fft','float64'),
                      cfo_tie_margin=getattr(d,'header_cfo_tie_margin',.01),
                      phdr_frontend=d.phdr_frontend, phdr_decimation=d.phdr_decimation,
                      soft_trials=d.dsp_soft_trials,
                      allow_refinements=d.candidate_local_retry and (
                          d.candidate_transaction_mode == 'eager' or span is not None),
                      retain_iq=d.candidate_transactions and not legacy and span is None and
                          d.candidate_transaction_mode == 'deferred')
        try:
            admitted=self.pool.submit(job)
        except Exception as exc:
            admitted=False
            d._acquisition_event('header_worker_start_failed',d.abs_sample,
                                 **fields,error=repr(exc))
        if not admitted:
            if self.pool.snapshots is not None:
                if retained is not None:
                    self.pool.release_retained(iq_bytes)
                    del c.transaction_snapshot
                else:
                    self.pool.snapshots.release(iq_bytes)
            d.runtime_counts['header_jobs_rejected'] += 1
            d._acquisition_event('header_candidate_enqueue_rejected',d.abs_sample,
                                 **fields,reason='header_queue_capacity')
            if legacy and track.header is None:
                d.track_manager.retire(track,'queue_rejected',d.abs_sample,'header_queue_capacity')
            return False
        if job.retain_iq:
            c.transaction_snapshot = (base, stop, snapshot)
        elif retained is not None:
            del c.transaction_snapshot
        submitted=time.perf_counter()
        self.pending[job_id]=dict(candidate=c,track=track,span=span,fields=fields,
                                 submitted_wall=submitted,submitted_sample=d.abs_sample)
        self._geometry_changed()
        if legacy:self.legacy_pending.add(track.track_id)
        d._consume_header_token()
        d.runtime_counts['header_jobs_submitted'] += 1
        if not legacy and span is None:
            d.runtime_counts['phdr_logical_candidates_submitted'] += 1
        if job.candidate_transaction:
            d.runtime_counts['phdr_transactions_submitted'] += int(span is None)
            d.runtime_counts['phdr_worker_segments_submitted'] += 1
            d.runtime_counts['phdr_owner_round_trip_refinements'] += int(span is not None)
        d._acquisition_event('header_candidate_enqueued',d.abs_sample,**fields,
                             job_id=job_id,submission_sequence=self.sequence,
                             submitted_wall=submitted, snapshot_start_wall=snapshot_start,
                             snapshot_complete_wall=snapshot_complete,
                             eligible_wall=getattr(c, "eligible_wall", None),
                             snapshot_copy_complete_wall=snapshot_complete, iq_bytes=len(iq_bytes),
                             detected_wall=getattr(c, "detected_wall", None),
                             admitted_wall=getattr(c, "admitted_wall", None))
        return True

    def collect(self):
        d=self.d
        # Batch 10.4k (review 2026-10-02, finding 6): HeaderPool.poll() hands over every ready
        # result at once and cannot return them again. They wait in an owner-held inbox and
        # leave it one at a time, after the diagnostics observer has seen them, as on the
        # payload path: an observer exception leaves the rest for the next collect.
        if not hasattr(self, 'inbox'):
            self.inbox = deque()
        self.inbox.extend(self.pool.poll())
        while self.inbox:
            kind,job_id,r = self.inbox[0]
            if self.pool.pool.observer is not None:
                self.pool.pool.observer('result_incorporation',job_id,dict(wall=time.perf_counter(),kind=kind))
            self.inbox.popleft()
            context=self.pending[job_id];c=context['candidate'];track=context['track']
            fields=context['fields'];span=context['span']
            if kind=='started':
                d.runtime_counts['header_jobs_started'] += 1
                d.timings.add('header_queue_wait',r['worker_wall']-context['submitted_wall'])
                d._acquisition_event('header_validation_started',d.abs_sample,
                    **fields,job_id=job_id,worker_wall=r['worker_wall'],
                    candidate_age_samples=d.abs_sample-(c.observed_sample if c else track.created_sample))
                continue
            del self.pending[job_id]
            self._geometry_changed()
            if track is not None:self.legacy_pending.discard(track.track_id)
            if r.get('cached_transaction_attempt'):
                d.runtime_counts['phdr_cached_refinements_committed'] += 1
                d.timings.add('cached_refinement_order_wait', time.perf_counter()-context['submitted_wall'])
                self._commit(c, track, span, fields, job_id, r)
                continue
            d.timings.add('header_job',r.get('wall_s',0.))
            d.timings.add('header_commit_wait',max(0.,time.perf_counter()-r.get('worker_wall',time.perf_counter())))
            if r.get('oscillator_cache') is not None:
                self.worker_oscillator_caches[r['worker_pid']]=r['oscillator_cache']
            if r.get('grid_cache') is not None:
                self.worker_grid_caches[r['worker_pid']]=r['grid_cache']
            for key,value in r.get('counters',{}).items():
                if key.endswith(('_max_s', '_peak_bytes')):d.processing[key]=max(d.processing[key],value)
                else:d.processing[key]+=value
            for key,values in r.get('timings',{}).items():
                for value in values:d.timings.add('phdr_'+key,value)
            d.processing['header_worker_wall_s']+=r.get('wall_s',0.)
            d.processing['header_worker_cpu_s']+=r.get('cpu_s',0.)
            d.processing['header_deserialize_wall_s']+=r.get('deserialize_wall_s',0.)
            d.processing['header_deserialize_cpu_s']+=r.get('deserialize_cpu_s',0.)
            d.processing['header_initialization_wall_s']+=r.get('initialization_wall_s',0.)
            d.processing['header_initialization_cpu_s']+=r.get('initialization_cpu_s',0.)
            if kind=='failed':
                d.runtime_counts['header_jobs_failed']+=1
                d._acquisition_event('header_worker_failed',d.abs_sample,**fields,job_id=job_id,**r)
                if c is not None and hasattr(c, 'transaction_snapshot'):
                    del c.transaction_snapshot  # pool already released failed job IQ
                self._clear_transaction(c)
                if track is not None and track.header is None:
                    d.track_manager.retire(track,'failed',d.abs_sample,'header_worker_failure')
                continue
            d.runtime_counts['header_jobs_completed']+=1
            if 'attempts' in r:
                attempts = r['attempts']
                if c is None or not attempts or attempts[0]['span_hz'] != span:
                    raise RuntimeError('Malformed candidate transaction result')
                c.transaction_attempts = {a['span_hz']: a for a in attempts[1:]}
                c.transaction_job_id = job_id
                c.transaction_complete_wall = r['worker_wall']
                d.timings.add('candidate_transaction_service', r['wall_s'])
                d.timings.add('candidate_transaction_latency', time.perf_counter()-context['submitted_wall'])
                d._acquisition_event('header_transaction_completed', d.abs_sample,
                    **fields, job_id=job_id, logical_attempts=len(attempts),
                    attempts=[dict(span_hz=a['span_hz'], valid=a['header'] is not None,
                        coalesced=a['coalesced'], wall_s=a['wall_s'], cpu_s=a['cpu_s'],
                        **{k:v for k,v in a['detail'].items() if k not in ('frequency_selection','previous_validation')}) for a in attempts])
            self._commit(c, track, span, fields, job_id, r)

    def _clear_transaction(self, c):
        if c is None:
            return
        snapshot = getattr(c, 'transaction_snapshot', None)
        if snapshot is not None:
            if self.pool.snapshots is not None:
                self.pool.release_retained(snapshot[2][0])
            del c.transaction_snapshot
        attempts = getattr(c, 'transaction_attempts', {})
        self.d.runtime_counts['phdr_precomputed_refinements_discarded'] += len(attempts)
        for name in ('transaction_attempts', 'transaction_job_id', 'transaction_complete_wall'):
            if hasattr(c, name):
                delattr(c, name)

    def _commit(self, c, track, span, fields, job_id, r):
        d = self.d
        h,detail=r['header'],r['detail']
        previous=detail.pop('previous_validation',None)
        if c is not None and previous is not None:
            c.previous_validation=previous
        selection=detail.pop('frequency_selection',None)
        if c is not None and selection is not None:
            c.frequency_selection=selection
        commit_started = time.perf_counter()
        event='header_local_refinement' if span is not None else 'header_validation_result'
        d._acquisition_event(event,d.abs_sample,**{k:v for k,v in fields.items() if k!='source'},job_id=job_id,valid=h is not None,
                            span_hz=span,wall_s=r['wall_s'],worker_cpu_s=r.get('cpu_s'),worker_pid=r.get('worker_pid'),worker_complete_wall=r.get('worker_wall'),
                            commit_started_wall=commit_started,
                            cached_transaction_attempt=r.get('cached_transaction_attempt',False),
                            exact_coalesced=r.get('coalesced',False),**detail)
        if h is not None:
            self._clear_transaction(c)
            d.runtime_counts['phdr_validation_success']+=1
            # An earlier result may have already promoted this energy candidate.
            # Associate through the manager, never replace its canonical header.
            active_track=track if track is not None and track.header is None and track.track_id in d.track_manager.active_tracks else None
            created_before = d.track_manager.counts['tracks_created']
            associated = d._accept_physical(h,0 if c is None else c.physical_index,
                candidate_id=fields['candidate_id'],
                source='candidate_local_retry' if span is not None else fields['source'],
                score=detail.get('sync_correlation',0.),
                energy=track.acquisition_energy_ratio if c is None else c.energy_ratio,
                track=active_track)
            if c is not None and getattr(c, 'replica_prediction', None) is not None:
                c._replica_new_track = d.track_manager.counts['tracks_created'] > created_before
                self._shadow_outcome(c, span, h, associated, r)
            d._acquisition_event("header_commit_associated",d.abs_sample,job_id=job_id,
                commit_started_wall=commit_started,**fields)
            return
        self._shadow_outcome(c, span, None, None, r)
        d.runtime_counts['phdr_demod_failure' if detail.get('reason') in
            ('header_correlation','header_iq_incomplete') else 'phdr_validation_failure']+=1
        if track is not None:
            if track.header is None:
                d.track_manager.retire(track,'rejected',d.abs_sample,'header_validation_failed',
                    stage='header_validation',rejection_detail=detail,**{k:v for k,v in fields.items() if k!='track_id'})
        elif span is None and d.candidate_local_retry and c.source=='sync' and (c.sync_correlation or 0)>=.6:
            if len(d._refinement_pending)<d.acquisition.MAX_PENDING:
                self._queue_refinements(c, (100,0), span, detail)
            else:
                self._clear_transaction(c)
                d._acquisition_event('header_refinement_rejected',d.abs_sample,
                                     **fields,reason='refinement_capacity')
        elif span==100:
            if len(d._refinement_pending)<d.acquisition.MAX_PENDING:
                self._queue_refinements(c, (0,), span, detail)
            else:
                self._clear_transaction(c)
                d._acquisition_event('header_refinement_rejected',d.abs_sample,
                                     **fields,reason='refinement_capacity')
        else:
            self._clear_transaction(c)
            d._acquisition_event('header_attempts_exhausted',d.abs_sample,
                                 **fields,reason=detail.get('reason'))
        d._acquisition_event('header_candidate_rejected',d.abs_sample,**fields,**detail)


    def _queue_refinements(self, c, spans, previous_span, detail):
        # The 100-Hz search is a subset of the initial 250-Hz FFT mask.
        # If that mask's winning bin is within 100 Hz, both select exactly
        # the same frequency (including ascending-bin argmax ties). The IQ
        # window and all validator arguments are identical. No score cutoff.
        frequency = detail.get('frequency_hz')
        retained = []
        for span in spans:
            duplicate = self.d.exact_refinement_dedup and frequency is not None and (
                span == 100 and previous_span is None and
                c.frequency_hz-100 <= frequency <= c.frequency_hz+100 or
                span == 0 and frequency == c.frequency_hz)
            if duplicate:
                self.d.runtime_counts['phdr_exact_refinements_coalesced'] += 1
                self.d._acquisition_event('header_refinement_coalesced', self.d.abs_sample,
                    **self.fields(c), span_hz=span, selected_frequency_hz=frequency,
                    reason='identical_frequency_window_validator')
            else:
                retained.append(span)
        if retained:
            self.d.runtime_counts['phdr_refinement_requests_queued'] += 1
            c.retry_spans = tuple(retained)
            self.d._refinement_pending.append(c)
            self._lateness_cache = None

        elif self.d.candidate_transactions:
            self._clear_transaction(c)

    def _geometry_changed(self):
        self._refinement_cache = None
        self._lateness_cache = None

    def _geometry_key(self):
        # Lightweight owners without revisioned queues use the direct calculation.
        a = getattr(self.d.acquisition, 'pending_revision', None)
        t = getattr(self.d.track_manager, 'refinement_revision', None)
        return None if a is None or t is None else (a, t)

    def _refinement_index(self):
        d = self.d
        key = self._geometry_key()
        cached = getattr(self, '_refinement_cache', None)
        if key is not None and cached is not None and cached[0] == key:
            return cached[1]
        if key is None:
            legacy = [t.detected_sample+8704 for t in d.track_manager.active_tracks.values() if t.header is None]
            known = sorted(t.header.start for t in d.track_manager._known() if t.header is not None)
        else:
            legacy, known = d.track_manager.refinement_geometry()
        originals = [c.header_start-c.physical_index*233472 for c in d.acquisition.pending]
        originals.extend(v['candidate'].header_start-v['candidate'].physical_index*233472
                         for v in getattr(self, 'parked', {}).values())
        originals.extend(c.header_start-c.physical_index*233472 for c in getattr(self, 'released', ()))
        originals.extend(v['candidate'].header_start-v['candidate'].physical_index*233472
                         for v in self.pending.values() if v['candidate'] is not None and v['span'] is None)
        originals.extend(legacy)
        originals.sort()
        index = originals, known
        if key is not None:
            self._refinement_cache = key, index
        return index

    def _refinement_state(self, candidate, original_starts, known_starts):
        start = candidate.header_start - candidate.physical_index*233472
        def nearby(starts, center, tolerance):
            i=bisect_left(starts,center-tolerance)
            return i<len(starts) and starts[i]<=center+tolerance
        if self.d.abs_sample < start+950000 or nearby(original_starts, start, 8192):
            return 'waiting'
        return 'obsolete' if nearby(known_starts, start, 4096) else 'ready'

    def sic_lateness(self, now):
        """Read pending eligibility without dispatching work or mutating queues."""
        d = self.d
        stamp = self._geometry_key()
        key = (stamp, d.abs_sample, self.max_age_s)
        cached = getattr(self, '_lateness_cache', None)
        if stamp is not None and cached is not None and cached[0] == key:
            oldest, due = cached[1:]
            return (now-min(now, oldest))/self.max_age_s, due
        oldest = float('inf')
        max_age_samples = self.max_age_s * 1e6
        for pending, timestamp in ((d.acquisition.pending, 'eligible_wall'), (self.released, 'sic_release_wall')):
            for c in pending:
                since = getattr(c, timestamp, None)
                if (since is not None and d.abs_sample >= c.header_start+335000
                        and d.abs_sample-c.observed_sample <= max_age_samples):
                    oldest = min(oldest, since)
        due = 0
        index = None
        for c in d._refinement_pending:
            age = d.abs_sample-c.observed_sample
            if 1500000 < age <= max_age_samples:
                if index is None:
                    index = self._refinement_index()
                if self._refinement_state(c, *index) == 'ready':
                    due += 1
        if stamp is not None:
            self._lateness_cache = key, oldest, due
        return (now-min(now, oldest))/self.max_age_s, due

    def _retry_one(self):
        d=self.d
        if not d._refinement_pending:return False
        # The original/known sets are stable during this single-owner scan.
        index = self._refinement_index()
        for _ in range(len(d._refinement_pending)):
            c=d._refinement_pending.popleft()
            self._lateness_cache = None
            if self._expired(c,phase='refinement'):continue
            state = self._refinement_state(c, *index)
            if state == 'waiting':
                d._refinement_pending.append(c)
                self._lateness_cache = None
                continue
            if state == 'obsolete':
                self._clear_transaction(c)
                d.runtime_counts['phdr_refinements_obsolete_skipped'] += 1
                d._acquisition_event('header_refinement_skipped',d.abs_sample,**self.fields(c),
                                     reason='established_packet_preserved')
                continue
            self._submit(c,span=getattr(c,'retry_spans',(100,0))[0])
            return True
        return False

    def service(self):
        d=self.d
        # Observe eligibility independently of worker availability.
        for c in d.acquisition.pending:
            if getattr(c, "eligible_wall", None) is None and d.abs_sample>=c.header_start+335000:
                c.eligible_wall = time.perf_counter()
                self._lateness_cache = None
                d._acquisition_event("header_candidate_eligible",d.abs_sample,
                    **self.fields(c),eligible_wall=c.eligible_wall)
        # Retain metadata only while the bounded ring can still serve useful data.
        before = len(d.acquisition.pending)
        d.acquisition.pending=[c for c in d.acquisition.pending if not self._expired(c)]
        if len(d.acquisition.pending) != before:
            self._geometry_changed()
        for track in list(d.track_manager.active_tracks.values()):
            if not self.pool.available or d._header_tokens==0:break
            if track.header is not None or track.track_id in self.legacy_pending:continue
            if d.abs_sample>=track.detected_sample+340000:
                self._submit(track=track)
        # Deferred replica skip: resolve parked candidates, then validate the
        # released ones (older than any new candidate) before new work.
        self._resolve_parked()
        if self.released:
            before = len(self.released)
            self.released = deque(c for c in self.released if not self._expired(c))
            if len(self.released) != before:
                self._geometry_changed()
        while self.released and self.pool.available and d._header_tokens!=0:
            c = self.released.popleft()
            self._geometry_changed()
            self._submit(c)
        if (self.pool.available and d._header_tokens!=0 and d._refinement_pending and
            d.abs_sample-d._refinement_pending[0].observed_sample>1500000):
            self._retry_one()
        limit=self.pool.available
        if d._header_tokens is not None:limit=min(limit,d._header_tokens)
        if limit:
            for c in d.acquisition.ready(d.abs_sample,limit):
                if not self._replica_gate(c):
                    self._submit(c)
        while self.pool.available and d._header_tokens!=0 and self._retry_one():pass
        d.runtime_counts['refinement_backlog_high_watermark']=max(
            d.runtime_counts['refinement_backlog_high_watermark'],len(d._refinement_pending))

    def close(self,reason):
        for c in self.d._refinement_pending:
            self._clear_transaction(c)
        self.pool.close(graceful=(reason=='finished'))
        for kind,job_id,_ in getattr(self,'inbox',()):
            if kind!='started':
                self.d.runtime_counts['header_results_stranded']+=1
                self.d._acquisition_event('header_result_stranded',self.d.abs_sample,job_id=job_id,
                                          result_kind=kind,reason=reason)
        if hasattr(self,'inbox'):self.inbox.clear()
        for job_id,ctx in self.pending.items():
            self.d.runtime_counts['header_jobs_cancelled']+=1
            self.d._acquisition_event('header_candidate_cancelled',self.d.abs_sample,
                                     **ctx['fields'],job_id=job_id,reason=reason)
        parked = [e['candidate'] for e in self.parked.values()] + list(self.released)
        self.parked.clear();self.released.clear()
        self._geometry_changed()
        for c in parked:
            self.d.runtime_counts['phdr_replica_parked_cancelled']+=1
            self.d._acquisition_event('header_candidate_cancelled',self.d.abs_sample,
                                     **self.fields(c),reason=reason,parked=True)
        for c in list(self.d.acquisition.recent)+[v['candidate'] for v in self.pending.values() if v['candidate'] is not None]:
            for name in ('transaction_snapshot','transaction_attempts','transaction_job_id','transaction_complete_wall'):
                if hasattr(c,name):delattr(c,name)
        self.pending.clear();self.legacy_pending.clear()
        self._geometry_changed()

    def grid_stats(self):
        caches=list(self.worker_grid_caches.values())
        result = dict(scope='asynchronous_workers',workers=len(caches),
            bytes=0 if self.pool.pool.closed else sum(c['bytes'] for c in caches),
            **{key:sum(c[key] for c in caches) for key in
               ('hits','misses','peak_bytes','max_bytes')})
        guarded=[c for c in caches if c.get('cfo_fft')=='float32_guarded']
        if guarded:
            result.update(cfo_fft='float32_guarded',
                cfo_tie_margins=sorted({c['cfo_tie_margin'] for c in guarded}),
                **{key:sum(c.get(key,0) for c in guarded) for key in
                   ('fast_ffts','fast_fallbacks','fast_unsupported','fast_window_peak_bytes')})
        return result

    def oscillator_stats(self):
        caches=list(self.worker_oscillator_caches.values())
        return dict(workers=len(caches),closed=self.pool.pool.closed,
            bytes=0 if self.pool.pool.closed else sum(c['bytes'] for c in caches),
            **{key:sum(c[key] for c in caches) for key in ('hits','misses','peak_bytes','max_bytes')})
