"""Exact PHDR hypotheses over one immutable, candidate-owned IQ window.

This cache lives for ONE worker transaction. Its namespace binds the complete
IQ bytes, dtype, shape, base, candidate and decoder configuration. Frequencies
are keyed by their exact float64 representation, never by a tolerance. No DSP
array or shared-memory view escapes the function. A deferred transaction spans
an initial segment and an eligible retry segment using the same retained lease.
The owner decides whether
precomputed retry outcomes remain eligible for association.
"""
from copy import deepcopy
from dataclasses import replace
import struct
import time
import numpy as np
from lrfhss_metrics import measure


class CandidateContext:
    def __init__(self, z, base, candidate, *, batch_header, grids, counters, frequency_selection=None, previous_validation=None,
                 frontend='reference', decimation=64):
        if z.flags.writeable:
            raise ValueError('Candidate transaction requires immutable IQ')
        import lrfhss_live as phy
        self.z, self.base, self.candidate = z, base, replace(candidate)
        self.batch_header, self.grids, self.counters = batch_header, grids, counters
        self.phy = phy
        self.selection = None
        self.prior_selection = frequency_selection
        self.prior_validation = previous_validation
        self.results = {}
        self.validator = (phy._validate_physical_frequency_batched if batch_header
                          else phy._validate_physical_frequency)
        self.retained_bytes = z.nbytes
        counters['phdr_transaction_context_peak_bytes'] = z.nbytes
        self.screen = None
        if frontend == 'screened':
            from lrfhss_phdr_screen import ScreenContext
            self.screen = ScreenContext(z, counters, decimation=decimation)
        elif frontend != 'reference':
            raise ValueError('Unknown PHDR frontend mode')

    def frequency(self, span):
        c, phy, counts = self.candidate, self.phy, self.counters
        if c.source == 'sync':
            if self.selection is None:
                from lrfhss_frequency_reuse import select
                with measure(counts, 'cfo_refinement'):
                    self.selection, reused = select(self.z, self.base, c, phy.FS,
                        self.grids if self.grids is not None else False, self.prior_selection)
                counts['frequency_reuse_hits' if reused else 'frequency_reuse_misses'] += 1
            else:
                counts['frequency_reuse_hits'] += 1
                counts['phdr_intermediate_cache_hits'] += 1
            return dict(self.selection.frequencies)[250. if span is None else float(span)]
        with measure(counts, 'cfo_refinement'):
            nominal = c.nominal_hz if span is None and c.source == 'energy' else c.frequency_hz
            width = (5000 if c.source == 'energy' else 250) if span is None else span
            estimate = phy.estimate_frequency if self.grids is None else self.grids.estimate
            return estimate(self.z, c.header_start-self.base+2000, .20, nominal, width) if width else nominal

    def attempt(self, span):
        counts = self.counters
        wall, cpu = time.perf_counter(), time.process_time()
        freq = self.frequency(span)
        # The namespace cannot change while this context exists. Include the
        # exact frequency bytes: +0 and -0 are deliberately distinct keys.
        key = struct.pack('!d', float(freq))
        counts['phdr_logical_attempts'] += 1
        if self.selection is not None and self.prior_validation is not None:
            prior = self.prior_validation
            identity = ('physical-validator-v1', self.selection.identity, freq, bool(self.batch_header))
            if prior.identity == identity:
                self.results[key] = (deepcopy(prior.header), deepcopy(prior.detail), 0., 0.)
        hit = key in self.results
        if hit:
            h, detail, prior_wall, prior_cpu = self.results[key]
            h, detail = deepcopy(h), deepcopy(detail)
            counts['phdr_validation_reuse_hits'] += 1
            counts['phdr_mix_lowpass_reuse_hits'] += 1
            counts['phdr_exact_refinements_coalesced'] += 1
            counts['phdr_intermediate_cache_hits'] += 1
            counts['phdr_reuse_estimated_avoided_wall_s'] += prior_wall
            counts['phdr_reuse_estimated_avoided_cpu_s'] += prior_cpu
        else:
            counts['phdr_validation_reuse_misses'] += 1
            counts['phdr_mix_lowpass_reuse_misses'] += 1
            counts['phdr_intermediate_cache_misses'] += 1
            counts['phdr_physical_validation_executions'] += 1
            counts['phdr_physical_executions'] += 1
            started, started_cpu = time.perf_counter(), time.process_time()
            with measure(counts, 'physical_header_validation'):
                screen = None
                if self.screen is not None:
                    try:
                        screen = self.screen.screen(freq, 250., self.candidate.header_start-self.base)
                    except (FloatingPointError, OverflowError):
                        # Optional screening must not discard a hypothesis on
                        # numerical uncertainty. Allocation/programming errors
                        # remain visible worker failures, never hidden successes.
                        counts['phdr_screen_numerical_fallbacks'] += 1
                        screen = dict(reject=False, reason='numerical_exception')
                if screen is not None and screen['reject']:
                    counts['phdr_screen_rejections'] += 1
                    h = None
                    detail = dict(header_crc_ok=None, viterbi_metric=None,
                        sync_correlation=self.candidate.sync_correlation,
                        frequency_hz=freq, cfo_hz=freq-self.candidate.nominal_hz,
                        reason='header_correlation')
                else:
                    counts['phdr_screen_fallbacks'] += int(screen is not None)
                    counts['phdr_physical_mix_lowpass_executions'] += 1
                    h, detail = self.validator(self.z, self.base, self.candidate, freq)
            self.results[key] = (deepcopy(h), deepcopy(detail),
                time.perf_counter()-started, time.process_time()-started_cpu)
        if span is not None:
            counts['phdr_worker_local_refinements_attempted'] += 1
            counts['phdr_iq_snapshot_reuse'] += 1
        if self.selection is not None:
            from lrfhss_header_reuse import ValidationResult
            detail = deepcopy(detail)
            detail['frequency_selection'] = self.selection
            detail['previous_validation'] = ValidationResult(
                ('physical-validator-v1', self.selection.identity, freq, bool(self.batch_header)),
                deepcopy(h), {k:deepcopy(v) for k,v in detail.items() if k != 'frequency_selection'})
        return dict(span_hz=span, header=h, detail=detail, coalesced=hit,
                    wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu)


def evaluate(z, base, candidate, *, batch_header=False, grids=None,
             counters=None, allow_refinements=True, first_span=None, frequency_selection=None, previous_validation=None,
             frontend='reference', decimation=64):
    from collections import Counter
    counts = Counter() if counters is None else counters
    context = CandidateContext(z, base, candidate, batch_header=batch_header,
                               grids=grids, counters=counts, frequency_selection=frequency_selection, previous_validation=previous_validation,
                               frontend=frontend, decimation=decimation)
    counts['phdr_candidate_transactions'] += int(first_span is None)
    counts['phdr_transaction_worker_segments'] += 1
    attempts = [context.attempt(first_span)]
    if (attempts[0]['header'] is None and allow_refinements and
        candidate.source == 'sync' and (candidate.sync_correlation or 0.) >= .6):
        for span in ((100, 0) if first_span is None else (0,) if first_span == 100 else ()):
            attempts.append(context.attempt(span))
            if attempts[-1]['header'] is not None:
                break
    # Only small results are returned. All views/cache storage die before the
    # worker sends the terminal message authorizing shared-slot reuse.
    if context.screen is not None:
        context.screen.close()
    counts['phdr_transaction_context_final_bytes'] = 0
    return attempts
