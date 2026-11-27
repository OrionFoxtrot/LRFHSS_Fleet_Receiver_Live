"""Continuous header search for the configured three-header LR-FHSS profile.

No Qt, packet payload demodulation, or fleet/application state lives here.
The full-rate IQ ring and PacketTrack lifetimes belong to LiveDecoder and its
TrackManager. This front end keeps only half a second of decimated channels.
"""
from __future__ import annotations

from collections import Counter, deque, OrderedDict
from dataclasses import dataclass
import math
import time
from lrfhss_metrics import measured, measure, phase_cycles

import numpy as np
from scipy import signal

SYNC_WORD = 0x12AD101B
SYNC_SIGNS = np.array([2*((SYNC_WORD >> n) & 1)-1 for n in range(31, -1, -1)], dtype=float)


def sync_score(soft32):
    """Pearson correlation with the known MSB-first sync, insensitive to DC bias."""
    x = np.asarray(soft32, dtype=float)
    x = x - x.mean()
    y = SYNC_SIGNS - SYNC_SIGNS.mean()
    denom = float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(np.dot(x, y)/denom) if denom > 1e-12 else 0.0


@dataclass
class HeaderCandidate:
    candidate_id: int
    header_start: int
    physical_index: int                 # zero based: PHDR #1 -> 0
    nominal_hz: float
    frequency_hz: float
    source: str
    sync_correlation: float | None
    energy_ratio: float
    observed_sample: int
    cfo_limit_hz: float = 5000.0
    allowed_hop_ids: tuple[int, ...] = (256,)


@dataclass
class HeaderObservation:
    """A CRC/profile-validated physical header, before packet-time normalization."""
    candidate_id: int
    physical_index: int
    physical_start: int
    packet_start: int
    frequency_hz: float
    cfo_hz: float
    sync_correlation: float
    viterbi_metric: float
    info: dict
    source: str
    energy_ratio: float


class HeaderAcquisition:
    """Search all three known header channels, even in uninterrupted activity.

    A wide channelizer feeds a small CFO bank selected from spectral peaks.
    Each bank member is narrow-filtered and correlated against the 32 sync
    symbols on *every* decimated sample phase. Peaks propose candidates only;
    full-rate header CRC/profile/timing validation remains mandatory.

    Energy proposals on PHDR #2/#3 retain an onset fallback. PHDR #1 keeps the
    established LiveDecoder energy path so successful isolated packets retain
    exactly their original timing/frequency calibration.
    """
    DECIMATION = 64                     # 32 samples per symbol after decimation
    SEARCH_CORRELATION = 0.55
    CFO_SPAN_HZ = 4500.0
    MAX_CFO_PEAKS = 3
    MAX_PENDING = 256
    CONTEXT_S = 0.45

    def __init__(self, fs, spb, nominal_hz, trigger_ratio, emit, *,
                 scheduler="fifo", schedule_phase_s=None, clustering="legacy",
                 diagnostic_callback=None, noise_mode="adaptive", gate_mode="normal",
                 freeze_s=.5, reset_s=None, energy_reference="legacy", search_backend="reference",
                 carrier_search_hz=4500.0, _outer_span_hz=None, channel_map=None,
                 admission_ranks=True):
        if not isinstance(admission_ranks, bool):
            raise ValueError('admission_ranks must be boolean')
        self.admission_ranks = admission_ranks
        if not math.isfinite(carrier_search_hz) or not 4500 <= carrier_search_hz <= 25000:
            raise ValueError("Carrier search must be between 4500 and 25000 Hz")
        self.carrier_search_hz = float(carrier_search_hz)
        self._outer_span_hz = _outer_span_hz
        if _outer_span_hz is not None:
            # A 15.625 kHz channel cannot represent +/-20 kHz. The companion
            # search uses 62.5 kHz at the supported 1 MS/s capture rate.
            if spb % 16 or _outer_span_hz+2000 >= fs/32:
                raise ValueError("Wide carrier search exceeds channelizer bandwidth")
            self.DECIMATION = 16
            self.CFO_SPAN_HZ = float(_outer_span_hz)+250.0
        if scheduler not in ("fifo", "quality_age") or clustering not in ("legacy", "narrow"):
            raise ValueError("Unknown acquisition policy")
        if schedule_phase_s is not None and not math.isfinite(schedule_phase_s):
            raise ValueError("Schedule phase must be finite")
        if search_backend not in ("reference", "batched"):
            raise ValueError("Unknown acquisition search backend")
        self.search_backend = search_backend
        self.scheduler = scheduler
        if noise_mode not in ("adaptive", "frozen", "search_gated", "spectral20", "spectral50", "guard20"):
            raise ValueError("Unknown acquisition noise mode")
        if gate_mode not in ("normal", "weak_signal", "bypass", "bypass_all_energy"):
            raise ValueError("Unknown diagnostic search gate")
        if energy_reference not in ("legacy", "background"):
            raise ValueError("Unknown coarse energy-onset reference")
        if freeze_s < 0 or (reset_s is not None and reset_s < freeze_s):
            raise ValueError("Invalid acquisition diagnostic time")
        self.diagnostic_callback = diagnostic_callback
        # Set only by the IQ owner. This pumps already-admitted worker I/O;
        # it never changes acquisition, noise state, candidates or tracks.
        self.progress_callback = None
        self.noise_mode, self.gate_mode = noise_mode, gate_mode
        self.freeze_sample = round(freeze_s*fs)
        self.reset_sample = None if reset_s is None else round(reset_s*fs)
        self._reset_done = False
        self._calibration_noise = [None]*len(nominal_hz)
        self._diagnostic_block = None
        self.schedule_phase_s = schedule_phase_s
        self.clustering = clustering
        self.validation_sequence = 0
        self.fs, self.spb = fs, spb
        self.nominal_hz = tuple(float(f) for f in nominal_hz)
        self.channel_map = None if channel_map is None else tuple(tuple(a) for a in channel_map)
        if self.channel_map is not None:
            if len(self.channel_map) != len(self.nominal_hz) or any(not a for a in self.channel_map):
                raise ValueError('Every search channel needs explicit (hop ID, copy) aliases')
            if any(not 0 <= h <= 511 or k not in (0, 1, 2) for a in self.channel_map for h, k in a):
                raise ValueError('Invalid search-channel alias')
        self.trigger_ratio = trigger_ratio
        self.emit = emit
        cutoff = 5000 if _outer_span_hz is None else _outer_span_hz+2000
        self.sos = signal.butter(6, cutoff, fs=fs, output="sos")
        self.zi = [np.zeros((len(self.sos), 2), complex) for _ in nominal_hz]
        self.history = [np.empty(0, complex) for _ in nominal_hz]
        self.history_start = 0
        self.noise = [None]*len(nominal_hz)
        self.energy_reference = energy_reference
        self.energy_references = [None]*len(nominal_hz)
        from lrfhss_noise import SpectralNoise
        self.noise_estimators = [SpectralNoise(fs, self.DECIMATION, self.sos,
            quantile=.5 if noise_mode == "spectral50" else .2, guard=noise_mode == "guard20") for _ in nominal_hz]
        self.search_energy_ratio = 2.0
        if _outer_span_hz is not None:
            # Keep the required SIGNAL excess equal to the original 5 kHz
            # channel's noise power. Applying a 2x gate to the wider noise
            # bandwidth would unnecessarily raise the acquisition floor.
            impulse = np.zeros(8192)
            impulse[0] = 1
            reference_sos = signal.butter(6, 5000, fs=fs, output="sos")
            reference_gain = float(np.sum(signal.sosfilt(reference_sos, impulse)**2))
            self.search_energy_ratio = 1.0+reference_gain/self.noise_estimators[0].channel_gain
        # A narrow hop can raise the integrated channel power only slightly.
        # Use a lower threshold at the sync peak within the standard CFO range.
        self.candidate_energy_ratio = (1.22 if gate_mode == "weak_signal" and _outer_span_hz is None
                                       else self.search_energy_ratio)
        self.burst_onset = [None]*len(nominal_hz)
        self.burst_end = [-1]*len(nominal_hz)
        self.pending: list[HeaderCandidate] = []
        self.pending_revision = 0
        self.recent = deque(maxlen=2048)
        self._recent_bins = {}
        self.counts = Counter()
        self.processing = Counter()
        self._next_id = 1
        self._mixer_length = 0
        self._mixers = []
        self.low_fs = fs/self.DECIMATION
        self.low_spb = spb//self.DECIMATION
        self.narrow_sos = signal.butter(4, 300, fs=self.low_fs, output="sos")
        self.kernel = np.zeros(31*self.low_spb+1)
        self.kernel[::self.low_spb] = SYNC_SIGNS-SYNC_SIGNS.mean()
        self.ones = np.zeros(len(self.kernel))
        self.ones[::self.low_spb] = 1
        self.kernel_energy = float(np.sum(self.kernel**2))
        from lrfhss_correlation import AcquisitionCorrelations
        self._correlations = AcquisitionCorrelations(self.kernel,self.ones)
        self._search_grids = OrderedDict()
        self.wide_search = None
        if carrier_search_hz > 4500 and _outer_span_hz is None:
            self.wide_search = HeaderAcquisition(fs, spb, nominal_hz, trigger_ratio,
                lambda *args, **kwargs: None, scheduler=scheduler,
                schedule_phase_s=schedule_phase_s, clustering=clustering,
                noise_mode=noise_mode, gate_mode=gate_mode, freeze_s=freeze_s,
                reset_s=reset_s, energy_reference=energy_reference,
                search_backend=search_backend, _outer_span_hz=carrier_search_hz,
                channel_map=self.channel_map)
            # Numerical proposals share the owner's deduplication, admission
            # queue and validation budget. The companion owns no packet state.
            self.wide_search._offer = self._offer_wide

    def _offer_wide(self, start, index, frequency, source, correlation, energy, sample):
        # Keep the established narrow path's energy timing. Outer carriers
        # require an observed sync peak before consuming a header job.
        if source != "sync" or abs(frequency-self.nominal_hz[index]) <= 4500:
            return
        self.counts["wide_candidates_proposed"] += 1
        self._offer(start, index, frequency, source, correlation, energy, sample,
                    self.carrier_search_hz+500.0)

    @measured("candidate_deduplication")
    def _offer(self, start, index, frequency, source, correlation, energy, sample,
               cfo_limit_hz=5000.0):
        nominal = self.nominal_hz[index]
        if self.channel_map is None:
            return self._offer_candidate(start, index, nominal, (256,), frequency,
                                         source, correlation, energy, sample, cfo_limit_hz)
        aliases = self.channel_map[index]
        for physical_index in sorted({k for _, k in aliases}):
            ids = tuple(h for h, k in aliases if k == physical_index)
            self._offer_candidate(start, physical_index, nominal, ids, frequency,
                                  source, correlation, energy, sample, cfo_limit_hz)

    def _offer_candidate(self, start, index, nominal, allowed_ids, frequency,
                         source, correlation, energy, sample, cfo_limit_hz):
        detected_wall = time.perf_counter()
        self.counts["candidate_peaks"] += 1
        fields = dict(candidate_sample=int(start), physical_header=index+1,
                      inferred_packet_start=int(start-index*114*self.spb),
                      frequency_hz=float(frequency), cfo_hz=float(frequency-nominal),
                      source=source, sync_correlation=correlation, energy_ratio=float(energy),
                      header_crc_ok=None, viterbi_metric=None,
                      carrier_search_path="wide" if cfo_limit_hz > 5000 else "narrow")
        if self._diagnostic_block is not None:
            row = self._diagnostic_block
            row["candidates_offered"] += 1
            row[source+"_candidates_offered"] += 1
            self.diagnostic_callback(dict(kind="raw_candidate",feed_start_sample=row["feed_start_sample"],
                                          feed_end_sample=sample,**fields))
        # Local nonmaximum suppression only; other times/CFOs/channels keep searching.
        for c in self._nearby_recent(start, index):
            if (c.physical_index == index and c.nominal_hz == nominal and abs(c.header_start-start) <= 2*self.spb
                    and abs(c.frequency_hz-frequency) <= (250 if self.clustering == "legacy" else 60)):
                # Narrow mode retains nearby frequency hypotheses; it does not claim radio identity.
                self.counts["peak_duplicates"] += 1
                self.emit("header_candidate_duplicate", sample, candidate_id=c.candidate_id,
                          reason="same_sync_peak_or_energy_onset",
                          retained_frequency_hz=c.frequency_hz,
                          frequency_delta_hz=float(frequency-c.frequency_hz),
                          timing_delta_samples=int(start-c.header_start), **fields)
                return
        if len(self.pending) >= self.MAX_PENDING:
            self.counts["candidate_capacity_rejections"] += 1
            self.emit("header_candidate_rejected", sample, reason="candidate_capacity", **fields)
            return
        c = HeaderCandidate(self._next_id, int(start), index, nominal,
                            float(frequency), source, correlation, float(energy), sample,
                            float(cfo_limit_hz), allowed_ids)
        self._next_id += 1
        c.detected_wall = detected_wall
        c.admitted_wall = time.perf_counter()
        c.eligible_wall = None
        self.pending.append(c)
        self.pending_revision += 1
        self.counts["pending_high_watermark"] = max(self.counts["pending_high_watermark"], len(self.pending))
        self._remember(c)
        self.counts["candidates_queued"] += 1
        self.counts["wide_candidates_queued"] += int(cfo_limit_hz > 5000)
        self.emit("header_candidate", sample, candidate_id=c.candidate_id,
                  detected_wall=c.detected_wall, admitted_wall=c.admitted_wall,
                  queue_fifo_rank=len(self.pending),
                  queue_priority_rank=1+sum(self._priority(other,sample)>priority
                      for priority in [self._priority(c,sample)]
                      for other in self.pending if other is not c)
                      if getattr(self, 'admission_ranks', True) else None, **fields)

    def _nearby_recent(self, start, index):
        # Exact index for the existing +/-2-symbol predicate. Neighboring
        # buckets contain every possible match. Preserve original insertion
        # order (candidate ID), including ties; no hypothesis is pruned.
        bucket = int(start)//(2*self.spb)
        candidates = [c for key in (bucket-1,bucket,bucket+1)
                      for c in self._recent_bins.get((index,key),{}).values()]
        return sorted(candidates,key=lambda c:c.candidate_id)

    def _remember(self, c):
        if len(self.recent) == self.recent.maxlen:
            old = self.recent[0]
            key = (old.physical_index,old.header_start//(2*self.spb))
            values = self._recent_bins[key]
            del values[old.candidate_id]
            if not values:del self._recent_bins[key]
        self.recent.append(c)
        key = (c.physical_index,c.header_start//(2*self.spb))
        self._recent_bins.setdefault(key,{})[c.candidate_id] = c

    def feed(self, z, start):
        end = start+len(z)
        # Diagnostic oracle only: restore the measured quiet calibration at an
        # explicitly supplied sample, without touching tracks, filters or jobs.
        reset_now = self.reset_sample is not None and not self._reset_done and start >= self.reset_sample
        if reset_now:
            self.noise = self._calibration_noise.copy()
            self._reset_done = True
        # Cache the regular 50 ms oscillators; the start phase keeps channel
        # filtering continuous at arbitrary stream/decimation boundaries.
        if len(z) != self._mixer_length:
            n = np.arange(len(z), dtype=float)
            self._mixers = [np.exp(-2j*np.pi*f*n/self.fs) for f in self.nominal_hz]
            self._mixer_length = len(z)
        offset = (-start) % self.DECIMATION
        first = start+offset
        for index, nominal in enumerate(self.nominal_hz):
            phase = np.exp(-2j*np.pi*phase_cycles(nominal,start,self.fs))
            mixed = z*(self._mixers[index]*phase)
            y, self.zi[index] = signal.sosfilt(self.sos, mixed, zi=self.zi[index])
            low = y[offset::self.DECIMATION]
            old = self.history[index]
            history = np.concatenate((old, low))
            history_start = first-len(old)*self.DECIMATION
            keep = int(self.CONTEXT_S*self.low_fs)
            if len(history) > keep:
                history_start += (len(history)-keep)*self.DECIMATION
                history = history[-keep:]
            self.history[index] = history
            self.history_start = history_start
            p = np.abs(y)**2
            cp = float(np.mean(p))
            history_power = float(np.mean(np.abs(history)**2))
            # Preserve the established coarse onset calibration independently
            # of the background estimate used by continuous sync acquisition.
            # This reference never gates _search or its correlation energy test.
            energy_reference = self.energy_references[index]
            if energy_reference is None:energy_reference=max(cp,1e-16)
            if cp<energy_reference*self.trigger_ratio:
                energy_reference=max(.97*energy_reference+.03*cp,1e-16)
            self.energy_references[index]=energy_reference
            noise = self.noise[index]
            noise_before = noise
            if noise is None:
                noise = max(cp, 1e-16)
            update_allowed = cp < noise*self.trigger_ratio
            if self.noise_mode == "frozen" and end > self.freeze_sample:
                update_allowed = False
            elif self.noise_mode == "search_gated":
                update_allowed = cp < self.search_energy_ratio*noise and history_power < self.search_energy_ratio*noise
            if self.noise_mode.startswith("spectral") or self.noise_mode == "guard20":
                estimate = self.noise_estimators[index].update(low, len(z)/self.fs)
                if estimate is not None:
                    noise = max(cp, estimate) if noise_before is None else estimate
                update_allowed = estimate is not None
            elif update_allowed:
                noise = max(.97*noise+.03*cp, 1e-16)
            self.noise[index] = noise
            if self.energy_reference != "legacy" or self.noise_mode not in ("spectral20","spectral50","guard20"):
                energy_reference=noise
            if end <= self.freeze_sample:
                self._calibration_noise[index] = noise
            if self.diagnostic_callback is not None:
                self._diagnostic_block = dict(kind="feed_channel",feed_start_sample=start,
                    feed_end_sample=end,physical_header=index+1,nominal_hz=nominal,
                    channel_power=cp,history_power=history_power,noise_before=noise_before,
                    noise_after=noise,power_noise_ratio=cp/noise,history_noise_ratio=history_power/noise,
                    noise_updated=bool(update_allowed),noise_reset=bool(reset_now),
                    low_gate_ratio=self.search_energy_ratio,low_gate_power=self.search_energy_ratio*noise,normal_gate_pass=bool(history_power>self.search_energy_ratio*noise),
                    search_entered=False,spectral_peaks_found=0,cfo_peaks_retained=0,
                    cfo_peaks_searched=0,max_correlation=None,correlation_peaks_above_threshold=0,
                    correlation_energy_rejections=0,candidates_offered=0,
                    energy_candidates_offered=0,sync_candidates_offered=0,
                    noise_mode=self.noise_mode,gate_mode=self.gate_mode,
                    energy_reference_mode=self.energy_reference,energy_onset_reference=energy_reference,
                    spectral_observation=self.noise_estimators[index].observation)
                if self.channel_map is not None:
                    self._diagnostic_block.update(physical_header=None, search_channel_index=index,
                                                  channel_aliases=self.channel_map[index])
            self.counts["search_gate_suppressed"] += int(
                len(history) >= int(.18*self.low_fs) and history_power <= self.search_energy_ratio*noise)
            # Coarse alternate-header fallback. Correlation search does NOT use
            # these burst boundaries as exclusion intervals.
            alternate_copy = bool(index) if self.channel_map is None else any(k > 0 for _, k in self.channel_map[index])
            coarse_window = self.spb
            coarse_step = 1
            coarse_start = start
            if alternate_copy and cp > energy_reference*self.trigger_ratio and len(p) >= coarse_window:
                cs = np.cumsum(np.r_[0.0, p])
                ma = (cs[coarse_window:]-cs[:-coarse_window])/coarse_window
                hits = np.flatnonzero(ma > energy_reference*self.trigger_ratio)
                if len(hits):
                    onset = coarse_start+int(hits[0])*coarse_step
                    if self.burst_onset[index] is not None and onset <= self.burst_end[index]+self.spb:
                        onset = self.burst_onset[index]
                    else:
                        self.burst_onset[index] = onset
                        self._offer(onset+round(4.25*self.spb), index, nominal,
                                    "energy", None, cp/energy_reference, end)
                    self.burst_end[index] = coarse_start+int(hits[-1])*coarse_step+self.spb
            # This low gate saves FFT/correlation work on quiet channels. The
            # established PHDR #1 energy detector still operates independently.
            if len(history) >= int(.18*self.low_fs) and (history_power > self.search_energy_ratio*noise or self.gate_mode != "normal"):
                if self._diagnostic_block is not None:self._diagnostic_block["search_entered"] = True
                self._search(history, history_start, index, noise, end)
            if self._diagnostic_block is not None:
                self.diagnostic_callback(self._diagnostic_block)
                self._diagnostic_block = None
            if self.progress_callback is not None:
                self.progress_callback()

        if self.wide_search is not None:
            wide = self.wide_search
            wide.counts.clear()
            wide.processing.clear()
            wide.feed(z, start)
            for key, value in wide.counts.items():
                self.counts["wide_"+key] += value
            for key, value in wide.processing.items():
                self.processing["wide_"+key] += value

    @measured("candidate_peak_generation")
    def _search(self, x, base, index, noise, sample):
        self.counts["search_windows"] += 1
        nfft = 1 << int(math.ceil(math.log2(len(x))))
        key=(len(x),nfft)
        if key not in self._search_grids:
            self._search_grids[key]=(np.hanning(len(x)),np.fft.fftshift(np.fft.fftfreq(nfft,1/self.low_fs)),np.arange(len(x))/self.low_fs)
            if len(self._search_grids)>16:
                self._search_grids.popitem(last=False)
        window,freq,t=self._search_grids[key]
        with measure(self.processing,"spectral_peak_search"):
            power = np.abs(np.fft.fftshift(np.fft.fft(x*window, nfft)))**2
        peaks, _ = signal.find_peaks(power, distance=max(1, round(180/(self.low_fs/nfft))))
        peaks = peaks[np.abs(freq[peaks]) <= self.CFO_SPAN_HZ]
        if self._outer_span_hz is not None:
            peaks = peaks[np.abs(freq[peaks]) > 4500]
        if self._diagnostic_block is not None:self._diagnostic_block["spectral_peaks_found"] += len(peaks)
        peaks = peaks[np.argsort(power[peaks])[::-1][:self.MAX_CFO_PEAKS]]
        if self._diagnostic_block is not None:self._diagnostic_block["cfo_peaks_retained"] += len(peaks)
        if not len(peaks):
            return
        half = self.low_spb//2
        retained = [peak for peak in peaks if power[peak] >= power[peaks[0]]*.08]
        batch_rows = None
        if self.search_backend == "batched" and retained:
            # Same scalar expression/order, broadcast only across CFO rows.
            cfos = freq[retained][:, None]
            bb = signal.sosfiltfilt(self.narrow_sos,
                x[None, :]*np.exp(-2j*np.pi*cfos*t[None, :]), axis=-1)
            # NumPy uses a different complex multiply loop for the strided
            # two-dimensional view. Keep that phase product rowwise to retain
            # reference bits at correlation thresholds (observed delta 4e-16).
            phase = np.stack([np.angle(row[half:]*np.conj(row[:-half])) for row in bb])
            soft = np.clip(phase/(np.pi/4), -1, 1)
            soft[np.abs(phase) > np.pi/2] = 0
            soft = np.pad(soft, ((0, 0), (half//2, half-half//2)))
            batch_rows = self._correlations.batched(soft)
        row_index = 0
        for peak in peaks:
            if power[peak] < power[peaks[0]]*.08:
                continue
            if self._diagnostic_block is not None:self._diagnostic_block["cfo_peaks_searched"] += 1
            cfo = float(freq[peak])
            if batch_rows is None:
                bb = signal.sosfiltfilt(self.narrow_sos, x*np.exp(-2j*np.pi*cfo*t))
                phase = np.angle(bb[half:]*np.conj(bb[:-half]))
                soft = np.clip(phase/(np.pi/4), -1, 1)
                soft[np.abs(phase) > np.pi/2] = 0
                soft = np.pad(soft, (half//2, half-half//2))
                numerator, summed, squared = self._correlations(soft)
            else:
                numerator, summed, squared = (a[row_index] for a in batch_rows)
            carrier_row = bb if batch_rows is None else bb[row_index]
            row_index += 1
            mean = summed/32
            variance = squared-32*mean**2
            correlation = numerator/np.sqrt(np.maximum(variance, 1e-12)*self.kernel_energy)
            candidates, _ = signal.find_peaks(correlation, height=self.SEARCH_CORRELATION,
                                               distance=2*self.low_spb)
            if self._diagnostic_block is not None:
                row = self._diagnostic_block
                row["max_correlation"] = max(row["max_correlation"] or -1.,float(np.max(correlation)))
                row["correlation_peaks_above_threshold"] += len(candidates)
            for k in candidates:
                # Discard filter/record edges; all full header samples will be
                # checked again before decoding. Sync centers are soft[41:73].
                if k < self.low_spb or k+32*self.low_spb > len(x)-self.low_spb:
                    continue
                if self._outer_span_hz is not None:
                    # Energy elsewhere in the 44 kHz channel must not promote
                    # phase correlations in an empty 300 Hz carrier filter.
                    # This is below the existing wide-channel energy gate.
                    local_power = float(np.mean(np.abs(carrier_row[k:k+32*self.low_spb])**2))
                    if local_power < .10*noise:
                        self.counts["narrow_energy_rejections"] += 1
                        continue
                energy = float(np.mean(np.abs(x[k:k+32*self.low_spb])**2)/noise)
                if energy < self.candidate_energy_ratio and self.gate_mode != "bypass_all_energy":
                    if self._diagnostic_block is not None:self._diagnostic_block["correlation_energy_rejections"] += 1
                    continue
                anchor = base+int(k)*self.DECIMATION-41*self.spb
                if anchor < 0:
                    continue
                self._offer(anchor, index, self.nominal_hz[index]+cfo, "sync",
                            float(correlation[k]), energy, sample)

    def priority_components(self, c, sample):
        age = max(0., (sample-c.observed_sample)/self.fs)
        quality = min(1., max(0., c.sync_correlation or 0.))
        # Unknown fleet phase is the default. Operator phase refers to first
        # packet's PHDR1 sample time. No CSID or ground truth enters this path.
        boost = 0.
        if self.schedule_phase_s is not None:
            packet_time = (c.header_start-c.physical_index*114*self.spb)/self.fs
            residual = abs((packet_time-self.schedule_phase_s+.05) % .1-.05)
            boost = .10*max(0., 1.-residual/.025)
        return dict(quality=quality, age_s=age, age_boost=.5*age,
                    schedule_boost=boost, overdue=age>=1.)

    def _priority(self, c, sample):
        # Same tuple/order as priority_components; avoid allocating a dict and
        # computing unused quality/phase terms for an already-overdue item.
        age = max(0., (sample-c.observed_sample)/self.fs)
        if age >= 1.:
            return (1,-c.observed_sample,-c.candidate_id)
        quality = min(1.,max(0.,c.sync_correlation or 0.))
        boost = 0.
        if self.schedule_phase_s is not None:
            packet_time = (c.header_start-c.physical_index*114*self.spb)/self.fs
            residual = abs((packet_time-self.schedule_phase_s+.05) % .1-.05)
            boost = .10*max(0.,1.-residual/.025)
        return (0,quality+.5*age+boost,-c.candidate_id)

    @measured("candidate_backlog_handling")
    def ready(self, sample, limit=None):
        eligible = [c for c in self.pending if sample >= c.header_start+int(.335*self.fs)]
        self.counts["validation_backlog_high_watermark"] = max(
            self.counts["validation_backlog_high_watermark"], len(eligible))
        fifo_rank={c.candidate_id:i+1 for i,c in enumerate(eligible)}
        ranked=sorted(eligible,key=lambda c:self._priority(c,sample),reverse=True)
        quality_rank={c.candidate_id:i+1 for i,c in enumerate(ranked)}
        ordered=ranked if self.scheduler=="quality_age" else eligible
        chosen=ordered if limit is None else ordered[:max(0,limit)]
        ids={c.candidate_id for c in chosen}
        self.pending=[c for c in self.pending if c.candidate_id not in ids]
        if ids:
            self.pending_revision += 1
        deferred=[c for c in eligible if c.candidate_id not in ids]
        self.counts["validation_backlog_depth"]=len(deferred)
        if deferred:
            self.counts["deferred_candidate_observations"]+=len(deferred)
            self.emit("header_validation_deferred",sample,depth=len(deferred),
                      oldest_age_samples=max(sample-c.observed_sample for c in deferred))
        for c in chosen:
            self.validation_sequence+=1
            components=self.priority_components(c,sample)
            self.counts["candidate_max_age_samples"]=max(
                self.counts["candidate_max_age_samples"],sample-c.observed_sample)
            self.counts["overdue_candidates_serviced"]+=int(components['overdue'])
            self.emit("candidate_scheduled",sample,candidate_id=c.candidate_id,
                scheduler=self.scheduler,validation_sequence=self.validation_sequence,
                fifo_rank=fifo_rank[c.candidate_id],priority_rank=quality_rank[c.candidate_id],
                candidate_age_samples=sample-c.observed_sample,**components)
        return chosen

    def finish(self, sample):
        for c in self.pending:
            self.emit("header_candidate_rejected", sample, candidate_id=c.candidate_id,
                      candidate_sample=c.header_start, physical_header=c.physical_index+1,
                      reason="end_of_stream_before_header_validation")
        self.pending.clear()
        self.pending_revision += 1
