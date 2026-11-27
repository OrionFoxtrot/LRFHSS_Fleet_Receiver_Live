"""Packet hypotheses and bounded lifecycle accounting for one IQ stream.

This module has no Qt or RF-demodulator dependency. Times are absolute complex
sample indices belonging to one LiveDecoder. A track is not a ChipSat identity:
CSID becomes available only after the PHY and application payload are decoded.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from lrfhss_live import Header
    from lrfhss_acquisition import HeaderObservation


@dataclass
class PacketTrack:
    track_id: int
    detected_sample: int
    created_sample: int
    acquisition_frequency_hz: float
    acquisition_energy_ratio: float | None = None
    status: str = "pending_header"
    header: Header | None = None
    cfo_hz: float | None = None
    payload_start: int | None = None
    expected_end_sample: int | None = None
    expected_hops_hz: tuple[float, ...] = ()
    last_processed_sample: int = 0
    finished_sample: int | None = None
    reason: str | None = None
    duplicate_of: int | None = None
    headers: dict[int, HeaderObservation] = field(default_factory=dict)


class TrackManager:
    """Own active hypotheses, bounded history, duplicate checks, and events.

    Duplicate comparisons are local in onset/frequency, then in decoded
    header timing/frequency/profile. An existing packet never reserves the
    acquisition channel for its full duration. These tolerances are association
    tolerances, not a promise to resolve colliding nearly coincident headers.
    """
    def __init__(self, *, event_callback: Callable[[dict], None] | None = None,
                 max_active: int = 256, history_limit: int = 512,
                 candidate_time_tolerance: int = 2048,
                 header_time_tolerance: int = 2048,
                 frequency_tolerance_hz: float = 40.0):
        if max_active < 1 or history_limit < 1:
            raise ValueError("Track limits must be positive")
        self.refinement_revision = 0
        self._refinement_geometry = None
        self.active_tracks: dict[int, PacketTrack] = {}
        self.completed_tracks: deque[PacketTrack] = deque(maxlen=history_limit)
        self.expired_tracks: deque[PacketTrack] = deque(maxlen=history_limit)
        self.rejected_tracks: deque[PacketTrack] = deque(maxlen=history_limit)
        self._recent: deque[PacketTrack] = deque(maxlen=history_limit)
        self.payload_tracks = {}
        self.max_active = max_active
        self.candidate_time_tolerance = candidate_time_tolerance
        self.header_time_tolerance = header_time_tolerance
        self.frequency_tolerance_hz = frequency_tolerance_hz
        # FFT peak estimates on different coded headers include modulation
        # bias. One half-symbol-rate covers that bias, without fleet dead time.
        self.observation_cfo_tolerance_hz = 250.0
        self.event_callback = event_callback
        self.counts = Counter()
        self._next_id = 1

    def _emit(self, event: str, sample: int, track: PacketTrack | None = None, **fields):
        if self.event_callback is not None:
            record = {"event": event, "sample": sample,
                      "active_tracks": len(self.active_tracks), **fields}
            if track is not None:
                record.update(track_id=track.track_id, detected_sample=track.detected_sample,
                              status=track.status)
            self.event_callback(record)

    def _geometry_changed(self):
        self.refinement_revision += 1
        self._refinement_geometry = None

    def refinement_geometry(self):
        """Read-only indexes shared by retry scheduling and SIC pressure."""
        if self._refinement_geometry is None:
            originals = tuple(t.detected_sample + 8704 for t in self.active_tracks.values() if t.header is None)
            known = tuple(sorted(t.header.start for t in self._known() if t.header is not None))
            self._refinement_geometry = originals, known
        return self._refinement_geometry

    def _known(self):
        yield from self.active_tracks.values()
        yield from self.payload_tracks.values()
        yield from self._recent

    # A packet whose payload was never published because only SIC had found its header
    # and that attempt failed its PHY CRC (Batch 10.4k, review 2026-10-02 finding 2).
    REOPENABLE_REASONS = frozenset({"sic_payload_crc_failed"})

    def _associable(self, existing, source):
        """May a header from ``source`` join ``existing``? A primary header never joins a
        retired SIC-only track whose payload failed unpublished: that packet is still
        undecoded, so the primary copy starts a fresh track. A SIC header still joins it,
        so cancellation cannot retry the same failure in a loop."""
        return not (source != "sic" and existing.status == "rejected"
                    and existing.reason in self.REOPENABLE_REASONS)

    def offer_candidate(self, onset: int, sample: int, frequency_hz: float) -> PacketTrack | None:
        self.counts["candidate_observations"] += 1
        for existing in self._known():
            if (abs(onset - existing.detected_sample) <= self.candidate_time_tolerance
                    and abs(frequency_hz - existing.acquisition_frequency_hz) <= self.frequency_tolerance_hz):
                self.counts["candidate_duplicates"] += 1
                self._emit("candidate_duplicate", sample, existing, observed_onset=onset)
                return None
        if len(self.active_tracks) >= self.max_active:
            self.counts["capacity_rejections"] += 1
            self._emit("capacity_rejected", sample, observed_onset=onset)
            return None
        track = PacketTrack(self._next_id, onset, sample, frequency_hz,
                            last_processed_sample=sample)
        self._next_id += 1
        self.active_tracks[track.track_id] = track
        self._geometry_changed()
        self.counts["tracks_created"] += 1
        self.counts["max_active_tracks"] = max(self.counts["max_active_tracks"], len(self.active_tracks))
        self._emit("candidate_created", sample, track)
        return track

    @staticmethod
    def _profile(header):
        return tuple(header.info.get(k) for k in (
            "payload_len", "modulation", "cr", "grid", "hop", "bw_code", "hop_seq_id"))

    def record_observation(self, track, observation, sample, association):
        old = track.headers.get(observation.physical_index)
        if old is None or observation.viterbi_metric < old.viterbi_metric:
            track.headers[observation.physical_index] = observation
        self.counts["physical_headers_validated"] += 1
        if association == "existing_track":
            self.counts["repeated_headers_associated"] += 1
        self._emit("header_observation", sample, track,
                   candidate_id=observation.candidate_id,
                   candidate_sample=observation.physical_start,
                   physical_header=observation.physical_index+1,
                   inferred_packet_start=observation.packet_start,
                   frequency_hz=observation.frequency_hz, cfo_hz=observation.cfo_hz,
                   sync_correlation=observation.sync_correlation,
                   viterbi_metric=observation.viterbi_metric, header_crc_ok=True,
                   physical_header_fields=observation.info.copy(),
                   source=observation.source, energy_ratio=observation.energy_ratio,
                   packet_timing_residual_samples=observation.packet_start-track.header.start,
                   cfo_residual_hz=observation.cfo_hz-track.cfo_hz,
                   association=association, observed_headers=sorted(k+1 for k in track.headers))

    def observe_header(self, header, observation, *, sample, payload_start, expected_end, hops_hz,
                       candidate_track=None):
        """Associate a validated physical header before allocating a new track.

        The caller normalizes header.start/freq_hz to the first-header reference
        used by the established payload decoder. The original physical header
        (including its countdown index) is retained in the observation.
        """
        for existing in self._known():
            other = existing.header
            if (other is not None and self._associable(existing, observation.source)
                    and self._profile(header) == self._profile(other)
                    and abs(header.start-other.start) <= self.header_time_tolerance
                    and abs(observation.cfo_hz-existing.cfo_hz) <= self.observation_cfo_tolerance_hz):
                if candidate_track is not None and candidate_track is not existing:
                    candidate_track.duplicate_of = existing.track_id
                    self.counts["header_duplicates"] += 1
                    self.retire(candidate_track, "duplicate", sample, "same_packet_header")
                self.record_observation(existing, observation, sample, "existing_track")
                return existing
        # A sync candidate can validate an energy hypothesis already in flight.
        track = candidate_track or next((t for t in self.active_tracks.values()
                      if t.header is None
                      # Legacy energy onset -> first-header symbol reference.
                      and abs(t.detected_sample+8704-header.start) <= self.header_time_tolerance), None)
        if track is None:
            if len(self.active_tracks) >= self.max_active:
                self.counts["capacity_rejections"] += 1
                self._emit("header_candidate_rejected", sample,
                           candidate_id=observation.candidate_id,
                           candidate_sample=observation.physical_start,
                           physical_header=observation.physical_index+1,
                           reason="track_capacity", header_crc_ok=True, source=observation.source)
                return None
            track = PacketTrack(self._next_id, observation.physical_start, sample,
                                observation.frequency_hz, last_processed_sample=sample)
            self._next_id += 1
            self.active_tracks[track.track_id] = track
            self._geometry_changed()
            self.counts["tracks_created"] += 1
            self.counts["max_active_tracks"] = max(self.counts["max_active_tracks"], len(self.active_tracks))
            self._emit("candidate_created", sample, track, source=observation.source)
        accepted = self.accept_header(track, header, sample=sample, cfo_hz=observation.cfo_hz,
                                      payload_start=payload_start, expected_end=expected_end, hops_hz=hops_hz,
                                      source=observation.source)
        if accepted:
            self.record_observation(track, observation, sample, "new_track")
            return track
        return None

    def accept_header(self, track: PacketTrack, header: Header, *, sample: int,
                      cfo_hz: float, payload_start: int, expected_end: int,
                      hops_hz, source=None) -> bool:
        self.counts["valid_header_observations"] += 1
        for existing in self._known():
            other = existing.header
            if existing is track or other is None or not self._associable(existing, source):
                continue
            if (abs(header.start - other.start) <= self.header_time_tolerance
                    and abs(header.freq_hz - other.freq_hz) <= self.frequency_tolerance_hz
                    and self._profile(header) == self._profile(other)):
                track.duplicate_of = existing.track_id
                self.counts["header_duplicates"] += 1
                self.retire(track, "duplicate", sample, "same_decoded_header")
                return False
        track.header = header
        self._geometry_changed()
        track.cfo_hz = float(cfo_hz)
        track.payload_start = payload_start
        track.expected_end_sample = expected_end
        track.expected_hops_hz = tuple(float(h) for h in hops_hz)
        track.status = "pending_payload"
        track.last_processed_sample = sample
        self.counts["headers_accepted"] += 1
        valid = sum(t.header is not None for t in self.active_tracks.values())
        self.counts["max_valid_tracks"] = max(self.counts["max_valid_tracks"], valid)
        self._emit("header_valid", sample, track, header_start=header.start, header=header.info.copy(),
                   cfo_hz=track.cfo_hz, expected_end_sample=expected_end, valid_tracks=valid)
        return True

    def reanchor(self, track, header, *, cfo_hz, payload_start, expected_end, hops_hz, sample, reason):
        """Replace a track's packet estimate (Batch 10.4k: a primary header confirming a
        SIC-found packet). The other observations stay in ``track.headers`` and become
        payload-retry alternates relative to the new estimate."""
        old = track.header
        track.header = header
        self._geometry_changed()
        track.cfo_hz = float(cfo_hz)
        track.payload_start = payload_start
        track.expected_end_sample = expected_end
        track.expected_hops_hz = tuple(float(h) for h in hops_hz)
        self.counts["tracks_reanchored"] += 1
        self._emit("track_reanchored", sample, track, reason=reason,
                   old_header_start=None if old is None else old.start, header_start=header.start,
                   old_frequency_hz=None if old is None else old.freq_hz, frequency_hz=header.freq_hz)

    def reopen_payload(self, track, sample, reason):
        """A queued track goes back to payload admission (its job was withdrawn or superseded)."""
        if self.payload_tracks.pop(track.track_id, None) is None:
            raise RuntimeError("only a queued payload track can be reopened")
        self.active_tracks[track.track_id] = track
        self._geometry_changed()
        track.status = "pending_payload"
        self.counts["tracks_reopened"] += 1
        self._emit("track_reopened", sample, track, reason=reason)

    def revive(self, track, sample, reason):
        """A retired track whose decode failed comes back for another payload attempt (Batch
        10.4l: a SIC-found packet with a new primary estimate). Its earlier completion stays
        counted; the next one is counted again, as the publication ledger does."""
        if track.track_id in self.active_tracks or track.track_id in self.payload_tracks:
            raise RuntimeError("only a retired track can be revived")
        try:
            self._recent.remove(track)
        except ValueError:
            pass
        self.active_tracks[track.track_id] = track
        self._geometry_changed()
        track.status, track.reason, track.finished_sample = "pending_payload", None, None
        self.counts["tracks_revived"] += 1
        self._emit("track_revived", sample, track, reason=reason)

    def queue_payload(self, track, sample, job_id):
        del self.active_tracks[track.track_id]
        self.payload_tracks[track.track_id] = track
        self._geometry_changed()
        track.status = "payload_queued"
        self._emit("payload_job_submitted", sample, track, job_id=job_id,
                   packet_start=track.header.start, header=track.header.info.copy())

    def retire(self, track: PacketTrack, status: str, sample: int, reason: str, **fields):
        if track.track_id in self.active_tracks:
            del self.active_tracks[track.track_id]
        elif track.track_id in self.payload_tracks:
            del self.payload_tracks[track.track_id]
        else:
            return
        track.status = status
        track.reason = reason
        track.finished_sample = track.last_processed_sample = sample
        self._recent.append(track)
        self._geometry_changed()
        if status == "completed":
            self.completed_tracks.append(track)
            self.counts["packets_completed"] += 1
            self.counts["phy_crc_pass" if fields.get("crc_ok") else "phy_crc_fail"] += 1
        elif status == "expired":
            self.expired_tracks.append(track)
            self.counts["tracks_expired"] += 1
        else:
            self.rejected_tracks.append(track)
            self.counts["tracks_rejected"] += 1
        self._emit("packet_completed" if status == "completed" else "track_" + status,
                   sample, track, reason=reason, expected_end_sample=track.expected_end_sample,
                   duplicate_of=track.duplicate_of, **fields)

    def stats(self) -> dict:
        names = ("candidate_observations", "candidate_duplicates", "tracks_created",
                 "valid_header_observations", "headers_accepted", "header_duplicates",
                 "packets_completed", "phy_crc_pass", "phy_crc_fail", "tracks_expired",
                 "tracks_rejected", "capacity_rejections", "max_active_tracks", "max_valid_tracks")
        names += ("physical_headers_validated", "repeated_headers_associated")
        return {**{name: self.counts[name] for name in names},
                "active_tracks": len(self.active_tracks),
                "valid_active_tracks": sum(t.header is not None for t in self.active_tracks.values())}
