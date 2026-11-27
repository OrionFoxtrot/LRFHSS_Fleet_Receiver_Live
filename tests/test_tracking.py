"""Lifecycle and stream-boundary tests; these do not simulate radio reception."""
import unittest
from unittest.mock import patch

import numpy as np

import lrfhss_live as phy
from lrfhss_tracking import TrackManager


def header(start, frequency=143875.0):
    return phy.Header(start, frequency, 10.0, {
        "payload_len": 55, "modulation": 0, "cr": 1, "grid": 1,
        "hop": 1, "bw_code": 6, "hop_seq_id": 256, "sync_index": 2})


class TrackLifecycleTests(unittest.TestCase):
    def accept(self, manager, track, h):
        return manager.accept_header(track, h, sample=h.start+340000,
            cfo_hz=h.freq_hz-phy.TRIGGER_HEADER_HZ, payload_start=h.start+700416,
            expected_end=h.start+2266416, hops_hz=[phy.TRIGGER_HEADER_HZ])

    def test_local_duplicate_and_independent_nearby_candidate(self):
        manager = TrackManager()
        first = manager.offer_candidate(1_000_000, 1_050_000, 144531.25)
        self.assertIsNone(manager.offer_candidate(1_000_500, 1_100_000, 144531.25))
        second = manager.offer_candidate(1_400_000, 1_450_000, 144531.25)
        self.assertIsNotNone(second)
        self.assertNotEqual(first.track_id, second.track_id)
        self.assertEqual(len(manager.active_tracks), 2)

    def test_distinct_frequency_hypothesis_is_not_a_duplicate(self):
        manager = TrackManager()
        self.assertIsNotNone(manager.offer_candidate(1_000_000, 1_050_000, 144000))
        self.assertIsNotNone(manager.offer_candidate(1_000_000, 1_050_000, 145000))

    def test_decoded_header_association_prevents_second_emission(self):
        manager = TrackManager()
        first = manager.offer_candidate(1_000_000, 1_050_000, 144531.25)
        self.assertTrue(self.accept(manager, first, header(1_008_704)))
        manager.retire(first, "completed", 3_300_000, "test", crc_ok=True)
        # Different coarse onset, but timing refinement finds the same header.
        second = manager.offer_candidate(1_004_000, 3_350_000, 144531.25)
        self.assertFalse(self.accept(manager, second, header(1_008_800)))
        self.assertEqual(second.duplicate_of, first.track_id)
        self.assertEqual(manager.stats()["packets_completed"], 1)
        self.assertEqual(manager.stats()["header_duplicates"], 1)

    def test_many_tracks_and_bounded_history(self):
        manager = TrackManager(max_active=100, history_limit=3)
        tracks = []
        for i in range(100):
            track = manager.offer_candidate(i*20000, 2_500_000, 144531.25)
            self.assertTrue(self.accept(manager, track, header(i*20000+8704)))
            tracks.append(track)
        self.assertEqual(manager.stats()["max_valid_tracks"], 100)
        self.assertIsNone(manager.offer_candidate(3_000_000, 3_050_000, 144531.25))
        self.assertEqual(manager.stats()["capacity_rejections"], 1)
        for track in tracks:
            manager.retire(track, "completed", 5_000_000, "test", crc_ok=True)
        self.assertEqual(manager.stats()["packets_completed"], 100)
        self.assertEqual(len(manager.completed_tracks), 3)
        self.assertEqual(len(manager.active_tracks), 0)

    def test_one_rejected_header_does_not_discard_another_track(self):
        decoder = phy.LiveDecoder()
        first = decoder.track_manager.offer_candidate(100000, 150000, 144531.25)
        second = decoder.track_manager.offer_candidate(500000, 550000, 144531.25)
        decoder.ring.add(0, np.zeros(1_000_000, dtype=np.complex64))
        decoder.abs_sample = 1_000_000
        good = header(58704)  # local start relative to the second track's base
        with patch.object(phy, "decode_first_header", side_effect=[None, good]):
            self.assertEqual(decoder._service(), [])
        self.assertEqual(first.status, "rejected")
        self.assertEqual(second.status, "pending_payload")
        self.assertEqual(second.header.start, 508704)
        self.assertEqual(len(decoder.track_manager.active_tracks), 1)

    def test_evicted_candidate_expires_instead_of_lingering(self):
        decoder = phy.LiveDecoder()
        track = decoder.track_manager.offer_candidate(100000, 150000, 144531.25)
        decoder.ring.add(1_000_000, np.zeros(100000, dtype=np.complex64))
        decoder.abs_sample = 1_100_000
        self.assertEqual(decoder._service(), [])
        self.assertEqual(track.status, "expired")
        self.assertEqual(track.reason, "header_iq_unavailable")
        self.assertEqual(decoder.stats()["active_tracks"], 0)


class StreamBoundaryTests(unittest.TestCase):
    def test_arbitrary_byte_chunks_preserve_iq_sample_clock(self):
        decoder = phy.LiveDecoder()
        raw = bytes([127, 128])*123456
        consumed=[]
        original_add=decoder.ring.add
        def capture_add(start,z):
            consumed.append(z.copy())
            original_add(start,z)
        with patch.object(decoder, "_trigger"), patch.object(decoder.ring,"add",side_effect=capture_add):
            for i in range(0, len(raw), 7777):
                decoder.feed_raw(raw[i:i+7777])
            decoder.finish()
        self.assertEqual(decoder.abs_sample, 123456)
        np.testing.assert_array_equal(np.concatenate(consumed), phy.u8_iq_to_complex(raw))
        self.assertFalse(decoder.ring.chunks)  # Batch 6 close releases large IQ.
        self.assertEqual(decoder.finish(), [])
        with self.assertRaises(RuntimeError):
            decoder.feed_raw(b"\0\0")

    def test_eof_does_not_invent_missing_samples(self):
        decoder = phy.LiveDecoder()
        track = decoder.track_manager.offer_candidate(100000, 150000, 144531.25)
        decoder.abs_sample = 200000
        self.assertEqual(decoder.finish(), [])
        self.assertEqual(track.status, "expired")
        self.assertEqual(track.reason, "end_of_stream")
        self.assertEqual(decoder.abs_sample, 200000)

    def test_incomplete_byte_pair_is_reported(self):
        decoder = phy.LiveDecoder()
        decoder.feed_raw(b"\x80")
        with self.assertRaisesRegex(ValueError, "byte pair"):
            decoder.finish()


if __name__ == "__main__":
    unittest.main()
