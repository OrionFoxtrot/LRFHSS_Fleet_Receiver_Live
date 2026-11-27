"""Non-Qt state tests: counters are observed sequence gaps, not timed losses."""
import unittest
from lrfhss_gui_app.session import FleetSession


class FleetDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.session = FleetSession()

    def add(self, counter, t=0, csid=5, phy_ok=True):
        envelope = {"received_epoch": 1000+t, "source_time_s": t, "source_id": "test",
            "phy": {"crc_ok": phy_ok, "fragment_quality": {"strongly_degraded": 1}},
            "record": {"packet_type": "DeSCENT_V2_5", "decoded": {
                "csid": csid, "packet_counter": counter}}}
        return self.session.add_packet(envelope, ignore_csid_zero=False)

    def test_wrap_gaps_and_late_fill(self):
        for counter in (65534, 1, 0, 65535, 1):
            self.add(counter)
        state = self.session.fleet[5]
        self.assertEqual(state.count, 5)
        self.assertEqual(state.estimated_counter_gaps, 0)
        self.assertEqual(state.duplicate_counters, 1)
        self.assertEqual(state.last_packet_counter, 1)

    def test_long_silence_is_not_a_packet_loss_estimate(self):
        self.add(64, 0)
        self.add(65, 10000)
        state = self.session.fleet[5]
        self.assertEqual(state.estimated_counter_gaps, 0)
        self.assertAlmostEqual(state.recent_packet_rate_hz(), 1/60)
        self.assertEqual(state.recent_time_basis, "test")
        self.assertEqual(state.latest_phy_quality, {"strongly_degraded": 1})

    def test_backward_counter_does_not_invent_a_large_gap(self):
        for c in (100, 102, 5, 6, 103):
            self.add(c)
        state = self.session.fleet[5]
        self.assertEqual(state.estimated_counter_gaps, 1)
        self.assertEqual(state.counter_discontinuities, 2)

    def test_phy_fail_cannot_commit_application_telemetry(self):
        result = self.add(2, phy_ok=False)
        self.assertFalse(result.fleet_updated)
        self.assertEqual(self.session.fleet, {})
        self.assertEqual(self.session.app_crc_pass, 0)

    def test_stage_snapshot_is_distinct_from_session_counts(self):
        self.add(64)
        self.session.update_receiver_stats({"tracks_created": 2, "phy_crc_fail": 1})
        diag = self.session.diagnostics()
        self.assertEqual(diag["source_run"]["tracks_created"], 2)
        self.assertEqual(diag["session"]["application_crc_pass"], 1)
        self.session.clear()
        self.assertEqual(self.session.receiver_stages, {})
        self.assertEqual(self.session.fleet, {})


if __name__ == "__main__":
    unittest.main()
