"""Admission, CRC and publication contracts of the live spare-worker retry mode."""
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from descent_payload import crc16_ccitt_false
from lrfhss_adaptive_retry import AdaptivePayloadPool
from lrfhss_gui_app.config import ReceiverConfig, decoder_options
from lrfhss_live import Header, Packet
from lrfhss_payload_jobs import PayloadJob, PayloadPool


class AdaptiveRetryTests(unittest.TestCase):
    def setUp(self):
        self.pool = AdaptivePayloadPool(4, 32)
        process = Mock()
        process.is_alive.return_value = True
        self.pool.slots = [dict(job=None, process=process) for _ in range(4)]
        self.addCleanup(self.pool.slots.clear)
        self.job = PayloadJob("primary", "rx", 1, 1000, 900, 100, 1500., 2., (), b"iq", "u8",
                              (), (), 0., payload_retry=((0, 50.), (-512, 0.)))

    def poll(self, kind, job_id, result):
        with patch.object(PayloadPool, "poll", return_value=[(kind, job_id, result)]):
            return self.pool.poll()

    def first_failure(self, job):
        with patch.object(PayloadPool, "submit", return_value=True) as submit:
            self.assertTrue(self.pool.submit(job))
            self.assertEqual(submit.call_args.args[0].payload_retry, ())
        result = dict(packet=None, detail={}, wall_s=1., cpu_s=.5)
        return result, self.poll("completed", job.job_id, result)

    def dispatch(self):
        with patch.object(PayloadPool, "try_submit_idle", return_value=True) as admit:
            self.pool.service_retries()
            return admit.call_args.args[0] if admit.called else None

    def test_live_default_and_explicit_modes(self):
        for mode, expected in (("live", "adaptive"), ("file", "grid")):
            self.assertEqual(decoder_options(ReceiverConfig(mode))["payload_retry"], expected)
        self.assertEqual(decoder_options(ReceiverConfig("live", payload_retry="off"))["payload_retry"], "off")

    def test_primary_queue_spare_slot_and_single_flight(self):
        _, messages = self.first_failure(self.job)
        self.assertEqual(messages, [("deferred", "primary", {})])
        self.pool.waiting.append(self.job)
        self.assertIsNone(self.dispatch())
        self.pool.waiting.clear()
        for slot in self.pool.slots[:3]:
            slot["job"] = "busy"
        self.assertIsNone(self.dispatch())
        self.pool.slots[1]["job"] = None
        trial = self.dispatch()
        self.assertEqual((trial.header_start, trial.frequency_hz), (100, 1550.))
        self.assertIsNone(self.dispatch())
        self.assertEqual(self.poll("started", trial.job_id, {}), [])

    def test_both_crcs_original_coordinates_and_learned_offset(self):
        self.first_failure(self.job)
        first = self.dispatch()
        detail = {}
        bad = Packet(bytes(55), True, Header(100, 1550., 2., {}), 800, 15, 3., (), detail)
        self.assertEqual(self.poll("completed", first.job_id, dict(packet=bad, detail=detail)), [])
        second = self.dispatch()
        payload = bytes(53) + crc16_ccitt_false(bytes(53)).to_bytes(2, "little")
        detail = {}
        good = Packet(payload, True, Header(-412, 1500., 2., {}), 288, 15, 3., (), detail)
        (kind, job_id, result), = self.poll("completed", second.job_id, dict(packet=good, detail=detail))
        self.assertEqual((kind, job_id, good.header.start, good.payload_start), ("completed", "primary", 100, 800))
        self.assertEqual(result["detail"]["payload_retry"]["rejected_app_crc"], 1)
        self.assertEqual(good.payload_diagnostics["payload_retry"]["accepted"], [-512, 0.])
        self.assertEqual(self.pool.deferred_bytes, 0)
        self.first_failure(replace(self.job, job_id="next"))
        self.assertEqual(self.dispatch().header_start, second.header_start)

    def test_capacity_and_worker_failure_preserve_original_result(self):
        self.pool.MAX_DEFERRED = 1
        original, _ = self.first_failure(self.job)
        skipped, messages = self.first_failure(replace(self.job, job_id="full"))
        self.assertEqual(messages, [("completed", "full", skipped)])
        trial = self.dispatch()
        self.assertEqual(self.poll("failed", trial.job_id, {}), [("completed", "primary", original)])
        self.assertIsNone(original["detail"]["payload_retry"]["accepted"])
        self.assertEqual((self.pool.counts["trial_errors"], self.pool.counts["capacity_skips"]), (1, 1))

    def test_exhausted_grid_returns_the_first_failure_once(self):
        original, _ = self.first_failure(replace(self.job, payload_retry=((0, 50.),)))
        trial = self.dispatch()
        messages = self.poll("completed", trial.job_id, dict(packet=None, detail={}))
        self.assertEqual(messages, [("completed", "primary", original)])
        self.assertEqual(self.pool.counts["exhausted"], 1)
        self.assertIsNone(self.dispatch())

    def test_round_robin_and_shutdown(self):
        self.first_failure(self.job)
        self.first_failure(replace(self.job, job_id="other"))
        trial = self.dispatch()
        self.assertEqual(self.poll("completed", trial.job_id, dict(packet=None, detail={})), [])
        self.assertTrue(self.dispatch().job_id.startswith("other:adaptive:"))
        self.pool.slots.clear()
        self.pool.close()
        self.assertEqual(self.pool.counts["abandoned"], 2)
        self.assertEqual((len(self.pool.deferred), self.pool.active_retry, self.pool.deferred_bytes), (0, None, 0))
