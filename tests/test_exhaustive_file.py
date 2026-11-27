"""Opt-in file backpressure, watchdog and unchanged default/live routing."""
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import Mock, patch

from lrfhss_exhaustive_file import ExhaustiveFile
import lrfhss_live as phy
from lrfhss_gui_app.config import ReceiverConfig, decoder_options, HOP_RR2


def decoder():
    return NS(abs_sample=0,input_abs_sample=0,ring=NS(keep=16_000_000,end=0),
              header_pipeline=None,acquisition=NS(pending=[]),_refinement_pending=deque(),
              acquisition_process=None,track_manager=NS(active_tracks={}),_payload_pending={},
              payload_pool=NS(can_admit=lambda:True),runtime_counts=Counter(),sic=None,
              payload_finish_timeout=120.)


class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def sleep(self,seconds):self.now+=seconds


class ExhaustiveBackpressureTests(unittest.TestCase):
    def controller(self,d,**kwargs):
        c=Clock()
        return ExhaustiveFile(d,clock=c,sleep=c.sleep,**kwargs),c

    def test_waits_for_worker_progress_without_advancing_rf(self):
        d=decoder();d._payload_pending['job']=object()
        mode,clock=self.controller(d)
        calls=[]
        def service():
            calls.append(d.abs_sample)
            if len(calls)==4:d._payload_pending.clear()
        self.assertTrue(mode.before_feed(service))
        self.assertEqual(calls,[0,0,0,0])
        self.assertEqual(mode.wait_count,1)
        self.assertGreater(mode.wait_wall_s,0)
        self.assertEqual(mode.watchdog_trips,0)

    def test_acquisition_must_commit_before_another_block(self):
        d=decoder();d.acquisition_process=NS(outstanding=1,capacity=20)
        mode,clock=self.controller(d)
        calls=[]
        def service():
            calls.append(1)
            if len(calls)==2:d.acquisition_process.outstanding=0
        mode.before_feed(service)
        self.assertEqual(len(calls),2)
        self.assertEqual(d.input_abs_sample,0)

    def test_future_iq_is_not_mistaken_for_host_backlog(self):
        d=decoder();d.abs_sample=d.input_abs_sample=d.ring.end=100_000
        d.acquisition.pending=[NS(candidate_id=1,header_start=0,observed_sample=100_000)]
        mode,clock=self.controller(d)
        self.assertEqual(mode.state()['reasons'],[])
        self.assertTrue(mode.before_feed(lambda:None))
        self.assertEqual(mode.wait_count,0)

    def test_age_and_iq_retention_have_two_second_margin(self):
        d=decoder();d.abs_sample=d.input_abs_sample=d.ring.end=14_000_000
        d.header_pipeline=NS(max_age_s=12.,released=[],parked={},pending={},pool=NS(available=1))
        d.acquisition.pending=[NS(candidate_id=1,header_start=0,observed_sample=0)]
        mode,clock=self.controller(d)
        state=mode.state()
        self.assertIn('candidate_age_guard',state['reasons'])
        self.assertIn('iq_retention_guard',state['reasons'])
        self.assertEqual(mode.margin_s,2.)

    def test_watchdog_logs_counts_and_leaves_wait(self):
        d=decoder();d._payload_pending['held']=object()
        warnings=[];mode,clock=self.controller(d,watchdog_s=.003,warn=warnings.append)
        self.assertTrue(mode.before_feed(lambda:None))
        self.assertEqual(mode.watchdog_trips,1)
        self.assertEqual(len(warnings),1)
        self.assertIn('continuing',warnings[0])
        self.assertLess(clock.now,.01)

    def test_wedged_full_acquisition_is_reported_without_blocking_feed(self):
        d=decoder();d.acquisition_process=NS(outstanding=20,capacity=20)
        mode,clock=self.controller(d,watchdog_s=.003,warn=lambda _:None)
        with self.assertRaisesRegex(RuntimeError,'cannot admit'):
            mode.before_feed(lambda:None)
        self.assertEqual(mode.watchdog_trips,1)

    def test_stop_interrupts_wait(self):
        mode,clock=self.controller(decoder())
        self.assertFalse(mode.before_feed(lambda:self.fail('must not service'),stop_event=NS(is_set=lambda:True)))

    def test_eof_allows_slow_progress_and_restores_normal_timeout(self):
        d=decoder();mode,clock=self.controller(d,watchdog_s=.1)
        def finish(*,stop_event):
            self.assertEqual(d.payload_finish_timeout,float('inf'))
            for _ in range(5):
                d.runtime_counts['payload_jobs_completed']+=1
                clock.now+=.2
                self.assertFalse(stop_event.is_set())
            return ['packet']
        d.finish=finish
        self.assertEqual(mode.finish(),['packet'])
        self.assertEqual(d.payload_finish_timeout,120.)
        self.assertEqual(mode.watchdog_trips,0)

    def test_eta_and_wait_counters_are_additive_diagnostics(self):
        mode,clock=self.controller(decoder());clock.now=30
        self.assertEqual(mode.snapshot(.25)['eta_s'],90.)
        self.assertIsNone(mode.snapshot(0)['eta_s'])
        self.assertEqual(mode.snapshot()['watchdog_trips'],0)

    def test_primary_payload_is_deferred_only_in_exhaustive_mode(self):
        d=phy.LiveDecoder.__new__(phy.LiveDecoder)
        d.payload_pool=NS(can_admit=lambda:False);d.runtime_counts=Counter()
        d.track_manager=NS(retire=Mock());d.abs_sample=10
        d._job_id=lambda _: 'job'
        track=NS(header=object())
        d._exhaustive_file=True
        d._submit_payload(track)
        d.track_manager.retire.assert_not_called()
        self.assertTrue(track.awaiting_primary_admission)
        self.assertEqual(d.runtime_counts['payload_jobs_rejected'],0)
        d._exhaustive_file=False
        d._submit_payload(track)
        self.assertEqual(d.runtime_counts['payload_jobs_rejected'],1)
        d.track_manager.retire.assert_called_once()

    def test_complete_deferred_payload_survives_exhaustive_eof(self):
        d=phy.LiveDecoder.__new__(phy.LiveDecoder)
        track=NS(header=object(),expected_end_sample=5,awaiting_primary_admission=True)
        d.sic=None;d.abs_sample=10;d.track_manager=NS(active_tracks={1:track},retire=Mock())
        d._exhaustive_file=True
        d._retire_active_at_eof();d.track_manager.retire.assert_not_called()
        d._exhaustive_file=False
        d._retire_active_at_eof();d.track_manager.retire.assert_called_once()

    def test_idle_service_requires_explicit_file_opt_in(self):
        d=phy.LiveDecoder.__new__(phy.LiveDecoder)
        # Use the undecorated owner body; publication machinery is not needed to
        # check the capability gate.
        with self.assertRaisesRegex(RuntimeError,'not enabled'):
            phy.LiveDecoder.service_file_backlog.__wrapped__(d)


class ExhaustiveRoutingTests(unittest.TestCase):
    def test_default_off_and_live_decoder_options_unchanged(self):
        self.assertFalse(ReceiverConfig(mode='file').exhaustive_file)
        for mode in ('live','file'):
            a=ReceiverConfig(mode=mode,hop_id_mode=HOP_RR2)
            b=ReceiverConfig(mode=mode,hop_id_mode=HOP_RR2,exhaustive_file=True)
            self.assertEqual(decoder_options(a),decoder_options(b))

    def test_cli_file_flag_reaches_rr2_file_adapter_only(self):
        with patch.object(sys,'argv',['lrfhss_live.py','--file','recording.iq','--nuc-rr2-20s-v2','--exhaustive-file']),patch.object(phy,'run_file',return_value=0) as run:
            self.assertEqual(phy.main(),0)
        self.assertTrue(run.call_args.kwargs['exhaustive_file'])
        self.assertEqual(run.call_args.kwargs['allowed_hop_ids'],(256,299))

    def test_cli_default_file_call_is_unchanged(self):
        with patch.object(sys,'argv',['lrfhss_live.py','--file','recording.iq']),patch.object(phy,'run_file',return_value=0) as run:
            phy.main()
        self.assertNotIn('exhaustive_file',run.call_args.kwargs)

    def test_cli_flag_has_no_effect_on_live_call(self):
        calls=[]
        for flag in ([],['--exhaustive-file']):
            with patch.object(sys,'argv',['lrfhss_live.py','--live','--nuc-rr2-20s-v2',*flag]),patch.object(phy,'run_live',return_value=0) as run:
                phy.main();calls.append(run.call_args)
        self.assertEqual(calls[0],calls[1])


if __name__=='__main__':unittest.main()
