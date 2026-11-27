"""State transitions at fixed RF coordinates must invalidate the SIC pressure indexes."""
import unittest
from collections import deque
from unittest.mock import patch
from lrfhss_gui_app.config import ReceiverConfig, decoder_options
from lrfhss_live import Header, LiveDecoder


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.d=LiveDecoder(**decoder_options(ReceiverConfig(mode='live')))
        self.addCleanup(self.d.close)
        self.h=self.d.header_pipeline
        self.d.abs_sample=8_000_000
        self.d._header_tokens=0
    def offer(self,start,observed=5_900_000):
        a=self.d.acquisition
        a._offer(start,0,a.nominal_hz[0],'sync',.8,4.,observed)
        return a.pending[-1]
    def ref(self,start=6_000_000):
        c=self.offer(start)
        self.d.acquisition.ready(self.d.abs_sample,1)
        self.h._queue_refinements(c,(100,0),None,{})
        return c
    def test_clock_and_queue_substitution_at_same_sample(self):
        c=self.offer(6_000_000)
        self.assertEqual(self.h.sic_lateness(20.),(0.,0))
        with patch('lrfhss_header_pipeline.time.perf_counter',return_value=18.):self.h.service()
        self.assertEqual(self.h.sic_lateness(20.),(2/12,0))
        self.assertEqual(self.h.sic_lateness(21.),(3/12,0))
        self.d.acquisition.ready(self.d.abs_sample,1)
        self.offer(6_040_000)
        self.assertEqual(self.h.sic_lateness(21.),(0.,0))
        with patch('lrfhss_header_pipeline.time.perf_counter',return_value=20.):self.h.service()
        self.assertEqual(self.h.sic_lateness(21.),(1/12,0))
        self.d.acquisition.finish(self.d.abs_sample)
        self.assertEqual(self.h.sic_lateness(21.),(0.,0))
    def test_refinement_creation_obstruction_and_age(self):
        self.assertEqual(self.h.sic_lateness(20.)[1],0)
        c=self.ref()
        self.assertEqual(self.h.sic_lateness(20.)[1],1)
        self.offer(c.header_start+8192)
        self.assertEqual(self.h.sic_lateness(20.)[1],0)
        self.d.acquisition.ready(self.d.abs_sample,1)
        self.assertEqual(self.h.sic_lateness(20.)[1],1)
        self.d.abs_sample=c.observed_sample+1_500_000
        self.assertEqual(self.h.sic_lateness(20.)[1],0)
        self.d.abs_sample+=1
        self.assertEqual(self.h.sic_lateness(20.)[1],1)
    def test_known_header_reanchor_history_eviction_and_release(self):
        c=self.ref();m=self.d.track_manager;m._recent=deque(maxlen=1)
        self.assertEqual(self.h.sic_lateness(20.)[1],1)
        track=m.offer_candidate(c.header_start-8704,self.d.abs_sample,100.)
        self.assertEqual(self.h.sic_lateness(20.)[1],0)
        hdr=Header(c.header_start+4096,100.,0.,{})
        args=dict(cfo_hz=0.,payload_start=7_000_000,expected_end=11_000_000,hops_hz=(),sample=self.d.abs_sample)
        m.accept_header(track,hdr,**args)
        self.assertEqual(self.h.sic_lateness(20.)[1],0)
        m.reanchor(track,Header(hdr.start+1,100.,0.,{}),reason='test',**args)
        self.assertEqual(self.h.sic_lateness(20.)[1],1)
        m.reanchor(track,hdr,reason='test',**args)
        m.retire(track,'completed',self.d.abs_sample,'payload_decoded',crc_ok=True)
        self.assertEqual(self.h.sic_lateness(20.)[1],0)
        second=m.offer_candidate(c.header_start+100_000,self.d.abs_sample,200.)
        m.retire(second,'expired',self.d.abs_sample,'end_of_stream')
        self.assertEqual(self.h.sic_lateness(20.)[1],1)
        self.h._park(c,track,{})
        self.assertEqual(self.h.sic_lateness(20.)[1],0)
        with patch('lrfhss_header_pipeline.time.perf_counter',return_value=19.):self.h._release(c.candidate_id,'test')
        self.assertEqual(self.h.sic_lateness(20.),(1/12,0))
