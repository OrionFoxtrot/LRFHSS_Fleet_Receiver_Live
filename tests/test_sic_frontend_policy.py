"""Exercise the actual front-end option builders without opening a radio."""
import sys
import unittest
from unittest.mock import patch

import lrfhss_live as phy
from lrfhss_gui_app.config import ReceiverConfig, decoder_options


class FrontendSicPolicyTests(unittest.TestCase):
    def test_gui_live_configuration_selects_qualified_policy(self):
        options=decoder_options(ReceiverConfig(mode='live'))
        self.assertEqual(options['sic_pressure'], {'mode':'lateness'})
        decoder=phy.LiveDecoder(**options)
        try:
            self.assertTrue(decoder.sic.lateness_pressure)
            self.assertEqual((options['header_workers'],options['payload_workers'],options['sic_workers']), (7,4,1))
            self.assertEqual(options['acquisition_options']['carrier_search_hz'],4500)
        finally:
            decoder.close()

    def test_cli_launcher_selects_qualified_policy(self):
        with patch.object(sys,'argv',['lrfhss_live.py','--live','--throughput']), patch.object(phy,'run_live') as run:
            phy.main()
        options=run.call_args.kwargs
        self.assertEqual(options['sic_pressure'],{'mode':'lateness'})
        self.assertEqual(options['phdr_sic'],'on')
        self.assertEqual(options['sic_eof'],'skip')
        self.assertEqual((options['header_workers'],options['payload_workers'],options['sic_workers']), (7,4,1))
