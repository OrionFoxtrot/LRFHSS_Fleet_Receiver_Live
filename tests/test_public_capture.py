"""Optional release-asset capture; no private test bundle is required."""
from pathlib import Path
import hashlib
import unittest

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "tests/fixtures/30s_lrfhss_test.iq"
SHA256 = "f09852865ad54b1e110aad6a104a71972158c686c88e39d4ddadc50ab7bdb3a8"
MISSING = ("Optional 30s_lrfhss_test.iq release asset is absent. Download it from "
           "the matching GitHub release and place it in tests/fixtures/; see README.md. "
           "The private test-data bundle is not required.")

class PublicCaptureTests(unittest.TestCase):
    @unittest.skipUnless(CAPTURE.is_file(), MISSING)
    def test_default_capture_has_two_crc_valid_packets(self):
        from lrfhss_live import LiveDecoder
        from lrfhss_gui_app.config import ReceiverConfig, decoder_options
        with CAPTURE.open("rb") as f:
            self.assertEqual(hashlib.file_digest(f,"sha256").hexdigest(),SHA256)
        config=ReceiverConfig("file");decoder=LiveDecoder(trigger_db=config.trigger_db,**decoder_options(config))
        packets=[]
        try:
            with CAPTURE.open("rb") as f:
                while raw:=f.read(100_000):packets.extend(decoder.feed_raw(raw))
            packets.extend(decoder.finish())
        finally:decoder.close()
        self.assertEqual(len(packets),2)
        self.assertTrue(all(p.crc_ok for p in packets))

if __name__=="__main__":unittest.main()
