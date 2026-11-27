"""tools/run_release_tests.py isolated mode: a child passes only with exit 0 and one valid result (Batch 10.4h review)."""
from __future__ import annotations
import json
import types
import unittest
from unittest.mock import patch


def marker(tests=1, failures=0, errors=0, skipped=0):
    return "ok\n@@BATCH06_TEST_RESULT@@" + json.dumps(dict(tests=tests, failures=failures, errors=errors, skipped=skipped)) + "\n"


class IsolatedRunnerTests(unittest.TestCase):
    def _run(self, children):
        import tools.run_release_tests as R
        ids = [f"tests.fake.T.test_{i}" for i in range(len(children))]
        calls = iter(children)
        def fake_run(*args, **kwargs):
            code, out = next(calls)
            return types.SimpleNamespace(returncode=code, stdout=out, stderr="")
        with patch.object(R, "suite_for", return_value=[types.SimpleNamespace(id=(lambda i=i: i)) for i in ids]), \
             patch.object(R, "flatten", side_effect=lambda suite: iter(suite)), \
             patch.object(R.subprocess, "run", side_effect=fake_run), patch("builtins.print"):
            return R.run_isolated()

    def test_a_clean_marker_with_a_nonzero_exit_fails(self):
        """The reproduced false pass: the child reports zero failures but exits 1."""
        self.assertFalse(self._run([(0, marker()), (1, marker())]))

    def test_a_missing_marker_fails_even_with_exit_zero(self):
        self.assertFalse(self._run([(0, "no marker here\n")]))

    def test_an_empty_run_fails(self):
        self.assertFalse(self._run([(0, marker(tests=0))]))

    def test_reported_failures_fail(self):
        self.assertFalse(self._run([(1, marker(failures=1))]))

    def test_clean_children_pass(self):
        self.assertTrue(self._run([(0, marker()), (0, marker(skipped=1))]))

    def test_totals_record_why(self):
        import tools.run_release_tests as R
        totals = dict(tests=0, failures=0, errors=0, skipped=0, timed_out=0, child_failures=0)
        with patch("builtins.print"):
            self.assertFalse(R.record_child(totals, "t", 1, marker()))
            self.assertTrue(R.record_child(totals, "u", 0, marker()))
        self.assertEqual((totals["tests"], totals["child_failures"]), (2, 1))
        self.assertFalse(R.isolated_passed(totals))


if __name__ == "__main__":
    unittest.main()
