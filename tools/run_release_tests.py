#!/usr/bin/env python3
"""Spawn-safe Batch 6 unittest runner.

`python -m unittest` installs unittest's command driver as `__main__`.  Tests in
this receiver deliberately spawn worker interpreters, and some Python 3.13 hosts
can interact poorly with that command-driver bootstrap.  This runner keeps the
driver behind a real `if __name__ == '__main__'` guard.

For release evidence, `--isolate-tests` runs every test case in a fresh parent
interpreter.  The test may still spawn the real receiver worker, but NumPy/SciPy
and multiprocessing state from one test cannot contaminate the next.

`--group` splits the suite (Batch 10.4j). "functional" holds every test whose verdict does
not depend on CPU speed or scheduling: run it any time, also while the laptop is in use.
"timing" holds the tests listed in tests/timing_sensitive.py (wall-clock limits, real
multi-process runs whose result depends on which worker finishes first): run them on an
idle host. "all" (default) runs both.
"""

if __package__ in (None, ""):
    import _src_path

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import unittest

# Reference modules live in the shared external dataset. Test imports must not
# leave interpreter-generated caches in that persistent input tree.
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lrfhss_test_data import ensure_test_data_links


def public_export():
    """Only generated public exports bypass private bundle discovery."""
    marker = ROOT / "EXPORT_MANIFEST.json"
    if not marker.is_file():
        return False
    return json.loads(marker.read_text(encoding="utf-8")).get("schema") == "lrfhss-public-export-v1"


def suite_for(name=None):
    loader = unittest.defaultTestLoader
    if name:
        return loader.loadTestsFromName(name)
    return loader.discover(str(ROOT / "tests"), pattern="test_*.py", top_level_dir=str(ROOT))


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


GROUPS = ("all", "functional", "timing")


def select(tests, group="all"):
    """The tests of one group, in discovery order. A test module that fails to import
    (unittest's _FailedTest) is not in the timing registry, so it lands in "functional"
    and is never hidden."""
    tests = list(tests)
    if group == "all":
        return tests
    if group not in GROUPS:
        raise ValueError(f"unknown test group {group!r}")
    if public_export():
        # Exported tests use deterministic synthetic inputs (plus an optional asset).
        return tests if group == "functional" else []
    from tests.timing_sensitive import is_timing_sensitive
    want = group == "timing"
    return [t for t in tests if is_timing_sensitive(t.id()) == want]


def selected_suite(name=None, group="all"):
    if group == "all":
        return suite_for(name)
    tests = list(flatten(suite_for(name)))
    chosen = select(tests, group)
    print(f"test group {group}: {len(chosen)} of {len(tests)} tests", flush=True)
    return unittest.TestSuite(chosen)


def run_here(name=None, quiet=False, marker=False, group="all"):
    result = unittest.TextTestRunner(verbosity=1 if quiet else 2).run(selected_suite(name, group))
    if marker:
        print("@@BATCH06_TEST_RESULT@@" + json.dumps({
            "tests": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
        }), flush=True)
    return result.wasSuccessful()


def run_isolated(name=None, timeout=180, group="all"):
    ids = [t.id() for t in select(flatten(suite_for(name)), group)]
    if group != "all":
        print(f"test group {group}: {len(ids)} tests", flush=True)
    totals = dict(tests=0, failures=0, errors=0, skipped=0, timed_out=0, child_failures=0)
    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    for index, test_id in enumerate(ids, 1):
        print(f"[{index}/{len(ids)}] {test_id}", flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--single", test_id],
                cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            totals["tests"] += 1; totals["timed_out"] += 1
            print((exc.stdout or "") + (exc.stderr or ""), end="")
            print(f"TIMEOUT after {timeout}s: {test_id}", flush=True)
            continue
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        record_child(totals, test_id, proc.returncode, proc.stdout)
    print("@@BATCH06_ISOLATED_SUMMARY@@" + json.dumps(totals), flush=True)
    return isolated_passed(totals)


def record_child(totals, test_id, returncode, stdout):
    """Fold one isolated child into the totals.

    A child passes only with exit code 0 AND exactly one result marker reporting at least
    one test run and no failure or error (Batch 10.4h review: a child that printed a clean
    marker but exited 1 used to count as a pass; a missing marker or an empty run must fail)."""
    markers = [line for line in stdout.splitlines() if line.startswith("@@BATCH06_TEST_RESULT@@")]
    data = None
    if len(markers) == 1:
        try:
            data = json.loads(markers[0].split("@@BATCH06_TEST_RESULT@@", 1)[1])
        except ValueError:
            data = None
    if data is not None:
        for key in ("tests", "failures", "errors", "skipped"):
            totals[key] += int(data.get(key, 0))
    else:
        totals["tests"] += 1
        totals["errors"] += 1
    reasons = []
    if returncode != 0:
        reasons.append(f"exit code {returncode}")
    if data is None:
        reasons.append("no valid result marker" if not markers else "unreadable or repeated result markers")
    elif int(data.get("tests", 0)) < 1:
        reasons.append("no test ran")
    if reasons:
        totals["child_failures"] += 1
        print(f"FAILED: {test_id} ({', '.join(reasons)})", flush=True)
    return not reasons


def isolated_passed(totals):
    return all(totals[k] == 0 for k in ("failures", "errors", "timed_out", "child_failures"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("test", nargs="?", help="optional unittest dotted test/module name")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--isolate-tests", action="store_true")
    ap.add_argument("--single", metavar="TEST_ID", help=argparse.SUPPRESS)
    ap.add_argument("--timeout", type=int, default=180, help="per-test timeout in isolated mode")
    ap.add_argument("--group", choices=GROUPS, default="all",
                    help="functional: results independent of CPU speed (safe while the laptop is used); "
                         "timing: the tests in tests/timing_sensitive.py (idle host); all (default): both")
    args = ap.parse_args()
    # Batch 7.5 keeps large fixtures/evidence in one reusable external bundle.
    # Resolve/create compatibility links before unittest discovery imports tests.
    if public_export():
        print("Public test subset; optional capture tests skip with download instructions.", flush=True)
    else:
        ensure_test_data_links()
    if args.single:
        ok = run_here(args.single, quiet=args.quiet, marker=True)
    elif args.isolate_tests:
        ok = run_isolated(args.test, args.timeout, args.group)
    else:
        ok = run_here(args.test, quiet=args.quiet, group=args.group)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
