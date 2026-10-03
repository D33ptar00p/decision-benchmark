"""Runs JevBench's own bench_eval.py, unmodified, with its `jev` backend pointed at a LOCAL server.

The harness hard-codes TypeSafe's hosted API and demands an API key. This wrapper replaces the
URL with a localhost one (and refuses anything else), so no data leaves this machine, and it
times every request (end to end: HTTP round trip + model).

Environment:
  JEV_DIR         path to a clone of github.com/model-collapse/jev-bench
  JEV_LOCAL_URL   http://127.0.0.1:<port>/v1/systemone (or localhost)
  JEV_TIMING_OUT  optional path; per-request timings {row_id: milliseconds} are written there
Other arguments are passed straight to bench_eval.py (--backend jev --model ... --data ...).
"""
import json
import os
import sys
import time

JEV_DIR = os.environ["JEV_DIR"]
URL = os.environ["JEV_LOCAL_URL"]
if not URL.startswith(("http://127.0.0.1:", "http://localhost:")):
    sys.exit(f"refusing non-local URL: {URL}")

sys.path.insert(0, JEV_DIR)
import bench_eval  # noqa: E402  (the harness, unmodified)

bench_eval.Jev.URL = URL
os.environ["TYPESAFE_API_KEY"] = "local-unused"  # the harness requires one; local servers ignore it

timings: dict[str, float] = {}
_orig_predict = bench_eval.Jev.predict


def _timed_predict(self, row):
    start = time.perf_counter()
    try:
        return _orig_predict(self, row)
    finally:
        timings[row["id"]] = (time.perf_counter() - start) * 1000


bench_eval.Jev.predict = _timed_predict

t0 = time.time()
bench_eval.main()
print(f"WALL_SECONDS {time.time() - t0:.1f}")
out = os.environ.get("JEV_TIMING_OUT")
if out:
    with open(out, "w") as f:
        json.dump(timings, f)
    print(f"wrote {len(timings)} per-request timings to {out}")
