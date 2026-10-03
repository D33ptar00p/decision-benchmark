import gzip
import importlib.util
import json
import time
from pathlib import Path

import pytest

from bench import config
from bench.summary import dataset_stats, overall_stats

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_benchmark.py"
spec = importlib.util.spec_from_file_location("run_benchmark", SCRIPT)
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)


def synthetic_run(ms_scale=1.0):
    results = [{"id": str(i), "ok": i % 2 == 0, "ms": 10.0 * (i + 1) * ms_scale, "err": None} for i in range(4)]
    total = sum(r["ms"] for r in results)
    summary = {"cases": 4, "correct": 2, "accuracy": 0.5, "avg_latency_ms": total / 4, "total_latency_ms": total}
    return {"d": {"results": results, "summary": summary}}


def test_dataset_and_overall_stats():
    runs = [synthetic_run(1.0), synthetic_run(2.0)]
    s = dataset_stats(runs, "d")
    assert s["accuracy_pct"] == 50.0 and s["cases"] == 4
    assert s["avg_latency_ms"] == pytest.approx((25.0 + 50.0) / 2)      # run means 25 and 50
    assert s["total_s"] == pytest.approx((0.1 + 0.2) / 2)               # 100 ms and 200 ms
    assert s["deterministic"] is True and s["errors"] == 0
    o = overall_stats(runs, ["d"])
    assert o["cases"] == 4 and o["accuracy_pct"] == 50.0 and o["total_s_per_run"] == pytest.approx([0.1, 0.2])
    single = dataset_stats(runs[:1], "d")                               # one run: no spread, no crash
    assert single["avg_latency_sd"] == 0.0


class FakeModel:
    name = "fake"

    def warmup(self):
        pass

    def decide(self, case):
        time.sleep(0.002)
        return case.expected if case.id != "2" else next(o for o in case.options if o != case.expected)


def test_run_models_writes_raw_and_summary(tmp_path, monkeypatch):
    ds = tmp_path / "data"
    ds.mkdir()
    rows = [{"id": str(i), "query": "q", "options": ["a", "b"], "expected": "a"} for i in range(4)]
    (ds / "toy.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(config, "DATASET_DIR", ds)
    monkeypatch.setitem(config.FACTORIES, "fake", FakeModel)
    out = tmp_path / "out"
    summary = rb.run_models(["fake"], rb.resolve_datasets("all"), cases=3, runs=2, out_dir=out)
    m = summary["models"]["fake"]
    assert m["datasets"]["toy"]["cases"] == 3                            # first 3 cases only
    assert m["datasets"]["toy"]["accuracy_pct"] == pytest.approx(200 / 3)  # case "2" is wrong
    assert m["overall"]["cases"] == 3 and len(m["conditions_at_run_start"]) == 2
    raw = json.load(gzip.open(out / "fake.json.gz", "rt"))
    assert len(raw) == 2 and len(raw[0]["toy"]["results"]) == 3
    md = rb.to_markdown(summary)
    assert "## fake" in md and "| toy | 3 |" in md and "Same answers in every run: True" in md


def test_resolve_datasets(tmp_path, monkeypatch):
    for n in ("a", "b", "sample"):
        (tmp_path / f"{n}.jsonl").write_text("")
    monkeypatch.setattr(config, "DATASET_DIR", tmp_path)
    assert rb.resolve_datasets("all") == ["a", "b"]                      # 'sample' is a smoke test
    assert rb.resolve_datasets("b, a") == ["b", "a"]
    with pytest.raises(SystemExit):
        rb.resolve_datasets("nope")
