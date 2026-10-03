import importlib.util
import json
from pathlib import Path

import pytest

from bench import hardware

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_jev_harness.py"
spec = importlib.util.spec_from_file_location("run_jev_harness", SCRIPT)
rjh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rjh)


def pred(i, dom, typ, correct, p, gold, metric="exact"):
    return {"id": i, "domain": dom, "type": typ, "correct": correct, "pred": p, "gold": gold,
            "eval_metric": metric}


def make_run(wall, answers_ok, ms):
    preds = [pred("a", "bank", "choice", answers_ok[0], "x", "x"),
             pred("b", "bank", "choice", answers_ok[1], "y", "z"),
             pred("c", "ops", "noul", True, "yes", "yes"),
             pred("d", "ops", "score", False, "2", "1", "ordinal"),
             pred("e", "ops", "score", True, "1", "1", "ordinal")]
    return {"preds": preds, "timings": {i: ms for i in "abcde"}, "wall": wall, "gpu_at_start": None}


def test_aggregate_groups_accuracy_latency_and_totals():
    runs = [make_run(10.0, (True, False), 100.0), make_run(14.0, (True, False), 200.0)]
    res = rjh.aggregate(runs)
    assert res["cases"] == 5 and res["runs"] == 2 and res["wall_s"] == [10.0, 14.0]
    assert res["by_type"]["choice"]["accuracy_pct"] == 50.0
    assert res["by_type"]["choice"]["avg_latency_ms"] == 150.0           # mean of runs (100, 200)
    assert res["by_type"]["choice"]["total_s"] == pytest.approx(0.3)     # 2 cases x 150 ms
    assert set(res["by_dataset"]) == {"bank|choice", "ops|noul", "ops|score"}
    assert res["by_dataset"]["ops|score"]["cases"] == 2
    assert res["deterministic"] is True and res["unanswered"] == [0, 0]
    assert res["ordinal"]["cases"] == 2 and res["ordinal"]["mae"] == 0.5
    assert res["overall"]["accuracy_pct"] == pytest.approx(60.0)


def test_aggregate_detects_changing_answers_and_unanswered():
    r1, r2 = make_run(1.0, (True, False), 10.0), make_run(1.0, (True, False), 10.0)
    r2["preds"][0]["pred"] = None
    res = rjh.aggregate([r1, r2])
    assert res["deterministic"] is False and res["unanswered"] == [0, 1]


def test_markdown_has_all_sections():
    res = {"m1": rjh.aggregate([make_run(10.0, (True, False), 100.0)])}
    md = rjh.to_markdown(res)
    for heading in ("## By question type", "## Per dataset", "## Run times and conditions", "## Score tasks"):
        assert heading in md
    assert "| bank | choice | 2 |" in md


def test_only_local_urls_are_accepted():
    assert rjh.check_local_url("http://127.0.0.1:8000/v1/systemone")
    assert rjh.check_local_url("http://localhost:9/v1/systemone")
    for bad in ("https://api.typesafe.ai/v1/systemone", "http://example.com:8000/x", "http://10.0.0.5:8000/x"):
        with pytest.raises(SystemExit):
            rjh.check_local_url(bad)


def test_parse_pairs():
    assert rjh.parse_pairs(["a=http://x:1/y?z=2", "b=c"], "--x") == {"a": "http://x:1/y?z=2", "b": "c"}
    with pytest.raises(SystemExit):
        rjh.parse_pairs(["nope"], "--x")


def test_ac_state(tmp_path, monkeypatch):
    monkeypatch.setattr(hardware, "POWER_SUPPLY_DIR", tmp_path)
    assert hardware.ac_state() is None                       # no AC supply entry: unknown
    (tmp_path / "AC0").mkdir()
    (tmp_path / "AC0" / "online").write_text("0\n")
    assert hardware.ac_state() is False
    (tmp_path / "AC0" / "online").write_text("1\n")
    assert hardware.ac_state() is True


def test_presets_use_exact_laya_checkpoint_names_and_local_binding():
    cmd, env, req = rjh.PRESETS["laya-typed"](8123, "cuda")
    assert req == "typed-decisions" and env["LAYA_MODELS"] == "typed-decisions"
    assert env["LAYA_HOST"] == "127.0.0.1" and env["LAYA_AUTO_TASK"] == "0" and env["LAYA_PORT"] == "8123"
    cmd, env, req = rjh.PRESETS["laya-english"](1, "cpu")
    assert req == "english"
    cmd, env, req = rjh.PRESETS["strands-decision"](8124, "cuda")
    assert "127.0.0.1" in cmd and "8124" in cmd and req == "strands-decider"
