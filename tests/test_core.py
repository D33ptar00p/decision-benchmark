import time

from fastapi.testclient import TestClient

from bench.runner import run_benchmark
from bench.schema import Case
from bench.scoring import is_correct, match_option

OPTS = ["card_arrival", "lost_card", "refund"]


class Fake:
    def __init__(self, name, answers, delay=0.0):
        self.name, self.answers, self.delay, self.i = name, answers, delay, 0

    def warmup(self): pass

    def decide(self, case):
        time.sleep(self.delay)
        a = self.answers[self.i]
        self.i += 1
        if isinstance(a, Exception):
            raise a
        return a


def test_match_option_exact_and_chatty():
    assert match_option(" Lost_Card. ", OPTS) == "lost_card"
    assert match_option("I choose refund because", OPTS) == "refund"
    assert match_option("banana", OPTS) is None


def test_is_correct():
    assert is_correct("refund", "Refund")
    assert not is_correct(None, "refund")


def test_runner_scores_errors_and_summarises():
    cases = [Case(str(i), "q", OPTS, "refund") for i in range(3)]
    fake = Fake("f", ["refund", "lost_card", RuntimeError("boom")], delay=0.01)
    events = list(run_benchmark([fake], cases))
    results = [e for e in events if e["type"] == "result"]
    assert [r["correct"] for r in results] == [True, False, False]
    assert "boom" in results[2]["error"]
    summary = next(e for e in events if e["type"] == "summary")
    assert summary["correct"] == 1 and summary["cases"] == 3
    assert summary["total_latency_ms"] >= 30
    assert abs(summary["avg_latency_ms"] * 3 - summary["total_latency_ms"]) < 1e-6
    assert events[-1]["type"] == "done"


def test_runner_timeout_counts_incorrect():
    cases = [Case("1", "q", OPTS, "refund")]
    events = list(run_benchmark([Fake("slow", ["refund"], delay=0.3)], cases, timeout_s=0.05))
    res = next(e for e in events if e["type"] == "result")
    assert not res["correct"] and "timeout" in res["error"]


def test_api_datasets_and_bad_dataset():
    from bench.app import app
    c = TestClient(app)
    assert any(d["name"] == "sample" for d in c.get("/api/datasets").json())
    r = c.post("/api/runs", json={"dataset": "../etc/passwd", "models": ["nimble"]})
    assert r.status_code == 404
    r = c.post("/api/runs", json={"dataset": "sample", "models": ["nope"]})
    assert r.status_code == 400


def test_api_rejects_too_many_options(tmp_path, monkeypatch):
    import json
    from bench import config
    from bench.app import app
    monkeypatch.setattr(config, "DATASET_DIR", tmp_path)
    (tmp_path / "wide.jsonl").write_text(json.dumps(
        {"id": "1", "query": "q", "options": [f"o{i}" for i in range(27)], "expected": "o0"}) + "\n")
    c = TestClient(app)
    r = c.post("/api/runs", json={"dataset": "wide", "models": ["nimble"]})
    assert r.status_code == 400 and "at most 26" in r.json()["detail"]
    assert c.post("/api/runs", json={"dataset": "wide", "models": ["strands-decision"]}).status_code == 200


def test_loader_reads_instructions_descriptions_and_structured_state(tmp_path):
    import json
    from bench.adapters.questions import question_for
    from bench.schema import load_dataset
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({
        "id": "1", "query": {"msg": "hi"}, "options": ["a", "b"], "expected": "a",
        "instructions": "Pick one.", "descriptions": {"a": "first"}, "category": "x"}) + "\n")
    case = load_dataset(p)[0]
    assert case.query == {"msg": "hi"} and case.category == "x"       # structured state kept as an object
    q = question_for(case)
    assert q["instructions"] == "Pick one." and q["criteria"] == {"a": "first", "b": ""}


def test_run_cases_have_text_queries_for_structured_state(tmp_path, monkeypatch):
    import json
    from bench import config
    from bench.app import app
    monkeypatch.setattr(config, "DATASET_DIR", tmp_path)
    (tmp_path / "s.jsonl").write_text(json.dumps(
        {"id": "1", "query": {"a": "x" * 500}, "options": ["a", "b"], "expected": "a"}) + "\n")
    body = TestClient(app).post("/api/runs", json={"dataset": "s", "models": ["nimble"]}).json()
    assert isinstance(body["cases"][0]["query"], str) and len(body["cases"][0]["query"]) <= 201


def test_models_use_exact_ollama_tags(monkeypatch):
    """tev1:0.8b must not look installed just because tev1:latest is."""
    import types
    from bench import app as app_module

    fake = types.SimpleNamespace(model="tev1:latest")
    monkeypatch.setattr(app_module.ollama, "Client",
                        lambda host: types.SimpleNamespace(list=lambda: types.SimpleNamespace(models=[fake])))
    by_name = {m["name"]: m for m in TestClient(app_module.app).get("/api/models").json()}
    assert by_name["tev"]["available"] is True
    assert by_name["tev-0.8b"]["available"] is False
    assert "tev1:0.8b" in by_name["tev-0.8b"]["hint"]
    assert not by_name["nimble"]["available"]
    assert by_name["strands-decision"]["label"] == "Strands Decider 2B"
    assert by_name["tev-0.8b"]["label"] == "TEV1 0.8B" and by_name["laya"]["label"] == "Laya 421M"


def test_laya_adapter_builds_choice_question(monkeypatch):
    import sys
    import types
    seen = {}

    class FakeAgent:
        def predict(self, state, questions):
            seen["state"], seen["questions"] = state, questions
            return {"answers": {"q": {"choice": "b"}}}

    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(load=lambda ckpt: FakeAgent()))
    from bench.adapters.laya import LayaAdapter
    case = Case("1", {"msg": "hi"}, ["a", "b"], "b", "Pick.", {"a": "first"})
    adapter = LayaAdapter("laya", "x/laya")
    assert adapter.decide(case) == "b"
    assert seen["state"] == {"msg": "hi"}
    q = seen["questions"]["q"]
    assert q["type"] == "choice" and q["instructions"] == "Pick."
    assert q["criteria"] == {"a": "first", "b": ""}


def test_all_models_registered_with_option_caps():
    from bench import config
    assert {"nimble", "tev", "tev-0.8b", "laya", "strands-decision"} <= set(config.FACTORIES)
    assert config.MAX_OPTIONS["tev-0.8b"] == 26


def test_run_model_on_datasets_loads_once_and_summarises_per_dataset():
    from bench.runner import run_model_on_datasets

    class Counting(Fake):
        warmups = 0
        closed = 0

        def warmup(self): Counting.warmups += 1

        def close(self): Counting.closed += 1

    a = [Case("a1", "q", OPTS, "refund"), Case("a2", "q", OPTS, "refund")]
    b = [Case("b1", "q", OPTS, "lost_card")]
    fake = Counting("m", ["refund", "lost_card", "lost_card"])
    events = list(run_model_on_datasets(fake, {"A": a, "B": b}))
    assert Counting.warmups == 1 and Counting.closed == 1
    assert [e["dataset"] for e in events if e["type"] == "result"] == ["A", "A", "B"]
    sums = {e["dataset"]: e for e in events if e["type"] == "summary"}
    assert sums["A"]["correct"] == 1 and sums["A"]["cases"] == 2
    assert sums["B"]["accuracy"] == 1.0
    assert [e["dataset"] for e in events if e["type"] == "dataset_start"] == ["A", "B"]
    assert events[-1]["type"] == "done" and len(events[-1]["summaries"]) == 2


def test_dataset_run_api_validates_and_returns_cases(tmp_path, monkeypatch):
    import json
    from bench import config
    from bench.app import app
    monkeypatch.setattr(config, "DATASET_DIR", tmp_path)
    row = {"id": "1", "query": {"k": "v"}, "options": ["a", "b"], "expected": "a"}
    (tmp_path / "one.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "two.jsonl").write_text(json.dumps({**row, "id": "2"}) + "\n")
    wide = {**row, "id": "3", "options": [f"o{i}" for i in range(27)], "expected": "o0"}
    (tmp_path / "wide.jsonl").write_text(json.dumps(wide) + "\n")
    c = TestClient(app)

    ok = c.post("/api/dataset-runs", json={"model": "nimble", "datasets": ["one", "two", "one"]})
    assert ok.status_code == 200
    body = ok.json()
    assert [d["name"] for d in body["datasets"]] == ["one", "two"]  # deduplicated, ordered
    assert isinstance(body["datasets"][0]["cases"][0]["query"], str)

    assert c.post("/api/dataset-runs", json={"model": "nope", "datasets": ["one"]}).status_code == 400
    assert c.post("/api/dataset-runs", json={"model": "nimble", "datasets": ["missing"]}).status_code == 404
    r = c.post("/api/dataset-runs", json={"model": "nimble", "datasets": ["one", "wide"]})
    assert r.status_code == 400 and "wide" in r.json()["detail"]
    assert c.post("/api/dataset-runs", json={"model": "nimble", "datasets": []}).status_code == 422
    assert c.get("/api/dataset-runs/unknown/events").status_code == 404


def test_pages_and_static_assets_are_served():
    from bench.app import app
    c = TestClient(app)
    assert "Compare datasets" in c.get("/datasets").text
    assert "Compare models" in c.get("/").text
    assert c.get("/static/common.js").status_code == 200 and c.get("/static/app.css").status_code == 200


NOUL = Case("n1", {"msg": "disk full"}, ["yes", "no"], "yes", "Is this an error?", kind="noul")
CHOICE = Case("c1", "hi", ["a", "b"], "a", "Pick.", {"a": "first"})


def test_question_for_and_label_from_answer():
    from bench.adapters.questions import label_from_answer, question_for
    assert question_for(NOUL) == {"type": "noul", "instructions": "Is this an error?"}
    assert question_for(CHOICE)["criteria"] == {"a": "first", "b": ""}
    assert label_from_answer(NOUL, {"noul": 0.5}) == "yes"      # threshold is inclusive
    assert label_from_answer(NOUL, {"noul": 0.49}) == "no"
    assert label_from_answer(CHOICE, {"choice": "b"}) == "b"
    import pytest
    with pytest.raises(ValueError, match="noul"):
        label_from_answer(NOUL, {"choice": "yes"})


def test_ollama_adapter_sends_noul_question_and_thresholds():
    import json
    import httpx
    from bench.adapters.ollama_decision import OllamaDecisionAdapter
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"answers": {"q": {"type": "noul", "noul": 0.93}}})

    adapter = OllamaDecisionAdapter("tev", "tev1:latest")
    adapter._http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    assert adapter.decide(NOUL) == "yes"
    assert seen[0]["questions"]["q"] == {"type": "noul", "instructions": "Is this an error?"}
    assert seen[0]["state"] == {"msg": "disk full"} and seen[0]["model"] == "tev1:latest"


def test_laya_adapter_noul(monkeypatch):
    import sys
    import types

    class FakeAgent:
        def predict(self, state, questions):
            assert questions["q"]["type"] == "noul"
            return {"answers": {"q": {"type": "noul", "noul": 0.12}}}

    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(load=lambda ckpt: FakeAgent()))
    from bench.adapters.laya import LayaAdapter
    assert LayaAdapter("laya", "x").decide(NOUL) == "no"


def test_strands_decider_adapter_builds_real_schema_objects():
    import types
    from bench.adapters.strands_decider import StrandsDeciderAdapter
    from strands_decider.schema import ChoiceAnswer, NoulAnswer, NoulQuestion
    asked = {}

    class FakeEngine:
        def ask(self, state, questions):
            asked["q"] = questions["q"]
            q = questions["q"]
            ans = NoulAnswer(noul=0.8) if isinstance(q, NoulQuestion) else ChoiceAnswer(
                choice="a", probabilities={"a": 0.9, "b": 0.1}, confidence=0.8)
            return types.SimpleNamespace(answers={"q": ans})

    adapter = StrandsDeciderAdapter("strands-decision", "x")
    adapter._engine = FakeEngine()
    assert adapter.decide(NOUL) == "yes" and isinstance(asked["q"], NoulQuestion)
    assert adapter.decide(CHOICE) == "a"


def test_dataset_kind_is_loaded(tmp_path):
    import json
    from bench.schema import load_dataset
    (tmp_path / "yn.jsonl").write_text(json.dumps(
        {"id": "1", "query": "q", "options": ["yes", "no"], "expected": "no", "kind": "noul"}) + "\n")
    assert load_dataset(tmp_path / "yn.jsonl")[0].kind == "noul"


SCORE = Case("s1", {"ticket": "x"}, ["0", "1", "2"], "1", "Rate it.",
             {"0": "low", "1": "mid", "2": "high"}, kind="score", metric="ordinal")


def test_qwk_and_mae_known_values():
    from bench.metrics import mae, qwk
    assert qwk([0, 1, 2, 1], [0, 1, 2, 1]) == 1.0
    assert qwk([0, 1, 2], [2, 1, 0]) < 0           # reversed ranking: worse than chance
    assert qwk([1, 1, 1], [1, 1, 1]) is None       # no label variation: undefined
    assert qwk([], []) is None and mae([], []) is None
    assert mae([0, 1, 2], [1, 1, 0]) == 1.0        # distances 1, 0, 2
    # hand-worked: 4 levels, one item off by one. observed disagreement = (1/9) / 4 pairs... in
    # the pooled form: num = 1/9, chance term = 32/36, so kappa = 1 - (1/9) / (32/36) = 0.875
    assert round(qwk([0, 1, 2, 3], [0, 1, 2, 2]), 4) == 0.875


def test_score_question_and_answer_mapping():
    from bench.adapters.questions import label_from_answer, question_for
    assert question_for(SCORE) == {"type": "score", "instructions": "Rate it.",
                                   "criteria": ["low", "mid", "high"]}
    # most likely level wins over the average score (JevBench's rule)
    assert label_from_answer(SCORE, {"score": 1.0, "probabilities": {"0": .5, "1": .1, "2": .4}}) == "0"
    assert label_from_answer(SCORE, {"score": 1.6}) == "2"            # fall back to rounding
    assert label_from_answer(SCORE, {"score": 7.0}) == "2"            # clamped to the scale
    assert label_from_answer(SCORE, {"score": -3.0}) == "0"
    import pytest
    with pytest.raises(ValueError, match="score"):
        label_from_answer(SCORE, {"choice": "1"})


def test_runner_reports_mae_and_qwk_for_ordinal_cases_only():
    exact = Case("m1", "q", ["0", "1", "2"], "2", kind="score", metric="exact")
    cases = [Case(str(i), "q", ["0", "1", "2"], g, kind="score", metric="ordinal")
             for i, g in enumerate(["0", "1", "2", "1"])] + [exact]
    fake = Fake("f", ["0", "1", "1", "1", "0"])     # 3rd is off by one; the exact-metric case is ignored
    summary = next(e for e in run_benchmark([fake], cases) if e["type"] == "summary")
    assert summary["ordinal_cases"] == 4 and summary["ordinal_answered"] == 4
    assert summary["mae"] == 0.25 and 0 < summary["qwk"] < 1
    assert summary["cases"] == 5 and summary["correct"] == 3        # exact match over all five


def test_runner_summary_has_no_ordinal_fields_for_choice_cases():
    summary = next(e for e in run_benchmark([Fake("f", ["refund"])], [Case("1", "q", OPTS, "refund")])
                   if e["type"] == "summary")
    assert "mae" not in summary and "qwk" not in summary


def test_ordinal_metrics_skip_unanswered_cases():
    cases = [Case(str(i), "q", ["0", "1", "2"], "1", kind="score", metric="ordinal") for i in range(3)]
    summary = next(e for e in run_benchmark([Fake("f", ["1", RuntimeError("boom"), "2"])], cases)
                   if e["type"] == "summary")
    assert summary["ordinal_cases"] == 3 and summary["ordinal_answered"] == 2   # error not scored as 0
    assert summary["mae"] == 0.5


def test_ollama_laya_and_strands_adapters_handle_score(monkeypatch):
    import json
    import sys
    import types
    import httpx
    from bench.adapters.laya import LayaAdapter
    from bench.adapters.ollama_decision import OllamaDecisionAdapter
    from bench.adapters.strands_decider import StrandsDeciderAdapter
    from strands_decider.schema import ScoreAnswer, ScoreQuestion

    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"answers": {"q": {"type": "score", "score": 1.9,
                                                           "probabilities": {"0": .1, "1": .2, "2": .7}}}})

    ollama = OllamaDecisionAdapter("tev", "tev1:latest")
    ollama._http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    assert ollama.decide(SCORE) == "2"
    assert sent[0]["questions"]["q"]["criteria"] == ["low", "mid", "high"]

    class FakeAgent:
        def predict(self, state, questions):
            assert questions["q"]["type"] == "score"
            return {"answers": {"q": {"type": "score", "score": 0.2, "probabilities": {"0": .8, "1": .1, "2": .1}}}}

    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(load=lambda c: FakeAgent()))
    assert LayaAdapter("laya", "x").decide(SCORE) == "0"

    class FakeEngine:
        def ask(self, state, questions):
            assert isinstance(questions["q"], ScoreQuestion)
            ans = ScoreAnswer(score=1.0, legend={"0": "low", "1": "mid", "2": "high"},
                              probabilities={"0": .2, "1": .6, "2": .2}, confidence=0.5)
            return types.SimpleNamespace(answers={"q": ans})

    strands = StrandsDeciderAdapter("strands-decision", "x")
    strands._engine = FakeEngine()
    assert strands.decide(SCORE) == "1"


def test_loader_reads_metric(tmp_path):
    import json
    from bench.schema import load_dataset
    (tmp_path / "s.jsonl").write_text(json.dumps(
        {"id": "1", "query": "q", "options": ["0", "1"], "expected": "1", "kind": "score",
         "metric": "ordinal", "descriptions": {"0": "a", "1": "b"}}) + "\n")
    case = load_dataset(tmp_path / "s.jsonl")[0]
    assert case.kind == "score" and case.metric == "ordinal"
