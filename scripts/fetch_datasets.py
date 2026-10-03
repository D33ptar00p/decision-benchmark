"""Normalize open datasets into datasets/<name>.jsonl: {id, query, options, expected}.

Usage: uv run python scripts/fetch_datasets.py [banking77] [clinc150] [decision-bench] [jevbench]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "datasets"
MAX_ROWS = 1000  # sample cap (the UI limit is also 1000)
# Ollama's decision API accepts 2-26 options per question, so each case offers the
# correct label plus (N_OPTIONS - 1) random distractors, identical for every model.
N_OPTIONS = 10


def write(name: str, rows: list[dict]) -> None:
    OUT.mkdir(exist_ok=True)
    with (OUT / f"{name}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> datasets/{name}.jsonl")


def sample(rows: list[dict]) -> list[dict]:
    rng = random.Random(0)  # fixed seed: same cases and distractors every run
    rng.shuffle(rows)
    rows = rows[:MAX_ROWS]
    for r in rows:
        others = [o for o in r["options"] if o != r["expected"]]
        options = [r["expected"], *rng.sample(others, N_OPTIONS - 1)]
        rng.shuffle(options)
        r["options"] = options
    return rows


def banking77() -> None:
    ds = load_dataset("mteb/banking77", split="test")  # PolyAI repo uses a dead loading script
    labels = sorted({r["label_text"] for r in ds})
    write("banking77", sample([
        {"id": f"b77-{i}", "query": r["text"], "options": labels, "expected": r["label_text"]}
        for i, r in enumerate(ds)
    ]))


def clinc150() -> None:
    ds = load_dataset("clinc/clinc_oos", "plus", split="test")
    labels = ds.features["intent"].names
    write("clinc150", sample([
        {"id": f"clinc-{i}", "query": r["text"], "options": labels, "expected": labels[r["intent"]]}
        for i, r in enumerate(ds)
    ]))


def decision_bench() -> None:
    """Convert a local Decision Bench jsonl (DECISION_BENCH_PATH; default datasets/source/, then the repo root).

    Text-only rows are kept; image rows are skipped (these models take text). All rows
    are kept (no cap) and shuffled with a fixed seed so categories are mixed.
    """
    import os

    candidates = [Path(os.environ["DECISION_BENCH_PATH"])] if os.environ.get("DECISION_BENCH_PATH") else [
        OUT / "source" / "decision-bench-1071.jsonl", OUT.parent / "decision-bench-1071.jsonl"]
    src = next((c for c in candidates if c.exists()), candidates[0])
    if not src.exists():
        raise SystemExit(f"Decision Bench file not found; put decision-bench-1071.jsonl in {OUT / 'source'} (or set DECISION_BENCH_PATH)")
    rows = []
    skipped = 0
    for line in src.open():
        r = json.loads(line)
        if r["modality"] != "text":
            skipped += 1
            continue
        q = json.loads(r["questions"])[0]  # one choice question per row
        try:
            state = json.loads(r["state"])
        except json.JSONDecodeError:
            state = r["state"]
        rows.append({
            "id": r["id"], "category": r["category"], "query": state,
            "instructions": f'{q["instructions"]} {q["ask"]}',
            "options": q["option_order"], "descriptions": q["options"], "expected": q["gold"],
        })
    random.Random(0).shuffle(rows)
    write("decision-bench", rows)
    print(f"skipped {skipped} image rows")


def _unwrap(value):
    """JevBench stores some JSON as (possibly doubly-encoded) strings; decode to a value."""
    for _ in range(2):
        if isinstance(value, str) and value.lstrip()[:1] in ('{', '[', '"'):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                break
    return value


def jevbench() -> None:
    """JevBench (github.com/model-collapse/jev-bench, Apache-2.0): choice, yes/no and score tasks.

    Source dir: JEVBENCH_DIR, else a fresh shallow clone.
    Cases with more than N_OPTIONS options (banking77) get distractor sampling, as above.
    """
    import os
    import subprocess
    import tempfile

    rng = random.Random(0)
    tmp = None
    src = os.environ.get("JEVBENCH_DIR")
    if not src:
        tmp = tempfile.TemporaryDirectory()
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/model-collapse/jev-bench.git", tmp.name], check=True)
        src = tmp.name
    rows = []
    for path in sorted(Path(src, "data").glob("*/choice.jsonl")):
        for line in path.open():
            r = json.loads(line)
            q = _unwrap(r["question"])
            options = list(q["options"])
            if r["label"] not in options:
                continue  # malformed row
            if len(options) > N_OPTIONS:
                others = [o for o in options if o != r["label"]]
                options = [r["label"], *rng.sample(others, N_OPTIONS - 1)]
                rng.shuffle(options)
            rows.append({
                "id": r["id"], "category": r["domain"], "query": _unwrap(r["state"]),
                "instructions": q["instructions"], "options": options,
                "descriptions": {o: q["options"][o] for o in options}, "expected": r["label"],
            })
    rng.shuffle(rows)
    write("jevbench-choice", rows)

    # yes/no (noul): the model returns P(yes); the harness answers "yes" at >= 0.5
    noul_rows = []
    for path in sorted(Path(src, "data").glob("*/noul.jsonl")):
        for line in path.open():
            r = json.loads(line)
            if r["label"] not in ("yes", "no"):
                continue  # malformed row
            noul_rows.append({
                "id": r["id"], "category": r["domain"], "kind": "noul",
                "query": _unwrap(r["state"]), "instructions": _unwrap(r["question"])["instructions"],
                "options": ["yes", "no"], "expected": r["label"],
            })
    random.Random(1).shuffle(noul_rows)
    write("jevbench-noul", noul_rows)

    # score: rate on an ordered scale. Options are the level indices "0".."n-1"; the
    # level descriptions travel as `descriptions`. 728 rows are ordinal (MAE + QWK),
    # 100 (moderation) are exact-match.
    score_rows = []
    for path in sorted(Path(src, "data").glob("*/score.jsonl")):
        for line in path.open():
            r = json.loads(line)
            q = _unwrap(r["question"])
            levels = q["levels"]
            if not str(r["label"]).isdigit() or int(r["label"]) >= len(levels):
                continue  # malformed row
            options = [str(i) for i in range(len(levels))]
            score_rows.append({
                "id": r["id"], "category": r["domain"], "kind": "score", "metric": r["eval_metric"],
                "query": _unwrap(r["state"]), "instructions": q["instructions"],
                "options": options, "descriptions": dict(zip(options, levels)),
                "expected": str(r["label"]),
            })
    random.Random(2).shuffle(score_rows)
    write("jevbench-score", score_rows)
    if tmp:
        tmp.cleanup()


FETCHERS = {"banking77": banking77, "clinc150": clinc150, "decision-bench": decision_bench,
            "jevbench": jevbench}

if __name__ == "__main__":
    for name in sys.argv[1:] or FETCHERS:
        FETCHERS[name]()
