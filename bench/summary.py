"""Statistics over repeated benchmark runs.

A "runs" value is a list with one entry per run; each entry maps dataset name ->
{"results": [{"id", "ok", "ms", "err"}, ...], "summary": {the runner's summary event}}.
This is the format written by scripts/run_benchmark.py and archived in reports/data/raw/.
"""
from __future__ import annotations

import statistics as st


def dataset_stats(runs: list[dict], dataset: str) -> dict:
    """Accuracy, latency and total time for one dataset, averaged over runs."""
    summaries = [r[dataset]["summary"] for r in runs]
    results = [r[dataset]["results"] for r in runs]
    sd = (lambda xs: st.stdev(xs) if len(xs) > 1 else 0.0)
    avg = [s["avg_latency_ms"] for s in summaries]
    tot = [s["total_latency_ms"] / 1000 for s in summaries]
    p95 = [sorted(x["ms"] for x in rs)[int(0.95 * len(rs))] for rs in results]
    out = {
        "cases": summaries[0]["cases"],
        "accuracy_pct": st.mean(s["accuracy"] * 100 for s in summaries),
        "avg_latency_ms": st.mean(avg), "avg_latency_sd": sd(avg), "avg_latency_per_run": avg,
        "total_s": st.mean(tot), "total_s_sd": sd(tot),
        "p95_ms": st.mean(p95),
        "deterministic": len({tuple(x["ok"] for x in rs) for rs in results}) == 1,
        "errors": st.mean(sum(1 for x in rs if x["err"]) for rs in results),
    }
    if "mae" in summaries[0]:
        out.update(mae=st.mean(s["mae"] for s in summaries), qwk=st.mean(s["qwk"] for s in summaries),
                   ordinal_cases=summaries[0]["ordinal_cases"])
    return out


def overall_stats(runs: list[dict], datasets: list[str]) -> dict:
    """Totals across all datasets: accuracy, average latency per case, time per run."""
    cases = sum(runs[0][d]["summary"]["cases"] for d in datasets)
    per_run_total = [sum(r[d]["summary"]["total_latency_ms"] for d in datasets) / 1000 for r in runs]
    per_run_correct = [sum(r[d]["summary"]["correct"] for d in datasets) for r in runs]
    return {"cases": cases, "accuracy_pct": 100 * st.mean(per_run_correct) / cases,
            "avg_latency_ms": st.mean(per_run_total) * 1000 / cases,
            "total_s": st.mean(per_run_total),
            "total_s_sd": st.stdev(per_run_total) if len(per_run_total) > 1 else 0.0,
            "total_s_per_run": per_run_total}
