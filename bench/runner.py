from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from bench.adapters.base import Adapter
from bench.metrics import mae, qwk
from bench.schema import Case, Result
from bench.scoring import is_correct, match_option


def run_case(adapter: Adapter, case: Case, timeout_s: float, pool: ThreadPoolExecutor) -> Result:
    start = time.perf_counter()
    predicted: str | None = None
    error: str | None = None
    future = pool.submit(adapter.decide, case)
    try:
        raw = future.result(timeout=timeout_s)
        predicted = match_option(raw, case.options)
        if predicted is None:
            error = f"answer not in options: {raw[:80]!r}"
    except FutureTimeout:
        error = f"timeout after {timeout_s}s"
    except Exception as exc:  # surfaced to the UI, counted as incorrect
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - start) * 1000
    return Result(adapter.name, case.id, is_correct(predicted, case.expected),
                  latency_ms, predicted, case.expected, error, case.metric == "ordinal")


def summarize(model: str, results: list[Result], **extra) -> dict:
    total = sum(r.latency_ms for r in results)
    n = len(results)
    correct = sum(r.correct for r in results)
    summary = {
        "type": "summary", "model": model, **extra, "cases": n, "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "avg_latency_ms": total / n if n else 0.0,
        "total_latency_ms": total,
    }
    ordinal = [r for r in results if r.ordinal]
    if ordinal:  # score tasks: distance-aware metrics over answered cases
        pairs = [(int(r.expected), int(r.predicted)) for r in ordinal if r.predicted is not None]
        gold, pred = [g for g, _ in pairs], [p for _, p in pairs]
        summary.update(ordinal_cases=len(ordinal), ordinal_answered=len(pairs),
                       mae=mae(gold, pred), qwk=qwk(gold, pred))
    return summary


def _close(adapter: Adapter) -> None:
    close = getattr(adapter, "close", None)
    if close:
        try:
            close()
        except Exception:
            pass  # best-effort unload


def run_benchmark(adapters: list[Adapter], cases: list[Case], timeout_s: float = 60.0) -> Iterator[dict]:
    """Sequential per model (VRAM-limited); yields start/result/summary/done events."""
    summaries: list[dict] = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for adapter in adapters:
            yield {"type": "model_start", "model": adapter.name, "total": len(cases)}
            try:
                adapter.warmup()  # excluded from timing
            except Exception as exc:
                yield {"type": "model_warning", "model": adapter.name,
                       "message": f"warmup failed: {type(exc).__name__}: {exc}"}
            results: list[Result] = []
            for case in cases:
                res = run_case(adapter, case, timeout_s, pool)
                results.append(res)
                yield {"type": "result", **res.to_dict()}
            summary = summarize(adapter.name, results)
            summaries.append(summary)
            yield summary
            _close(adapter)
    yield {"type": "done", "summaries": summaries}


def run_model_on_datasets(adapter: Adapter, datasets: dict[str, list[Case]],
                          timeout_s: float = 60.0) -> Iterator[dict]:
    """One model over several datasets: loads once, then yields per-dataset events.

    Every result and summary carries its `dataset` name; `done` lists the summaries.
    """
    summaries: list[dict] = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        yield {"type": "model_start", "model": adapter.name,
               "total": sum(len(c) for c in datasets.values())}
        try:
            adapter.warmup()  # once, excluded from timing
        except Exception as exc:
            yield {"type": "model_warning", "model": adapter.name,
                   "message": f"warmup failed: {type(exc).__name__}: {exc}"}
        for name, cases in datasets.items():
            yield {"type": "dataset_start", "dataset": name, "total": len(cases)}
            results: list[Result] = []
            for case in cases:
                res = run_case(adapter, case, timeout_s, pool)
                results.append(res)
                yield {"type": "result", "dataset": name, **res.to_dict()}
            summary = summarize(adapter.name, results, dataset=name)
            summaries.append(summary)
            yield summary
        _close(adapter)
    yield {"type": "done", "summaries": summaries}
