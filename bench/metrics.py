"""Ordinal metrics for score tasks, matching JevBench's harness (bench_eval.py)."""
from __future__ import annotations


def mae(gold: list[int], pred: list[int]) -> float | None:
    """Mean absolute error in levels; None when there are no predictions."""
    if not gold:
        return None
    return sum(abs(g - p) for g, p in zip(gold, pred, strict=True)) / len(gold)


def qwk(gold: list[int], pred: list[int]) -> float | None:
    """Quadratic weighted kappa over one pooled confusion matrix.

    1.0 = perfect agreement, 0 = chance level, negative = worse than chance. None when
    undefined (no data, or no variation in the labels so chance agreement is total).
    """
    if not gold:
        return None
    n = max(gold + pred) + 1
    observed = [[0.0] * n for _ in range(n)]
    for g, p in zip(gold, pred, strict=True):
        observed[g][p] += 1
    total = len(gold)
    rows = [sum(observed[i]) for i in range(n)]
    cols = [sum(observed[i][j] for i in range(n)) for j in range(n)]
    scale = (n - 1) ** 2 or 1
    num = den = 0.0
    for i in range(n):
        for j in range(n):
            w = (i - j) ** 2 / scale
            num += w * observed[i][j]
            den += w * rows[i] * cols[j] / total
    return 1 - num / den if den > 0 else None
