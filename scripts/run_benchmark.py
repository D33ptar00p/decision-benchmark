#!/usr/bin/env python3
"""Run our own benchmark (the same runner as the web app) repeatedly, from the command line.

Examples
  uv run python scripts/run_benchmark.py strands-decision --runs 5 --cases 500
  uv run python scripts/run_benchmark.py laya --laya-checkpoint typed-decisions --datasets banking77,jevbench-score
  uv run python scripts/run_benchmark.py strands-decision laya nimble --runs 3 --cases 200

Model ids are those in bench/config.py (FACTORIES). For every model and every run a fresh
adapter is built (so the model is reloaded) and run over the first --cases cases of each
dataset; the timer covers only the model's decision call. Results go to
results/benchmark/<timestamp>/: <model>.json.gz (every case and timing), summary.json, summary.md.

Use AC power and a cool GPU for comparable latency (--cooldown-c, --require-ac).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench import config  # noqa: E402
from bench.hardware import ac_state, gpu_state, wait_cooldown, wait_for_ac  # noqa: E402
from bench.runner import run_model_on_datasets  # noqa: E402
from bench.schema import load_dataset  # noqa: E402
from bench.summary import dataset_stats, overall_stats  # noqa: E402


def resolve_datasets(spec: str) -> list[str]:
    available = sorted(p.stem for p in config.DATASET_DIR.glob("*.jsonl"))
    if spec == "all":
        return [d for d in available if d != "sample"]
    names = [d.strip() for d in spec.split(",") if d.strip()]
    missing = [d for d in names if d not in available]
    if missing:
        raise SystemExit(f"unknown dataset(s) {missing}; available: {available}")
    return names


def run_once(model: str, data: dict, timeout_s: float) -> dict:
    """One full pass of `model` over the datasets; returns {dataset: {results, summary}}."""
    per = {d: {"results": []} for d in data}
    for ev in run_model_on_datasets(config.FACTORIES[model](), data, timeout_s=timeout_s):
        if ev["type"] == "result":
            per[ev["dataset"]]["results"].append(
                {"id": ev["case_id"], "ok": ev["correct"], "ms": ev["latency_ms"], "err": ev["error"]})
        elif ev["type"] == "summary":
            per[ev["dataset"]]["summary"] = {k: v for k, v in ev.items() if k not in ("type", "model")}
        elif ev["type"] == "model_warning":
            print(f"    WARNING {ev['message']}")
    return per


def run_models(models: list[str], datasets: list[str], cases: int, runs: int, out_dir: Path,
               timeout_s: float = 60.0, cooldown_c: int = 0, label_suffix: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {d: load_dataset(config.DATASET_DIR / f"{d}.jsonl", cases) for d in datasets}
    short = [d for d, c in data.items() if len(c) < cases]
    if short:
        print(f"note: fewer than {cases} cases available in {short}; using what exists "
              f"({', '.join(f'{d}={len(data[d])}' for d in short)})")
    summary: dict = {"cases_per_dataset": cases, "runs": runs, "datasets": datasets, "models": {}}
    for model in models:
        label = (label_suffix or {}).get(model, model)
        print(f"\n=== {label}")
        wait_cooldown(cooldown_c)
        all_runs, conditions = [], []
        for r in range(1, runs + 1):
            conditions.append({"gpu": gpu_state(), "ac": ac_state()})
            start = time.time()
            all_runs.append(run_once(model, data, timeout_s))
            total = sum(all_runs[-1][d]["summary"]["total_latency_ms"] for d in datasets) / 1000
            acc = overall_stats(all_runs[-1:], datasets)["accuracy_pct"]
            print(f"  run {r}/{runs}: {acc:.1f}% accuracy, {total:.1f} s decision time, {time.time() - start:.0f} s wall")
        with gzip.open(out_dir / f"{label}.json.gz", "wt") as f:
            json.dump(all_runs, f)
        summary["models"][label] = {
            "datasets": {d: dataset_stats(all_runs, d) for d in datasets},
            "overall": overall_stats(all_runs, datasets),
            "conditions_at_run_start": conditions,
        }
    return summary


def to_markdown(summary: dict) -> str:
    lines = ["# Benchmark results", "",
             f"{summary['runs']} run(s), up to {summary['cases_per_dataset']} cases per dataset. Latency is the model's decision "
             "time only (loading and warm-up excluded). +/- is the standard deviation across runs.", ""]
    for label, m in summary["models"].items():
        lines += [f"## {label}", "", "| Dataset | Cases | Accuracy | Avg latency per case | Slowest 5% (p95) | Total time per dataset |",
                  "|---|---|---|---|---|---|"]
        for d, s in m["datasets"].items():
            extra = f" (MAE {s['mae']:.2f}, QWK {s['qwk']:.2f})" if "mae" in s else ""
            lines.append(f"| {d} | {s['cases']} | {s['accuracy_pct']:.1f}%{extra} | {s['avg_latency_ms']:.1f} ms (+/-{s['avg_latency_sd']:.1f}) "
                         f"| {s['p95_ms']:.0f} ms | {s['total_s']:.1f} s (+/-{s['total_s_sd']:.1f}) |")
        o = m["overall"]
        lines.append(f"| **All** | **{o['cases']:,}** | **{o['accuracy_pct']:.1f}%** | **{o['avg_latency_ms']:.1f} ms** | | "
                     f"**{o['total_s']:.1f} s** per run (+/-{o['total_s_sd']:.1f}) |")
        deterministic = all(s["deterministic"] for s in m["datasets"].values())
        errors = sum(s["errors"] for s in m["datasets"].values())
        lines += ["", f"Same answers in every run: {deterministic}. Errors per run: {errors:.0f}.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help=f"model ids: {', '.join(config.FACTORIES)}")
    ap.add_argument("--datasets", default="all", help="comma list or 'all' (every dataset except 'sample')")
    ap.add_argument("--cases", type=int, default=500, help="cases per dataset (the first N of each file)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds allowed per case")
    ap.add_argument("--laya-checkpoint", help="Laya checkpoint (english, typed-decisions or a Hugging Face id)")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--cooldown-c", type=int, default=60, help="wait for the GPU to reach this temperature before each model (0 = off)")
    ap.add_argument("--require-ac", action="store_true", help="wait until AC power is connected")
    ap.add_argument("--list", action="store_true", help="list model ids and datasets, then exit")
    args = ap.parse_args()

    if args.list:
        print("models:", ", ".join(config.FACTORIES))
        print("datasets:", ", ".join(sorted(p.stem for p in config.DATASET_DIR.glob("*.jsonl"))))
        return
    unknown = [m for m in args.models if m not in config.FACTORIES]
    if not args.models or unknown:
        ap.error(f"give model ids from: {', '.join(config.FACTORIES)}" + (f" (unknown: {unknown})" if unknown else ""))
    suffix = {}
    if args.laya_checkpoint:
        config.LAYA_CHECKPOINT = args.laya_checkpoint
        suffix["laya"] = f"laya-{args.laya_checkpoint.split('/')[-1]}"
    if args.require_ac:
        wait_for_ac()
    elif ac_state() is False:
        print("WARNING: running on battery; the GPU is power-capped and latency will be unreliable.")

    out_dir = args.out_dir or ROOT / "results" / "benchmark" / datetime.now().strftime("%Y%m%d-%H%M%S")
    summary = run_models(args.models, resolve_datasets(args.datasets), args.cases, args.runs, out_dir,
                         args.timeout, args.cooldown_c, suffix)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    md = to_markdown(summary)
    (out_dir / "summary.md").write_text(md)
    print("\n" + md)
    print(f"results written to {out_dir}")


if __name__ == "__main__":
    main()
