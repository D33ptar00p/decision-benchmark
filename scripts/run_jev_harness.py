#!/usr/bin/env python3
"""Run the official JevBench harness (bench_eval.py, unmodified) against one or more local models.

Examples
  # models the script can start and stop itself (presets):
  uv run python scripts/run_jev_harness.py strands-decision laya-typed --runs 3

  # a server you already started (the URL must be local):
  uv run python scripts/run_jev_harness.py mymodel \\
      --server mymodel=http://127.0.0.1:9000/v1/systemone --request-model mymodel=my-model-id

  # quick check on a few cases, no cool-down wait:
  uv run python scripts/run_jev_harness.py laya-english --runs 1 --limit 50 --cooldown-c 0

Presets: strands-decision, laya-english, laya-typed.
What it does, per model: waits for the GPU to cool (optional) -> starts the model's local
/v1/systemone server -> runs the harness --runs times with per-request timing -> stops the
server. Then it writes summary.md / summary.json: accuracy by question type and by dataset,
average latency per case, total time per dataset, run times and GPU state per run.

Notes
  * The harness normally calls TypeSafe's hosted API. Here it is pointed at a local server
    (scripts/jev_harness_wrapper.py refuses non-local URLs), so no data leaves the machine.
  * The Laya server only treats exact checkpoint names (english, typed-decisions) in the
    request's `model` field as a checkpoint choice; anything else means "auto-route". The
    presets send the exact name. For --server models set --request-model accordingly.
  * Latency is only comparable under the same conditions: use AC power, a cool GPU
    (--cooldown-c) and close other GPU programs. Add --require-ac to wait for the charger.
  * The JevBench repo is cloned to .cache/jev-bench on first use (or pass --jev-dir).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import signal
import socket
import statistics as st
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.hardware import ac_state, auto_device, gpu_state, wait_cooldown, wait_for_ac  # noqa: E402
from bench.metrics import mae, qwk  # noqa: E402  (same definitions as the harness)

JEV_REPO = "https://github.com/model-collapse/jev-bench.git"
WRAPPER = Path(__file__).resolve().parent / "jev_harness_wrapper.py"
BIN = Path(sys.executable).parent
TYPE_NAMES = {"choice": "choice", "noul": "yes/no", "score": "score"}


# ----------------------------------------------------------------------------- presets
def _strands(port: int, device: str) -> tuple[list[str], dict, str]:
    cmd = [str(BIN / "strands-decider"), "serve", "StrandsAgents/strands-decider-2B-hobson-v19",
           "--host", "127.0.0.1", "--port", str(port), "--device", device,
           "--model-name", "strands-decider"]
    return cmd, {}, "strands-decider"


def _laya(checkpoint: str):
    def build(port: int, device: str) -> tuple[list[str], dict, str]:
        env = {"LAYA_HOST": "127.0.0.1", "LAYA_PORT": str(port), "LAYA_DEVICE": device,
               "LAYA_MODELS": checkpoint, "LAYA_DEFAULT_MODEL": checkpoint,
               "LAYA_AUTO_TASK": "0", "LAYA_JEV_STRICT": "1"}
        return [str(BIN / "laya-serve")], env, checkpoint  # request model = exact checkpoint name
    return build


PRESETS = {"strands-decision": _strands, "laya-english": _laya("english"), "laya-typed": _laya("typed-decisions")}


# ----------------------------------------------------------------------------- helpers
def check_local_url(url: str) -> str:
    if not url.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise SystemExit(f"refusing non-local URL: {url}")
    return url


def parse_pairs(items: list[str] | None, what: str) -> dict[str, str]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"{what} expects NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def health_ok(url: str) -> bool:
    base = re.match(r"(http://[^/]+)", url).group(1)
    try:
        urllib.request.urlopen(base + "/health", timeout=3)
        return True
    except Exception:
        return False


class Server:
    """A preset's local /v1/systemone server, started and stopped by this script."""

    def __init__(self, name: str, cmd: list[str], env: dict, port: int, log_path: Path):
        self.name, self.cmd, self.env, self.port, self.log_path = name, cmd, env, port, log_path
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/systemone"

    def start(self, timeout_s: int = 600) -> None:
        if not Path(self.cmd[0]).exists():
            raise SystemExit(f"{self.cmd[0]} not found - is the package installed in this environment?")
        with open(self.log_path, "w") as log:
            self.proc = subprocess.Popen(self.cmd, env={**os.environ, **self.env}, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        start = time.time()
        while time.time() - start < timeout_s:
            if self.proc.poll() is not None:
                raise SystemExit(f"{self.name} server exited early; see {self.log_path}")
            if health_ok(self.url):
                return
            time.sleep(2)
        self.stop()
        raise SystemExit(f"{self.name} server did not become healthy in {timeout_s}s; see {self.log_path}")

    def stop(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        os.killpg(self.proc.pid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()


def ensure_jev_repo(jev_dir: Path) -> None:
    if (jev_dir / "bench_eval.py").exists() and (jev_dir / "data").exists():
        return
    print(f"JevBench not found at {jev_dir}; cloning {JEV_REPO} ...")
    jev_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", JEV_REPO, str(jev_dir)], check=True)


# ----------------------------------------------------------------------------- one harness run
def run_harness(jev_dir: Path, url: str, request_model: str, prefix: Path, extra: list[str]) -> dict:
    env = {**os.environ, "JEV_DIR": str(jev_dir), "JEV_LOCAL_URL": url,
           "JEV_TIMING_OUT": f"{prefix}.timing.json"}
    cmd = [sys.executable, str(WRAPPER), "--backend", "jev", "--model", request_model,
           "--data", str(jev_dir / "data"), "--out", f"{prefix}.json", *extra]
    start_state = gpu_state()
    with open(f"{prefix}.txt", "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    text = Path(f"{prefix}.txt").read_text()
    wall = re.search(r"WALL_SECONDS ([\d.]+)", text)
    if proc.returncode != 0 or not wall:
        raise SystemExit(f"harness run failed (exit {proc.returncode}); see {prefix}.txt")
    return {"preds": json.load(open(f"{prefix}.json")), "timings": json.load(open(f"{prefix}.timing.json")),
            "wall": float(wall.group(1)), "gpu_at_start": start_state}


# ----------------------------------------------------------------------------- aggregation
def aggregate(runs: list[dict]) -> dict:
    """runs: [{preds: [harness prediction dicts], timings: {id: ms}, wall: s, gpu_at_start: dict|None}]."""
    ids = [p["id"] for p in runs[0]["preds"]]
    meta = {p["id"]: (p["domain"], p["type"]) for p in runs[0]["preds"]}
    by_run = [{p["id"]: p for p in r["preds"]} for r in runs]

    def group(sel: list[str]) -> dict:
        acc = st.mean(100 * sum(1 for i in sel if pr[i]["correct"]) / len(sel) for pr in by_run)
        avg = st.mean(st.mean(r["timings"][i] for i in sel) for r in runs)
        tot = st.mean(sum(r["timings"][i] for i in sel) / 1000 for r in runs)
        return {"cases": len(sel), "accuracy_pct": acc, "avg_latency_ms": avg, "total_s": tot}

    result = {"runs": len(runs), "cases": len(ids), "overall": group(ids),
              "wall_s": [r["wall"] for r in runs], "gpu_at_start": [r["gpu_at_start"] for r in runs],
              "deterministic": len({json.dumps(sorted((i, pr[i]["pred"]) for i in pr)) for pr in by_run}) == 1,
              "unanswered": [sum(1 for p in r["preds"] if p["pred"] is None) for r in runs]}
    result["by_type"] = {t: group([i for i in ids if meta[i][1] == t]) for t in TYPE_NAMES
                         if any(meta[i][1] == t for i in ids)}
    groups = collections.defaultdict(list)
    for i in ids:
        groups[meta[i]].append(i)
    result["by_dataset"] = {f"{d}|{t}": group(sel) for (d, t), sel in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0]))}
    ordinal = [p for p in runs[0]["preds"] if p["eval_metric"] == "ordinal" and p["pred"] is not None
               and str(p["pred"]).lstrip("-").isdigit()]
    if ordinal:
        gold, pred = [int(p["gold"]) for p in ordinal], [int(p["pred"]) for p in ordinal]
        result["ordinal"] = {"cases": len(ordinal), "mae": mae(gold, pred), "qwk": qwk(gold, pred)}
    return result


def to_markdown(results: dict[str, dict]) -> str:
    names = list(results)
    lines = ["# JevBench harness results", "",
             "Cells: accuracy · average latency per case (end to end, HTTP + model) · total time, mean over runs.", ""]
    lines += ["## By question type", "", "| Type | Cases | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for t in TYPE_NAMES:
        if all(t in r["by_type"] for r in results.values()):
            n = next(iter(results.values()))["by_type"][t]["cases"]
            lines.append(f"| {TYPE_NAMES[t]} | {n:,} | " + " | ".join(
                "{accuracy_pct:.1f}% · {avg_latency_ms:.0f} ms · {total_s:.1f} s".format(**r["by_type"][t]) for r in results.values()) + " |")
    o = {m: r["overall"] for m, r in results.items()}
    n_all = next(iter(results.values()))["cases"]
    lines.append(f"| **All** | **{n_all:,}** | " + " | ".join(
        "**{accuracy_pct:.1f}% · {avg_latency_ms:.0f} ms · {total_s:.1f} s**".format(**o[m]) for m in names) + " |")
    lines += ["", "## Per dataset (topic and question type)", "",
              "| Dataset | Type | Cases | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
    first = next(iter(results.values()))["by_dataset"]
    for key in first:
        d, t = key.split("|")
        lines.append(f"| {d} | {TYPE_NAMES[t]} | {first[key]['cases']} | " + " | ".join(
            "{accuracy_pct:.1f}% · {avg_latency_ms:.0f} ms · {total_s:.1f} s".format(**r["by_dataset"][key]) for r in results.values()) + " |")
    lines += ["", "## Run times and conditions", "", "| Model | Run times (s) | Mean (s) | GPU at start of each run (C / MHz / throttled) | Same answers every run | Unanswered per run |",
              "|---|---|---|---|---|---|"]
    for m, r in results.items():
        gpu = "; ".join("n/a" if g is None else f"{g['temp_c']} / {g['clock_mhz']} / {'yes' if g['thermal_throttle'] else 'no'}" for g in r["gpu_at_start"])
        lines.append(f"| {m} | {', '.join(f'{x:.1f}' for x in r['wall_s'])} | {st.mean(r['wall_s']):.1f} | {gpu} | {r['deterministic']} | {r['unanswered']} |")
    ordinal = {m: r["ordinal"] for m, r in results.items() if "ordinal" in r}
    if ordinal:
        lines += ["", "## Score tasks, ordinal cases (lower MAE and higher QWK are better)", "", "| Model | Cases | MAE (levels) | QWK |", "|---|---|---|---|"]
        for m, x in ordinal.items():
            lines.append(f"| {m} | {x['cases']} | {x['mae']:.2f} | {x['qwk']:.3f} |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help=f"preset names ({', '.join(PRESETS)}) or names given with --server")
    ap.add_argument("--server", action="append", metavar="NAME=URL",
                    help="a /v1/systemone server you started yourself (local URLs only)")
    ap.add_argument("--request-model", action="append", metavar="NAME=VALUE",
                    help="value for the request's `model` field (default: the model name)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="harness --limit (0 = all 2,934 cases)")
    ap.add_argument("--tier", choices=["all", "gold", "silver"], default="all")
    ap.add_argument("--types", default="", help="comma list, e.g. choice,noul")
    ap.add_argument("--jev-dir", type=Path, default=ROOT / ".cache" / "jev-bench")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default="auto", help="cuda | cpu | mps | auto (presets only)")
    ap.add_argument("--cooldown-c", type=int, default=60, help="wait for the GPU to reach this temperature before each model (0 = off)")
    ap.add_argument("--require-ac", action="store_true", help="wait until AC power is connected before starting")
    ap.add_argument("--list", action="store_true", help="list presets and exit")
    args = ap.parse_args()

    if args.list:
        print("presets:", ", ".join(PRESETS))
        return
    servers = {k: check_local_url(v) for k, v in parse_pairs(args.server, "--server").items()}
    request_models = parse_pairs(args.request_model, "--request-model")
    if not args.models:
        ap.error("give at least one model (preset or --server NAME=URL)")
    unknown = [m for m in args.models if m not in PRESETS and m not in servers]
    if unknown:
        ap.error(f"unknown model(s) {unknown}: not a preset and no --server given")

    ensure_jev_repo(args.jev_dir)
    out_dir = args.out_dir or ROOT / "results" / "jev-harness" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = auto_device() if args.device == "auto" else args.device
    extra = (["--limit", str(args.limit)] if args.limit else []) + (["--tier", args.tier] if args.tier != "all" else []) \
        + (["--types", args.types] if args.types else [])

    if args.require_ac:
        wait_for_ac()
    elif ac_state() is False:
        print("WARNING: running on battery; the GPU is power-capped and latency will be unreliable.")

    results: dict[str, dict] = {}
    for name in args.models:
        print(f"\n=== {name}")
        (out_dir / name).mkdir(exist_ok=True)
        server = None
        try:
            wait_cooldown(args.cooldown_c)
            if name in PRESETS:
                port = free_port()
                cmd, env, req_model = PRESETS[name](port, device)
                server = Server(name, cmd, env, port, out_dir / name / "server.log")
                print(f"starting server on port {port} (device {device}) ...")
                server.start()
                url = server.url
            else:
                url, req_model = servers[name], request_models.get(name, name)
                if not health_ok(url):
                    print(f"  (no /health answer from {url}; continuing anyway)")
            req_model = request_models.get(name, req_model)
            runs = []
            for r in range(1, args.runs + 1):
                print(f"  run {r}/{args.runs} ...", flush=True)
                runs.append(run_harness(args.jev_dir, url, req_model, out_dir / name / f"run{r}", extra))
                o = aggregate(runs[-1:])["overall"]
                print(f"    {o['accuracy_pct']:.1f}% in {runs[-1]['wall']:.1f} s")
            results[name] = aggregate(runs)
        finally:
            if server:
                server.stop()

    (out_dir / "summary.json").write_text(json.dumps(results, indent=1))
    md = to_markdown(results)
    (out_dir / "summary.md").write_text(md)
    print("\n" + md)
    print(f"results written to {out_dir}")


if __name__ == "__main__":
    main()
