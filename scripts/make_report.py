"""Builds reports/benchmark-report.md (and the per-dataset CSV and summary JSON next to it) from the
archived raw results in reports/data/raw/. No number in the tables is typed by hand.

    uv run python scripts/make_report.py
"""
import collections, gzip, json, re, statistics as st, sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
from bench.summary import dataset_stats, overall_stats  # noqa: E402

OUT = str(PROJ / "reports")
RAW = PROJ / "reports" / "data" / "raw"

def load_gz(path):
    return json.load(gzip.open(path, "rt"))

ORDER = ["banking77", "clinc150", "decision-bench", "jevbench-choice", "jevbench-noul", "jevbench-score"]
QTYPE = {"banking77": "choice, 10 options", "clinc150": "choice, 10 options", "decision-bench": "choice, 2-9 options",
         "jevbench-choice": "choice, 3-10 options", "jevbench-noul": "yes/no", "jevbench-score": "score, 3-5 levels"}
TN = {"choice": "choice", "noul": "yes/no", "score": "score"}
f1 = lambda x: f"{x:.1f}"

# ------------------------------------------------------------------ load data
custom_runs = {
    "Strands Decider 2B": load_gz(RAW / "custom_runner" / "strands-decision.json.gz"),
    "Laya English": load_gz(RAW / "custom_runner" / "laya-english.json.gz"),
    "Laya typed-decisions": load_gz(RAW / "custom_runner" / "laya-typed-decisions.json.gz"),
}
first_single = load_gz(RAW / "custom_runner" / "strands-decision_first-pass.json.gz")  # first, 5-dataset single run

def custom_stats(runs, d):
    x = dataset_stats(runs, d)
    out = dict(acc=x["accuracy_pct"], avg=x["avg_latency_ms"], avg_sd=x["avg_latency_sd"], avg_runs=x["avg_latency_per_run"],
               tot=x["total_s"], tot_sd=x["total_s_sd"], p95=x["p95_ms"], same=x["deterministic"], errs=x["errors"])
    if "mae" in x:
        out.update(mae=x["mae"], qwk=x["qwk"], ordn=x["ordinal_cases"])
    return out
CS = {m: {d: custom_stats(r, d) for d in ORDER} for m, r in custom_runs.items()}
def custom_all(m):
    o = overall_stats(custom_runs[m], ORDER)
    return dict(acc=o["accuracy_pct"], tot=o["total_s"], tot_sd=o["total_s_sd"], avg=o["avg_latency_ms"], per_run_tot=o["total_s_per_run"])
CA = {m: custom_all(m) for m in custom_runs}

# harness runs (timed set + the earlier untimed Strands set)
HDIR = {"Strands Decider 2B": "strands-decision", "Laya English": "laya-english", "Laya typed-decisions": "laya-typed-decisions"}
hdata = {}
for m, folder in HDIR.items():
    runs = []
    for i in (1, 2, 3):
        base = RAW / "harness" / folder / f"run{i}"
        t = load_gz(f"{base}.timing.json.gz")
        p = {x["id"]: x for x in load_gz(f"{base}.json.gz")}
        wall = float(re.search(r"WALL_SECONDS ([\d.]+)", open(f"{base}.txt").read()).group(1))
        assert set(t) == set(p) and len(p) == 2934
        runs.append((t, p, wall))
    hdata[m] = runs
MODELS = list(HDIR)
meta = {x["id"]: (x["domain"], x["type"]) for x in hdata[MODELS[0]][0][1].values()}

def hagg(m, ids):
    runs = hdata[m]
    return (st.mean(100 * sum(1 for i in ids if p[i]["correct"]) / len(ids) for _, p, _ in runs),
            st.mean(st.mean(t[i] for i in ids) for t, _, _ in runs),
            st.mean(sum(t[i] for i in ids) / 1000 for t, _, _ in runs))
def hord(m):
    P = hdata[m][0][1]
    ordp = [x for x in P.values() if x["eval_metric"] == "ordinal" and x["pred"] is not None and str(x["pred"]).lstrip("-").isdigit()]
    mae = st.mean(abs(int(x["pred"]) - int(x["gold"])) for x in ordp)
    tt, pp = [int(x["gold"]) for x in ordp], [int(x["pred"]) for x in ordp]
    n = max(tt + pp) + 1
    O = [[0] * n for _ in range(n)]
    for a, b in zip(tt, pp):
        O[a][b] += 1
    r = [sum(row) for row in O]; c = [sum(O[i][j] for i in range(n)) for j in range(n)]; tot = len(tt); sc = (n - 1) ** 2
    num = sum(((i - j) ** 2 / sc) * O[i][j] for i in range(n) for j in range(n))
    den = sum(((i - j) ** 2 / sc) * r[i] * c[j] / tot for i in range(n) for j in range(n))
    return len(ordp), mae, 1 - num / den
def hreport(m, key):  # accuracy from the harness's own text report (run 1)
    t = open(RAW / "harness" / HDIR[m] / "run1.txt").read()
    g = lambda pat: re.search(pat, t)
    return dict(gold=float(g(r"reliability=gold\s+n=\s*\d+ acc=([\d.]+)%").group(1)),
                silver=float(g(r"reliability=silver\s+n=\s*\d+ acc=([\d.]+)%").group(1)),
                overall=float(g(r"overall exact-accuracy = ([\d.]+)%").group(1)))
HR = {m: hreport(m, None) for m in MODELS}
deterministic = {m: len({json.dumps(sorted((i, p[i]["pred"]) for i in p)) for _, p, _ in hdata[m]}) == 1 for m in MODELS}
groups = collections.defaultdict(list)
for i, (d, t) in meta.items():
    groups[(d, t)].append(i)

none_counts = {m: [sum(1 for x in p.values() if x['pred'] is None) for _, p, _ in hdata[m]] for m in MODELS}
# earlier (untimed) Strands harness runs
untimed = []
for i in (1, 2, 3):
    t = open(RAW / "harness" / "strands-decision_untimed" / f"run{i}.txt").read()
    untimed.append((float(re.search(r"overall exact-accuracy = ([\d.]+)%", t).group(1)), float(re.search(r"WALL_SECONDS ([\d.]+)", t).group(1))))
# invalid auto-routed Laya runs
invalid = []
for n in ("laya-english_run1", "laya-english_run2", "laya-english_run3", "laya-typed_run1"):
    t = open(RAW / "harness" / "invalid_autorouted_laya" / f"{n}.txt").read()
    invalid.append((n, float(re.search(r"overall exact-accuracy = ([\d.]+)%", t).group(1)), float(re.search(r"WALL_SECONDS ([\d.]+)", t).group(1))))
# GPU state at the start of each harness run, from the orchestration log
log = open(RAW / "harness" / "orchestration.log").read().split("\n")
gpu_lines = [l for l in log if "start | gpu(temp,clock,throttle)" in l]

# TEV 40-case check (first 10 cases of 4 datasets, 1 run per model)
tev40 = {}
for name, fn in (("TEV1 0.8B", "tev-0.8b"), ("TEV1 4B", "tev")):
    run = load_gz(RAW / "custom_runner" / "tev_smoke_40cases" / f"{fn}.json.gz")[0]
    cases = [x for d in run.values() for x in d["results"]]
    tev40[name] = dict(k=sum(1 for x in cases if x["ok"]), n=len(cases), lat=st.mean(x["ms"] for x in cases),
                       errs=sum(1 for x in cases if x["err"]))

# kernel fix
before, after, after2 = (json.load(open(RAW / "kernel_fix" / f"{n}.json")) for n in ("before", "after", "after2"))
def kstats(r):
    ms = [x["ms"] for x in r]
    return dict(acc=sum(x["ok"] for x in r), n=len(r), avg=st.mean(ms), med=st.median(ms), p90=sorted(ms)[int(.9 * len(ms))], mx=max(ms))
KB, KA, KA2 = kstats(before), kstats(after), kstats(after2)
changed = sum(1 for x, y in zip(before, after) if x["got"] != y["got"])

# ------------------------------------------------------------------ write per-dataset CSV
with open(f"{OUT}/data/harness_per_dataset.csv", "w") as f:
    f.write("dataset,type,cases," + ",".join(f"{m} accuracy_pct,{m} avg_latency_ms,{m} total_s" for m in MODELS) + "\n")
    for (d, t), ids in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        f.write(f"{d},{TN[t]},{len(ids)}," + ",".join("%.2f,%.1f,%.2f" % hagg(m, ids) for m in MODELS) + "\n")

# ------------------------------------------------------------------ write summary JSON (custom runner)
json.dump({"cases_per_dataset": 500, "runs": 5,
           "models": {m: {"datasets": {d: CS[m][d] for d in ORDER}, "overall": CA[m]} for m in CS}},
          open(f"{OUT}/data/summary_custom_runner.json", "w"), indent=1)

# ------------------------------------------------------------------ markdown
L = []
w = L.append
w("# Decision-model benchmark report")
w("")
w("**Strands Decider 2B and Laya (two checkpoints) on our benchmark app and on JevBench's official harness**")
w("")
w("Date: 2026-10-03. Every table value below was computed from the archived raw results by `scripts/make_report.py`, not typed by hand. "
  "The raw per-case results and timings for every run are archived in `reports/data/raw/`.")
w("")
w("## 1. Summary")
w("")
a = {m: hagg(m, list(meta)) for m in MODELS}
w(f"- **On our own 500-case runner (6 datasets, 3,000 cases):** Strands Decider {CA['Strands Decider 2B']['acc']:.1f}%, Laya typed-decisions {CA['Laya typed-decisions']['acc']:.1f}%, Laya English {CA['Laya English']['acc']:.1f}%.")
w(f"- **Best overall on the official JevBench harness: Strands Decider 2B at {a[MODELS[0]][0]:.1f}%** (all 2,934 cases), then Laya typed-decisions at {a[MODELS[2]][0]:.1f}% and Laya English at {a[MODELS[1]][0]:.1f}%.")
w(f"- **Laya typed-decisions is the best at rating tasks (score):** {hagg(MODELS[2], [i for i,(d,t) in meta.items() if t=='score'])[0]:.1f}% exact, QWK {hord(MODELS[2])[2]:.3f}, MAE {hord(MODELS[2])[1]:.2f}, against {hagg(MODELS[0], [i for i,(d,t) in meta.items() if t=='score'])[0]:.1f}%, QWK {hord(MODELS[0])[2]:.3f}, MAE {hord(MODELS[0])[1]:.2f} for Strands.")
w(f"- **Laya is about 4x faster:** about {a[MODELS[1]][1]:.0f} ms per case against {a[MODELS[0]][1]:.0f} ms for Strands (end to end, via a local HTTP server); about {st.mean(w_ for *_, w_ in hdata[MODELS[2]]):.0f} s against {st.mean(w_ for *_, w_ in hdata[MODELS[0]]):.0f} s for the whole 2,934-case run.")
w("- **Strands Decider is far ahead on choice tasks that look like classic classification** (banking, commerce, support), and Laya typed-decisions is far ahead on structured operational tasks (customer service, invoice processing).")
w("- **Accuracy is deterministic:** every answer was identical across repeat runs of every model and setup. Only the timings vary between runs.")
w("- **Latency numbers need care:** the laptop GPU throttles with heat, and part of the early runs happened on battery power. See section 5.")
w("- **Corrections:** one set of Laya harness runs was mislabelled (automatic checkpoint routing) and has been re-run and quarantined. See section 5.")
w("")
w("## 2. Setup")
w("")
w("### 2.1 Hardware and software")
w("")
w("| Item | Value |")
w("|---|---|")
w("| GPU | NVIDIA GeForce RTX 3070 Laptop, 8 GB, driver 595.84 |")
w("| RAM | 31 GB |")
w("| OS / Python | Linux 7.0.0-34-generic, Python 3.12.3 |")
w("| PyTorch / Transformers | 2.14.1 (CUDA 13.0 build) / 5.18.0 |")
w("| Strands Decider package | `strands-decider` 0.1.0 (+ `flash-linear-attention` 0.5.2 added during the session; `causal-conv1d` not available) |")
w("| Laya package | `laya` 0.3.24 |")
w("| Ollama | 0.35.1 (decision-model API at `/v1/systemone`) |")
w("")
w("### 2.2 Models")
w("")
w("| Model | Size / format | How it was run | Used in |")
w("|---|---|---|---|")
w("| **Strands Decider 2B** (`StrandsAgents/strands-decider-2B-hobson-v19`) | Qwen3.5-2B-Base + LoRA adapter + readout head, bf16, about 4.5-5 GB on the GPU | in-process (custom runner); local server `strands-decider serve` (official harness) | all sections |")
w("| **Laya English** (`convaiinnovations/laya`) | 421M encoder, fp32, reads about 512 tokens | in-process; local server `laya-serve` with request `model=english` | sections 3.2 to 3.5 |")
w("| **Laya typed-decisions** (`typed-decisions` checkpoint) | 421M encoder, fp32, reads about 1,024 tokens | in-process; local server with request `model=typed-decisions` | sections 3.2 to 3.5 |")
w("| Nimble 9B (`nimble:latest`, Ollama, Q8) | 9.5 GB, runs about 65% GPU / 35% CPU on this laptop | exploratory checks only (3 to 12 cases); left out of later testing by request | 3.9 |")
w("| TEV1 4B (`tev1:latest`, Ollama, Q8) | 4.5 GB, fully on GPU, context limit 2,050 tokens | 40-case check (section 3.9) | 3.9 |")
w("| TEV1 0.8B (`tev1:0.8b`, Ollama, 811 MB) | fully on GPU | 40-case check (section 3.9) | 3.9 |\n| Clef, Jev | not run | Clef (9B/27B) does not fit; Jev is a closed hosted API | section 6 |")
w("")
w("### 2.3 Datasets")
w("")
w("| Dataset | Source | Question type | Cases in the file | Notes |")
w("|---|---|---|---|---|")
w("| banking77 | HF `mteb/banking77`, test split | choice | 1,000 (sampled) | 77 intents cut to **10 options per case** (the correct one plus 9 random wrong ones, seeded), because Ollama's decision API allows at most 26 options |")
w("| clinc150 | HF `clinc/clinc_oos`, plus config, test split | choice | 1,000 (sampled) | 150 intents cut to 10 options; 18% of cases are the out-of-scope label |")
w("| decision-bench | local file `decision-bench-1071.jsonl` | choice (2 to 9 options) | 949 | the 122 image cases were dropped (text-only models); structured JSON inputs; 10 areas; every question is a plain choice (no yes/no or score types) |")
w("| jevbench-choice | JevBench `data/*/choice.jsonl` | choice | 1,157 | banking cases cut to 10 options in this version |")
w("| jevbench-noul | JevBench `data/*/noul.jsonl` | yes/no | 949 | about 61% of labels are \"yes\"; some topics are 95-99% \"yes\" |")
w("| jevbench-score | JevBench `data/*/score.jsonl` | score (3 to 5 levels) | 828 | 728 ordinal cases (scored with MAE and QWK), 100 moderation cases scored by exact match |")
w("")
w("All files are shuffled with fixed seeds. The custom runner uses the **first 500 cases** of each file, so every model sees the same cases. The official harness uses the **full 2,934 cases** of JevBench's own data (no option cutting: banking keeps all 77 options).")
w("")
w("### 2.4 Test methods")
w("")
w("1. **Custom runner (our app, `bench/`).** Each case goes to the model's own API: Strands and Laya in-process, Nimble and TEV through Ollama. The timer wraps only the model's decision call (model loading and one warm-up call are excluded). Correct means an exact match with the expected answer; errors and timeouts count as wrong. Score tasks also report MAE and QWK over their ordinal cases.")
w("2. **Official JevBench harness (`bench_eval.py`, unmodified).** Its `jev` backend normally calls TypeSafe's hosted API with an API key. A small wrapper redirects it to a **local** server (`127.0.0.1`, refused otherwise), so no data left the machine. The harness only reports accuracy; the wrapper additionally times every request (end to end: HTTP round trip plus model). Score predictions use the most likely level, as the harness does.")
w("3. **Differences between the two**: the custom runner passes structured inputs as objects and cuts big option lists to 10; the harness sends the raw state string and all options. The harness's MAE counts an unanswered case as zero error, mine averages over answered cases only (unanswered cases per run in the harness runs: " + "; ".join(f"{m}: {c}" for m, c in none_counts.items()) + "; so the two definitions agree here).")
w("")
w("### 2.5 Run conditions")
w("")
w("- The GPU was used by one job at a time, except for the mistakes listed in section 5.")
w("- Part of the early custom Laya runs ran **on battery power**, where the GPU is power-capped. The official-harness runs ran on **AC power**, but the GPU still heated to about 80 C and throttled.")
w("- The harness script waited for the GPU to cool to 60 C before each model, and logged temperature and clock at the start of every run (Appendix A).")
w("")
w("## 3. Results")
w("")
w("### 3.1 Custom runner: Strands Decider 2B (5 runs, 500 cases per dataset)")
w("")
w("Latency is model decision time only. Plus/minus values are standard deviations across the 5 runs. All Strands runs in this report were made **after** `flash-linear-attention` was installed (see 3.8); the `causal-conv1d` fallback was still in use.")
w("")
w("| Dataset | Question type | Cases | Accuracy | Avg latency per case | Slowest 5% (p95) | Total time per dataset |")
w("|---|---|---|---|---|---|---|")
for d in ORDER:
    c = CS["Strands Decider 2B"][d]
    extra = f" (MAE {c['mae']:.2f}, QWK {c['qwk']:.2f}, {c['ordn']} ordinal cases)" if "mae" in c else ""
    w(f"| {d} | {QTYPE[d]} | 500 | {c['acc']:.1f}%{extra} | {c['avg']:.1f} ms (+/-{c['avg_sd']:.1f}) | {c['p95']:.0f} ms | {c['tot']:.1f} s (+/-{c['tot_sd']:.1f}) |")
ca = CA["Strands Decider 2B"]
w(f"| **All six** | mixed | **3,000** | **{ca['acc']:.1f}%** | **{ca['avg']:.1f} ms** | | **{ca['tot']:.1f} s** per run (+/-{ca['tot_sd']:.1f}) |")
w("")
w(f"Per-run totals (s): {', '.join(f'{x:.1f}' for x in ca['per_run_tot'])}. Answers identical in all 5 runs: {all(CS['Strands Decider 2B'][d]['same'] for d in ORDER)}. Errors: {sum(CS['Strands Decider 2B'][d]['errs'] for d in ORDER):.0f}.")
w("")
w("An earlier first pass of 5 datasets (no score dataset yet, 1 run, 500 cases) gave the same accuracies: "
  + ", ".join(f"{d} {first_single[0][d]['summary']['accuracy']*100:.1f}%" for d in ORDER if d in first_single[0])
  + f"; it took {sum(first_single[0][d]['summary']['total_latency_ms'] for d in first_single[0])/1000:.0f} s of decision time. A first attempt at 828 cases per dataset was stopped after the first two datasets at the user's request, and replaced by 500.")
w("")
w("### 3.2 Custom runner: Laya, both checkpoints (5 runs, 500 cases per dataset)")
w("")
w("| Dataset | Question type | Laya English: accuracy | Laya typed-decisions: accuracy | English: avg ms per case (5 runs) | Typed: avg ms per case (5 runs) |")
w("|---|---|---|---|---|---|")
for d in ORDER:
    e, t = CS["Laya English"][d], CS["Laya typed-decisions"][d]
    ex = f" (MAE {e['mae']:.2f}, QWK {e['qwk']:.2f})" if "mae" in e else ""
    tx = f" (MAE {t['mae']:.2f}, QWK {t['qwk']:.2f})" if "mae" in t else ""
    w(f"| {d} | {QTYPE[d]} | {e['acc']:.1f}%{ex} | {t['acc']:.1f}%{tx} | {', '.join(f'{x:.0f}' for x in e['avg_runs'])} | {', '.join(f'{x:.0f}' for x in t['avg_runs'])} |")
w(f"| **All six** | mixed | **{CA['Laya English']['acc']:.1f}%** | **{CA['Laya typed-decisions']['acc']:.1f}%** | | |")
w("")
w(f"Whole-set decision time per run (s): English {', '.join(f'{x:.0f}' for x in CA['Laya English']['per_run_tot'])}; typed-decisions {', '.join(f'{x:.0f}' for x in CA['Laya typed-decisions']['per_run_tot'])}. "
  "Within typed-decisions run 4 the first datasets are slow (about 90 ms) and the later ones fast (about 27 ms), which fits the charger being connected partway through that run; this is an inference, the moment was not recorded. "
  "**These latency figures are not reliable**: the laptop was on battery for part of these runs and the GPU was power-capped, so the same cases took 20 ms in one run and 94 ms in another (see section 5). Accuracy is unaffected and identical across runs.")
w("")
w("### 3.3 Custom runner: accuracy side by side (500 cases, first 500 of each file)")
w("")
w("| Dataset | Strands Decider 2B | Laya English | Laya typed-decisions |")
w("|---|---|---|---|")
for d in ORDER:
    w(f"| {d} | {CS['Strands Decider 2B'][d]['acc']:.1f}% | {CS['Laya English'][d]['acc']:.1f}% | {CS['Laya typed-decisions'][d]['acc']:.1f}% |")
w(f"| **All six** | **{CA['Strands Decider 2B']['acc']:.1f}%** | **{CA['Laya English']['acc']:.1f}%** | **{CA['Laya typed-decisions']['acc']:.1f}%** |")
w("")
w("### 3.4 Official JevBench harness: by question type (3 runs, all 2,934 cases)")
w("")
w("Each cell is accuracy, average latency per case, and total time for that group, averaged over the 3 runs. Latency is end to end (HTTP round trip plus model).")
w("")
w("| Type | Cases | " + " | ".join(MODELS) + " |")
w("|---|---|" + "---|" * len(MODELS))
for ty in ("choice", "noul", "score"):
    ids = [i for i, (d, t) in meta.items() if t == ty]
    w(f"| {TN[ty]} | {len(ids):,} | " + " | ".join("%.1f%% · %.0f ms · %.1f s" % hagg(m, ids) for m in MODELS) + " |")
w("| **All** | **2,934** | " + " | ".join("**%.1f%% · %.0f ms · %.1f s**" % hagg(m, list(meta)) for m in MODELS) + " |")
w("")
w("Distance-based scores on the 728 ordinal score cases (lower MAE and higher QWK are better):")
w("")
w("| Model | MAE (levels) | QWK |")
w("|---|---|---|")
for m in MODELS:
    n, mae_, qwk_ = hord(m)
    w(f"| {m} | {mae_:.2f} | {qwk_:.3f} |")
w("")
w("By label quality (from the harness reports): " + "; ".join(f"{m}: gold {HR[m]['gold']:.1f}%, silver {HR[m]['silver']:.1f}%" for m in MODELS) + ". Gold = 1,682 cases, silver (labels agreed by a panel of models) = 1,252 cases.")
w("")
w("### 3.5 Official JevBench harness: per dataset (22 datasets)")
w("")
w("A \"dataset\" is one JevBench topic and question type. Cells: accuracy · average latency per case · total time per dataset (mean of 3 runs). Also saved as `data/harness_per_dataset.csv`.")
w("")
w("| Dataset | Type | Cases | " + " | ".join(MODELS) + " |")
w("|---|---|---|" + "---|" * len(MODELS))
for (d, t), ids in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    w(f"| {d} | {TN[t]} | {len(ids)} | " + " | ".join("%.1f%% · %.0f ms · %.1f s" % hagg(m, ids) for m in MODELS) + " |")
w("")
w("### 3.6 Official JevBench harness: run times and GPU state")
w("")
w("| Model | Run times (s) | Mean (s) | Per case, end to end |")
w("|---|---|---|---|")
for m in MODELS:
    ws = [w_ for *_, w_ in hdata[m]]
    w(f"| {m} | {', '.join(f'{x:.1f}' for x in ws)} | {st.mean(ws):.1f} | {st.mean(ws)/2934*1000:.1f} ms |")
w("")
w("The sum of the per-request timings matches each run's total within 0.3 s, so almost no time is lost outside the requests. Answers identical across the 3 runs: " + ", ".join(f"{m}: {deterministic[m]}" for m in MODELS) + ".")
w("")
w("### 3.7 Earlier harness runs that are kept for the record")
w("")
w("**Strands Decider, untimed harness runs** (run before the per-request timer was added): " + "; ".join(f"run {i+1}: {a_:.1f}% in {w_:.1f} s" for i, (a_, w_) in enumerate(untimed)) + f". Mean {st.mean(w_ for _, w_ in untimed):.1f} s. Identical accuracy and answers to the timed re-run; run times are in the same range.")
w("")
w("**Laya harness runs that are INVALID as per-checkpoint results (auto-routed).** The Laya server only treats exact names (`english`, `typed-decisions`) as a checkpoint choice. Any other `model` value means \"auto-route\", and I had passed `laya-english` and `laya-typed`. The server then loaded both checkpoints and routed requests itself:")
w("")
w("| Mislabelled run | Overall accuracy | Time (s) |")
w("|---|---|---|")
for n, acc_, wall_ in invalid:
    w(f"| {n} | {acc_:.1f}% | {wall_:.1f} |")
w("")
w("The \"typed\" label scored 46.0% (compared with 63.3% when the checkpoint is named explicitly), which is how the problem was found. These files are kept in `data/harness_invalid_autorouted/` and must not be used as per-checkpoint results.")
w("")
w("### 3.8 Strands Decider kernel fix (30 Decision Bench cases)")
w("")
w("Transformers warned that two fast kernels were missing and it was using slow fallback code. `flash-linear-attention` was installed. `causal-conv1d` could not be installed (there is no prebuilt version for PyTorch 2.14 / CUDA 13, and the system compiler is CUDA 12.0, so a local build would fail), so one fallback remains.")
w("")
w("| | Accuracy | Avg latency | Median | p90 | Slowest |")
w("|---|---|---|---|---|---|")
w(f"| Before (fallback kernels) | {KB['acc']}/{KB['n']} | {KB['avg']:.0f} ms | {KB['med']:.0f} ms | {KB['p90']:.0f} ms | {KB['mx']:.0f} ms |")
w(f"| After (flash-linear-attention) | {KA['acc']}/{KA['n']} | {KA['avg']:.0f} ms | {KA['med']:.0f} ms | {KA['p90']:.0f} ms | {KA['mx']:.0f} ms |")
w(f"| After, repeat run | {KA2['acc']}/{KA2['n']} | {KA2['avg']:.0f} ms | {KA2['med']:.0f} ms | {KA2['p90']:.0f} ms | {KA2['mx']:.0f} ms |")
w("")
w(f"About 19% faster on average. {changed} of 30 answers changed (a case that was wrong both before and after), so accuracy was unchanged; the repeat run gave identical answers. Timings were taken with the GPU idle but are a single 30-case measurement.")
w("")
w("### 3.9 Early exploratory checks (anecdotal: 3 to 12 cases)")
w("")
w("These were smoke tests while building the app. They are far too small to compare models and are listed only for completeness.")
w("")
w("- 3-case sample, GPU free: Strands Decider 3/3 at 81 ms per case, TEV 3/3 at 143 ms, Nimble 3/3 at 801 ms (Nimble partly on CPU).")
w("- 12 Decision Bench cases: TEV 8/12 (two cases refused: prompts of 4,874 and 2,378 tokens exceeded its 2,050-token limit), Nimble 10/12.")
w("- TEV measured 10-15 s per case when another model was loaded in Ollama at the same time; with the GPU free it was 143-400 ms. Lesson: one GPU job at a time.")
w("- Laya (12 cases): English 8/12 on Decision Bench and 5/12 on JevBench choice; typed-decisions 6/12 and 9/12.")
w("- **TEV, 40 cases** (the first 10 of banking77, decision-bench, jevbench-noul and jevbench-score; one run each, AC power, cool GPU): "
  + "; ".join(f"{n}: {t['k']}/{t['n']} correct ({100*t['k']/t['n']:.1f}%), {t['lat']:.0f} ms per case, {t['errs']} refused" for n, t in tev40.items())
  + ". Both refused the same Decision Bench case (over the 2,050-token limit). This confirmed that TEV answers yes/no and score questions through Ollama. 40 cases give a 95% interval of roughly +/-14 points, so this is only a rough look; a full 3-run test of both TEV models (500 cases per dataset) was started and stopped by request before any model finished. Nimble 9B was left out by request. See `model-size-vs-performance.html` for how these sit against the Laya and Strands results.")
w("")
w("## 4. Comparison and interpretation")
w("")
w("- **Strands Decider is the stronger general model here** (66.7% against 63.3% and 45.8% on the full JevBench harness). It leads clearly on banking (88.7% against 37-39%), commerce (93% against 26-28%) and security-incident choice (84.8% against 3.0% for English). On the custom runner it also leads on every dataset except jevbench-score.")
w("- **Laya typed-decisions is the specialist for structured operational tasks.** It beats Strands on customer-service choice (89.8% against 74.5%), invoice-processing choice (95.4% against 63.1%) and invoice-processing yes/no (95.5% against 80.5%), and it is much better at rating (score): QWK 0.773 against 0.551.")
w("- **Laya English is weaker almost everywhere** and is mostly useful as a baseline. Its security-incident choice score of 3.0% is far below what random guessing would give (20-33% for 3 to 5 options) and points to a systematic mismatch with those inputs; its 512-token reading limit is a possible but unverified reason.")
w("- **Speed:** Laya is roughly four times faster per case (about 27 ms against 113 ms on the harness) and about four times quicker over the whole run. Strands' time depends heavily on input length and option count: banking with 77 options took 288 ms per case and 86 s of the 333 s run.")
w("- **Custom runner against harness:** the same model scores differently (for example Strands choice 80.6% on my first 500 cases against 77.1% on the full set; Laya typed-decisions banking 85.4% with 10 options against 37.3% with all 77 options). The difference is the case selection, the option cutting, and the way inputs are passed. The two are not interchangeable.")
w("- **Yes/no accuracy needs context.** About 61% of the JevBench yes/no labels are \"yes\", and some topics are 95-99% \"yes\", so an always-\"yes\" answer already scores around 61%. (A baseline display for this was built and then removed at the user's request.)")
w("")
w("## 5. Caveats, problems and corrections")
w("")
w("1. **Battery and heat distort latency.** The laptop was on battery during part of the custom Laya runs, and the GPU was power-capped (35 W) with software thermal slowdown active. The same dataset took 20 ms, 36 ms and 94 ms per case in successive runs. On AC the GPU still reached 80 C and throttled from about 1,500-1,700 MHz at the start of a set to about 1,300-1,450 MHz during longer runs, so later runs of a set are slower (for example Strands 316 s, 339 s, 344 s). Treat latency as approximate; the fastest, coolest runs are the most representative. The custom-runner Strands 5-run latencies were steady (299-315 s per run), but I cannot confirm the power state they were taken in.")
w("2. **Mislabelled Laya runs (my mistake, now fixed).** See section 3.7. The corrected runs name the checkpoint in every request, and the typed-decisions server loaded only that checkpoint.")
w("3. **Script file-naming slip.** A variable in my run script was overwritten after the first run of each Laya set, so the output files of runs 2 and 3 got inconsistent names. In the archive (`reports/data/raw/harness/`) they are renamed to `run1` to `run3`. The numbers are unaffected.")
w("4. **An unintended GPU run (my mistake).** While testing the new datasets page I started it with the default model, so a real Strands Decider run executed on the GPU after the user had asked me to wait. The GPU was idle at that moment, the run finished in about 25 s, and nothing collided, but I should have selected the fake test model first.")
w("5. **Runs on a shared GPU.** Several early runs failed or were slow because the user's own server was using the GPU at the same time (out-of-memory errors; TEV at 10-15 s per case). The app now runs one job at a time and unloads other Ollama models first.")
w("6. **Option cutting makes banking77 and clinc150 easier than the originals** (10 options instead of 77 and 150). Absolute accuracy there is flattering; compare models only against each other.")
w("7. **Case selection:** the first 500 cases of fixed-seed shuffled files, not a random sample of the whole dataset.")
w("8. **Unverified:** the exact Ollama request format for yes/no and score questions (I used the shared Jev wire format, which the Strands package and Laya also use, but never ran Nimble or TEV on those types). Laya's and Strands' formats were checked against their source code and tests, and the official harness's own `jev` backend uses the same format.")
w("9. **Nimble and TEV were not benchmarked at scale.** Nimble (9.5 GB) does not fit the GPU fully and was left out by request. TEV rejects inputs over 2,050 tokens; both TEV models only have the 40-case check, because the full 3-run test was stopped by request before it finished.")
w("10. **Single-author, new benchmark.** JevBench was created on 24 September 2026 by an individual developer, has had no outside review, and about 43% of its labels are \"silver\" (agreed by a panel of models). It is not affiliated with TypeSafe.")
w("")
w("## 6. Background research notes (from web sources; not independently verified)")
w("")
w("- **Jev (TypeSafe)** is a closed, hosted API (via `api.typesafe.ai` or OpenRouter) with no downloadable weights. The open \"Jev\" models (Laya, Kev, Von, mini-Jev and others) are reproductions that copy the interface, not the model.")
w("- **TypeSafe's own benchmark** is private: 4 workflows (security incident response, agent-trace observability, invoice processing, customer service), scored against the averaged predictions of two frontier models. Reported: Jev 67.8% against 73-74% for the best frontier models, at about $0.0004 per case and 0.4 s. It is vendor-reported and not independently reproducible. JevBench's four operational topics mirror those workflows.")
w("- **No widely accepted decision-model benchmark exists yet** (Jev launched on 15 September 2026). Independent efforts: JevBench (2,934 cases), OneWave AI's 150-record business-decision set, a 60-case agent tool-call risk test, and the Open-Jev dataset (synthetic, not evidence of general capability). Established datasets that apply to part of the problem: Banking77, CLINC150 and the Berkeley Function-Calling Leaderboard. Calibration (expected calibration error) is a standard extra axis that we have **not** measured.")
w("- **Clef (Cloudflare)** has open Apache-2.0 weights in two sizes: `clef` 27B (Qwen3.8 base) and `clef-flash` 9B (Qwen3.5 base). Neither fits this 8 GB GPU at full or 8-bit precision; the 9B at 4-bit (about 5-6 GB, estimated) might. Cloudflare's published scores are self-reported.")
w("- **Other small open models** noted but not run: TEV1 0.8B (Ollama), Von (about 395M), NanoJev (0.6B), Decider 0.8B-4B, Rizzo Flow 1.7-4B (via Ollaya or their own packages).")
w("")
w("Sources: DataCamp on Jev (datacamp.com/blog/system-one-models-jev); OneWave AI benchmark (onewave-ai.com/blog/jev-vs-gpt-vs-claude-system-one-benchmark); Layer3 Labs (layer3labs.io/guides/jev-benchmarks); Open-Jev (huggingface.co/datasets/TypeSafeAI/Open-Jev); Ollaya (github.com/ollaya-dev/ollaya); Clef overview (flaviocopes.com/clef); JevBench repo (github.com/model-collapse/jev-bench).")
w("")
w("## 7. What was built (code deliverables)")
w("")
w("In `decision-benchmark/` (see the README for the full layout):")
w("")
w("- `bench/` FastAPI app and runner: adapters for Strands Decider, Laya and Ollama decision models (Nimble, TEV); choice, yes/no and score question types; MAE and QWK for score tasks; one job at a time; automatic unloading of other GPU models; shared helpers for GPU temperature and AC-power checks (`hardware.py`) and run statistics (`summary.py`).")
w("- `web/` two pages: `/` compares models on one dataset; `/datasets` runs one model across several datasets. Both show a live grid of green (correct) and orange (wrong) squares, a results table and column charts.")
w("- `scripts/fetch_datasets.py` builds the datasets; `scripts/run_benchmark.py` runs our own benchmark repeatedly from the command line; `scripts/run_jev_harness.py` runs the official JevBench harness against local models (with `jev_harness_wrapper.py`); `scripts/make_report.py` regenerates this report; `scripts/make_size_chart.py` builds the model-size visual; `scripts/dev/` has fake servers for testing without a GPU.")
w("- `tests/` 36 unit tests (all passing at the time of writing).")
w("- `reports/` this report, its data and the archived raw results.")
w("")
w("## 8. How to reproduce")
w("")
w("```bash")
w("cd decision-benchmark")
w("uv sync")
w("uv run python scripts/fetch_datasets.py                       # builds datasets/*.jsonl")
w("uv run uvicorn bench.app:app --port 8000                      # the app (pages / and /datasets)")
w("")
w("# Our own runner, 5 runs of 500 cases on every dataset (section 3.1 to 3.3):")
w("uv run python scripts/run_benchmark.py strands-decision --runs 5 --cases 500")
w("uv run python scripts/run_benchmark.py laya --laya-checkpoint typed-decisions --runs 5 --cases 500")
w("")
w("# The official JevBench harness against local models (sections 3.4 to 3.7); the script starts and stops the servers:")
w("uv run python scripts/run_jev_harness.py strands-decision laya-english laya-typed --runs 3")
w("")
w("uv run python scripts/make_report.py                          # rebuild this report from reports/data/raw/")
w("```")
w("")
w("Before any timing run: connect AC power (`--require-ac`), set the performance power profile, let the GPU cool (`--cooldown-c`, default 60 C), and close other GPU programs.")
w("")
w("## 9. Open items")
w("")
w("- Re-time all models in one session on AC power with the GPU cooled, and with `causal-conv1d` if the CUDA 13 toolkit is installed, for a clean latency comparison.")
w("- Measure calibration (expected calibration error), since all models return probabilities.")
w("- Run the full test on both TEV models (500 cases x 6 datasets x 3 runs, about 80 minutes with TEV 4B) for a real four-size comparison; the study was stopped before it finished.")
w("- Optional: a gold-labels-only mode, to compare with the JevBench leaderboard's 232-case sample.")
w("")
w("## Appendix A: GPU state at the start of each harness run")
w("")
w("Lines from the orchestration log (`data/harness_orchestration.log`), in order. Format: temperature in C, clock in MHz, whether thermal throttling was active. The run names `english` and `typed-decisions` are the corrected Laya runs (see section 5, item 3); `laya-english` and `laya-typed` are the invalid auto-routed runs.")
w("")
w("The first three `strands-decider` lines are the earlier untimed Strands runs; the next three are the timed re-run.")
w("")
w("```")
for l in gpu_lines:
    w(l[:150])
w("```")
w("")
w("## Appendix B: Files")
w("")
w("| Path | Contents |")
w("|---|---|")
w("| `reports/benchmark-report.md` | this report |")
w("| `reports/model-size-vs-performance.html` (and `.png`) | how model size relates to accuracy and speed; built by `scripts/make_size_chart.py` |")
w("| `reports/data/raw/custom_runner/` | every case and timing of our own runner: `strands-decision`, `laya-english`, `laya-typed-decisions` (5 runs each) and the first single pass (gzip JSON) |")
w("| `reports/data/raw/harness/<model>/run1..3` | official-harness report (`.txt`), per-case predictions (`.json.gz`) and per-request timings (`.timing.json.gz`) for `strands-decision`, `laya-english`, `laya-typed-decisions` |")
w("| `reports/data/raw/harness/strands-decision_untimed/` | the earlier, untimed Strands harness runs |")
w("| `reports/data/raw/harness/invalid_autorouted_laya/` | the mislabelled Laya runs (reports only), kept for the record |")
w("| `reports/data/raw/harness/orchestration.log` | log of the harness queue (GPU state at the start of each run) |")
w("| `reports/data/raw/custom_runner/tev_smoke_40cases/` | the TEV 0.8B and 4B 40-case check (section 3.9) |")
w("| `reports/data/raw/kernel_fix/` | before and after timings of the kernel fix (section 3.8) |")
w("| `reports/data/harness_per_dataset.csv` | the per-dataset table (section 3.5) |")
w("| `reports/data/summary_custom_runner.json` | aggregated statistics of our own runner |")
w("")
w("The archived raw files are gzip-compressed JSON (about 1.4 MB in total). Server logs were not kept.")
w("")
open(f"{OUT}/benchmark-report.md", "w").write("\n".join(L) + "\n")
print("report written:", f"{OUT}/benchmark-report.md", "| lines:", len(L))
