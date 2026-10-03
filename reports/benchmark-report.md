# Decision-model benchmark report

**Strands Decider 2B and Laya (two checkpoints) on our benchmark app and on JevBench's official harness**

Date: 2026-10-03. Every table value below was computed from the archived raw results by `scripts/make_report.py`, not typed by hand. The raw per-case results and timings for every run are archived in `reports/data/raw/`.

## 1. Summary

- **On our own 500-case runner (6 datasets, 3,000 cases):** Strands Decider 78.4%, Laya typed-decisions 73.4%, Laya English 63.2%.
- **Best overall on the official JevBench harness: Strands Decider 2B at 66.7%** (all 2,934 cases), then Laya typed-decisions at 63.3% and Laya English at 45.8%.
- **Laya typed-decisions is the best at rating tasks (score):** 64.3% exact, QWK 0.773, MAE 0.38, against 50.5%, QWK 0.551, MAE 0.66 for Strands.
- **Laya is about 4x faster:** about 27 ms per case against 113 ms for Strands (end to end, via a local HTTP server); about 79 s against 333 s for the whole 2,934-case run.
- **Strands Decider is far ahead on choice tasks that look like classic classification** (banking, commerce, support), and Laya typed-decisions is far ahead on structured operational tasks (customer service, invoice processing).
- **Accuracy is deterministic:** every answer was identical across repeat runs of every model and setup. Only the timings vary between runs.
- **Latency numbers need care:** the laptop GPU throttles with heat, and part of the early runs happened on battery power. See section 5.
- **Corrections:** one set of Laya harness runs was mislabelled (automatic checkpoint routing) and has been re-run and quarantined. See section 5.

## 2. Setup

### 2.1 Hardware and software

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Laptop, 8 GB, driver 595.84 |
| RAM | 31 GB |
| OS / Python | Linux 7.0.0-34-generic, Python 3.12.3 |
| PyTorch / Transformers | 2.14.1 (CUDA 13.0 build) / 5.18.0 |
| Strands Decider package | `strands-decider` 0.1.0 (+ `flash-linear-attention` 0.5.2 added during the session; `causal-conv1d` not available) |
| Laya package | `laya` 0.3.24 |
| Ollama | 0.35.1 (decision-model API at `/v1/systemone`) |

### 2.2 Models

| Model | Size / format | How it was run | Used in |
|---|---|---|---|
| **Strands Decider 2B** (`StrandsAgents/strands-decider-2B-hobson-v19`) | Qwen3.5-2B-Base + LoRA adapter + readout head, bf16, about 4.5-5 GB on the GPU | in-process (custom runner); local server `strands-decider serve` (official harness) | all sections |
| **Laya English** (`convaiinnovations/laya`) | 421M encoder, fp32, reads about 512 tokens | in-process; local server `laya-serve` with request `model=english` | sections 3.2 to 3.5 |
| **Laya typed-decisions** (`typed-decisions` checkpoint) | 421M encoder, fp32, reads about 1,024 tokens | in-process; local server with request `model=typed-decisions` | sections 3.2 to 3.5 |
| Nimble 9B (`nimble:latest`, Ollama, Q8) | 9.5 GB, runs about 65% GPU / 35% CPU on this laptop | exploratory checks only (3 to 12 cases); left out of later testing by request | 3.9 |
| TEV1 4B (`tev1:latest`, Ollama, Q8) | 4.5 GB, fully on GPU, context limit 2,050 tokens | 40-case check (section 3.9) | 3.9 |
| TEV1 0.8B (`tev1:0.8b`, Ollama, 811 MB) | fully on GPU | 40-case check (section 3.9) | 3.9 |
| Clef, Jev | not run | Clef (9B/27B) does not fit; Jev is a closed hosted API | section 6 |

### 2.3 Datasets

| Dataset | Source | Question type | Cases in the file | Notes |
|---|---|---|---|---|
| banking77 | HF `mteb/banking77`, test split | choice | 1,000 (sampled) | 77 intents cut to **10 options per case** (the correct one plus 9 random wrong ones, seeded), because Ollama's decision API allows at most 26 options |
| clinc150 | HF `clinc/clinc_oos`, plus config, test split | choice | 1,000 (sampled) | 150 intents cut to 10 options; 18% of cases are the out-of-scope label |
| decision-bench | local file `decision-bench-1071.jsonl` | choice (2 to 9 options) | 949 | the 122 image cases were dropped (text-only models); structured JSON inputs; 10 areas; every question is a plain choice (no yes/no or score types) |
| jevbench-choice | JevBench `data/*/choice.jsonl` | choice | 1,157 | banking cases cut to 10 options in this version |
| jevbench-noul | JevBench `data/*/noul.jsonl` | yes/no | 949 | about 61% of labels are "yes"; some topics are 95-99% "yes" |
| jevbench-score | JevBench `data/*/score.jsonl` | score (3 to 5 levels) | 828 | 728 ordinal cases (scored with MAE and QWK), 100 moderation cases scored by exact match |

All files are shuffled with fixed seeds. The custom runner uses the **first 500 cases** of each file, so every model sees the same cases. The official harness uses the **full 2,934 cases** of JevBench's own data (no option cutting: banking keeps all 77 options).

### 2.4 Test methods

1. **Custom runner (our app, `bench/`).** Each case goes to the model's own API: Strands and Laya in-process, Nimble and TEV through Ollama. The timer wraps only the model's decision call (model loading and one warm-up call are excluded). Correct means an exact match with the expected answer; errors and timeouts count as wrong. Score tasks also report MAE and QWK over their ordinal cases.
2. **Official JevBench harness (`bench_eval.py`, unmodified).** Its `jev` backend normally calls TypeSafe's hosted API with an API key. A small wrapper redirects it to a **local** server (`127.0.0.1`, refused otherwise), so no data left the machine. The harness only reports accuracy; the wrapper additionally times every request (end to end: HTTP round trip plus model). Score predictions use the most likely level, as the harness does.
3. **Differences between the two**: the custom runner passes structured inputs as objects and cuts big option lists to 10; the harness sends the raw state string and all options. The harness's MAE counts an unanswered case as zero error, mine averages over answered cases only (unanswered cases per run in the harness runs: Strands Decider 2B: [0, 0, 0]; Laya English: [0, 0, 0]; Laya typed-decisions: [0, 0, 0]; so the two definitions agree here).

### 2.5 Run conditions

- The GPU was used by one job at a time, except for the mistakes listed in section 5.
- Part of the early custom Laya runs ran **on battery power**, where the GPU is power-capped. The official-harness runs ran on **AC power**, but the GPU still heated to about 80 C and throttled.
- The harness script waited for the GPU to cool to 60 C before each model, and logged temperature and clock at the start of every run (Appendix A).

## 3. Results

### 3.1 Custom runner: Strands Decider 2B (5 runs, 500 cases per dataset)

Latency is model decision time only. Plus/minus values are standard deviations across the 5 runs. All Strands runs in this report were made **after** `flash-linear-attention` was installed (see 3.8); the `causal-conv1d` fallback was still in use.

| Dataset | Question type | Cases | Accuracy | Avg latency per case | Slowest 5% (p95) | Total time per dataset |
|---|---|---|---|---|---|---|
| banking77 | choice, 10 options | 500 | 97.4% | 48.6 ms (+/-2.2) | 57 ms | 24.3 s (+/-1.1) |
| clinc150 | choice, 10 options | 500 | 98.8% | 45.2 ms (+/-1.6) | 48 ms | 22.6 s (+/-0.8) |
| decision-bench | choice, 2-9 options | 500 | 75.2% | 225.9 ms (+/-5.3) | 550 ms | 112.9 s (+/-2.7) |
| jevbench-choice | choice, 3-10 options | 500 | 80.6% | 81.6 ms (+/-1.5) | 163 ms | 40.8 s (+/-0.8) |
| jevbench-noul | yes/no | 500 | 70.6% | 94.0 ms (+/-1.6) | 160 ms | 47.0 s (+/-0.8) |
| jevbench-score | score, 3-5 levels | 500 | 47.6% (MAE 0.63, QWK 0.57, 435 ordinal cases) | 123.7 ms (+/-2.2) | 173 ms | 61.9 s (+/-1.1) |
| **All six** | mixed | **3,000** | **78.4%** | **103.2 ms** | | **309.5 s** per run (+/-6.0) |

Per-run totals (s): 299.4, 311.2, 312.9, 314.7, 309.4. Answers identical in all 5 runs: True. Errors: 0.

An earlier first pass of 5 datasets (no score dataset yet, 1 run, 500 cases) gave the same accuracies: banking77 97.4%, clinc150 98.8%, decision-bench 75.2%, jevbench-choice 80.6%, jevbench-noul 70.6%; it took 229 s of decision time. A first attempt at 828 cases per dataset was stopped after the first two datasets at the user's request, and replaced by 500.

### 3.2 Custom runner: Laya, both checkpoints (5 runs, 500 cases per dataset)

| Dataset | Question type | Laya English: accuracy | Laya typed-decisions: accuracy | English: avg ms per case (5 runs) | Typed: avg ms per case (5 runs) |
|---|---|---|---|---|---|
| banking77 | choice, 10 options | 84.0% | 85.4% | 17, 19, 20, 32, 31 | 36, 20, 94, 94, 18 |
| clinc150 | choice, 10 options | 92.8% | 93.0% | 16, 17, 19, 29, 31 | 37, 18, 90, 90, 16 |
| decision-bench | choice, 2-9 options | 55.2% | 60.6% | 35, 39, 50, 53, 53 | 62, 53, 191, 191, 45 |
| jevbench-choice | choice, 3-10 options | 54.4% | 72.0% | 23, 24, 39, 41, 39 | 27, 26, 108, 108, 23 |
| jevbench-noul | yes/no | 57.0% | 66.0% | 26, 24, 40, 40, 38 | 27, 54, 107, 57, 24 |
| jevbench-score | score, 3-5 levels | 35.6% (MAE 1.16, QWK 0.13) | 63.2% (MAE 0.38, QWK 0.76) | 35, 30, 39, 41, 43 | 31, 123, 122, 27, 30 |
| **All six** | mixed | **63.2%** | **73.4%** | | |

Whole-set decision time per run (s): English 76, 76, 103, 118, 118; typed-decisions 110, 146, 356, 283, 78. Within typed-decisions run 4 the first datasets are slow (about 90 ms) and the later ones fast (about 27 ms), which fits the charger being connected partway through that run; this is an inference, the moment was not recorded. **These latency figures are not reliable**: the laptop was on battery for part of these runs and the GPU was power-capped, so the same cases took 20 ms in one run and 94 ms in another (see section 5). Accuracy is unaffected and identical across runs.

### 3.3 Custom runner: accuracy side by side (500 cases, first 500 of each file)

| Dataset | Strands Decider 2B | Laya English | Laya typed-decisions |
|---|---|---|---|
| banking77 | 97.4% | 84.0% | 85.4% |
| clinc150 | 98.8% | 92.8% | 93.0% |
| decision-bench | 75.2% | 55.2% | 60.6% |
| jevbench-choice | 80.6% | 54.4% | 72.0% |
| jevbench-noul | 70.6% | 57.0% | 66.0% |
| jevbench-score | 47.6% | 35.6% | 63.2% |
| **All six** | **78.4%** | **63.2%** | **73.4%** |

### 3.4 Official JevBench harness: by question type (3 runs, all 2,934 cases)

Each cell is accuracy, average latency per case, and total time for that group, averaged over the 3 runs. Latency is end to end (HTTP round trip plus model).

| Type | Cases | Strands Decider 2B | Laya English | Laya typed-decisions |
|---|---|---|---|---|
| choice | 1,157 | 77.1% · 135 ms · 156.7 s | 42.8% · 27 ms · 31.0 s | 58.7% · 27 ms · 30.9 s |
| yes/no | 949 | 68.1% · 88 ms · 84.0 s | 57.0% · 25 ms · 23.3 s | 68.2% · 24 ms · 23.2 s |
| score | 828 | 50.5% · 111 ms · 92.2 s | 37.2% · 29 ms · 24.3 s | 64.3% · 30 ms · 24.5 s |
| **All** | **2,934** | **66.7% · 113 ms · 332.9 s** | **45.8% · 27 ms · 78.6 s** | **63.3% · 27 ms · 78.6 s** |

Distance-based scores on the 728 ordinal score cases (lower MAE and higher QWK are better):

| Model | MAE (levels) | QWK |
|---|---|---|
| Strands Decider 2B | 0.66 | 0.551 |
| Laya English | 1.14 | 0.165 |
| Laya typed-decisions | 0.38 | 0.773 |

By label quality (from the harness reports): Strands Decider 2B: gold 71.9%, silver 59.6%; Laya English: gold 52.7%, silver 36.5%; Laya typed-decisions: gold 62.7%, silver 64.2%. Gold = 1,682 cases, silver (labels agreed by a panel of models) = 1,252 cases.

### 3.5 Official JevBench harness: per dataset (22 datasets)

A "dataset" is one JevBench topic and question type. Cells: accuracy · average latency per case · total time per dataset (mean of 3 runs). Also saved as `data/harness_per_dataset.csv`.

| Dataset | Type | Cases | Strands Decider 2B | Laya English | Laya typed-decisions |
|---|---|---|---|---|---|
| agent_trace_observability | choice | 156 | 59.0% · 73 ms · 11.4 s | 46.8% · 24 ms · 3.8 s | 61.5% · 24 ms · 3.7 s |
| banking | choice | 300 | 88.7% · 288 ms · 86.4 s | 38.7% · 35 ms · 10.6 s | 37.3% · 35 ms · 10.5 s |
| commerce | choice | 100 | 93.0% · 55 ms · 5.5 s | 28.0% · 16 ms · 1.6 s | 26.0% · 16 ms · 1.6 s |
| customer_service | choice | 137 | 74.5% · 130 ms · 17.8 s | 32.1% · 34 ms · 4.7 s | 89.8% · 35 ms · 4.8 s |
| invoice_processing | choice | 65 | 63.1% · 143 ms · 9.3 s | 61.5% · 34 ms · 2.2 s | 95.4% · 34 ms · 2.2 s |
| org | choice | 100 | 54.0% · 55 ms · 5.5 s | 10.0% · 16 ms · 1.6 s | 24.0% · 16 ms · 1.6 s |
| reviews | choice | 100 | 60.0% · 48 ms · 4.8 s | 81.0% · 16 ms · 1.6 s | 60.0% · 16 ms · 1.6 s |
| security_incidents | choice | 99 | 84.8% · 113 ms · 11.2 s | 3.0% · 33 ms · 3.3 s | 76.8% · 33 ms · 3.3 s |
| support | choice | 100 | 100.0% · 49 ms · 4.9 s | 100.0% · 16 ms · 1.6 s | 100.0% · 16 ms · 1.6 s |
| agent_trace_observability | yes/no | 60 | 56.7% · 66 ms · 3.9 s | 18.3% · 22 ms · 1.3 s | 41.7% · 23 ms · 1.4 s |
| customer_service | yes/no | 96 | 77.1% · 121 ms · 11.6 s | 84.4% · 32 ms · 3.1 s | 55.2% · 32 ms · 3.1 s |
| invoice_processing | yes/no | 200 | 80.5% · 134 ms · 26.7 s | 38.0% · 33 ms · 6.6 s | 95.5% · 33 ms · 6.5 s |
| invoices | yes/no | 100 | 52.0% · 54 ms · 5.4 s | 48.0% · 17 ms · 1.7 s | 51.0% · 17 ms · 1.7 s |
| logic | yes/no | 100 | 37.0% · 50 ms · 5.0 s | 59.0% · 16 ms · 1.6 s | 59.0% · 16 ms · 1.6 s |
| observability | yes/no | 100 | 100.0% · 49 ms · 4.9 s | 100.0% · 16 ms · 1.6 s | 100.0% · 16 ms · 1.6 s |
| quant | yes/no | 100 | 69.0% · 50 ms · 5.0 s | 62.0% · 16 ms · 1.6 s | 65.0% · 16 ms · 1.6 s |
| security_incidents | yes/no | 193 | 61.7% · 111 ms · 21.4 s | 53.9% · 30 ms · 5.8 s | 53.4% · 29 ms · 5.7 s |
| agent_trace_observability | score | 168 | 42.9% · 79 ms · 13.3 s | 50.0% · 24 ms · 4.0 s | 69.6% · 24 ms · 4.1 s |
| customer_service | score | 184 | 60.9% · 129 ms · 23.8 s | 25.5% · 34 ms · 6.2 s | 66.8% · 34 ms · 6.3 s |
| invoice_processing | score | 184 | 48.4% · 148 ms · 27.3 s | 42.9% · 34 ms · 6.2 s | 61.4% · 34 ms · 6.2 s |
| moderation | score | 100 | 43.0% · 50 ms · 5.0 s | 70.0% · 16 ms · 1.6 s | 46.0% · 16 ms · 1.6 s |
| security_incidents | score | 192 | 53.1% · 119 ms · 22.8 s | 14.6% · 33 ms · 6.3 s | 69.3% · 33 ms · 6.4 s |

### 3.6 Official JevBench harness: run times and GPU state

| Model | Run times (s) | Mean (s) | Per case, end to end |
|---|---|---|---|
| Strands Decider 2B | 316.0, 338.9, 344.2 | 333.0 | 113.5 ms |
| Laya English | 75.2, 79.7, 81.4 | 78.8 | 26.8 ms |
| Laya typed-decisions | 74.5, 80.0, 81.7 | 78.7 | 26.8 ms |

The sum of the per-request timings matches each run's total within 0.3 s, so almost no time is lost outside the requests. Answers identical across the 3 runs: Strands Decider 2B: True, Laya English: True, Laya typed-decisions: True.

### 3.7 Earlier harness runs that are kept for the record

**Strands Decider, untimed harness runs** (run before the per-request timer was added): run 1: 66.7% in 306.9 s; run 2: 66.7% in 317.7 s; run 3: 66.7% in 368.2 s. Mean 330.9 s. Identical accuracy and answers to the timed re-run; run times are in the same range.

**Laya harness runs that are INVALID as per-checkpoint results (auto-routed).** The Laya server only treats exact names (`english`, `typed-decisions`) as a checkpoint choice. Any other `model` value means "auto-route", and I had passed `laya-english` and `laya-typed`. The server then loaded both checkpoints and routed requests itself:

| Mislabelled run | Overall accuracy | Time (s) |
|---|---|---|
| laya-english_run1 | 45.8% | 81.7 |
| laya-english_run2 | 45.8% | 86.0 |
| laya-english_run3 | 45.8% | 88.9 |
| laya-typed_run1 | 46.0% | 78.8 |

The "typed" label scored 46.0% (compared with 63.3% when the checkpoint is named explicitly), which is how the problem was found. These files are kept in `data/harness_invalid_autorouted/` and must not be used as per-checkpoint results.

### 3.8 Strands Decider kernel fix (30 Decision Bench cases)

Transformers warned that two fast kernels were missing and it was using slow fallback code. `flash-linear-attention` was installed. `causal-conv1d` could not be installed (there is no prebuilt version for PyTorch 2.14 / CUDA 13, and the system compiler is CUDA 12.0, so a local build would fail), so one fallback remains.

| | Accuracy | Avg latency | Median | p90 | Slowest |
|---|---|---|---|---|---|
| Before (fallback kernels) | 22/30 | 340 ms | 220 ms | 976 ms | 1154 ms |
| After (flash-linear-attention) | 22/30 | 274 ms | 178 ms | 782 ms | 928 ms |
| After, repeat run | 22/30 | 276 ms | 178 ms | 777 ms | 926 ms |

About 19% faster on average. 1 of 30 answers changed (a case that was wrong both before and after), so accuracy was unchanged; the repeat run gave identical answers. Timings were taken with the GPU idle but are a single 30-case measurement.

### 3.9 Early exploratory checks (anecdotal: 3 to 12 cases)

These were smoke tests while building the app. They are far too small to compare models and are listed only for completeness.

- 3-case sample, GPU free: Strands Decider 3/3 at 81 ms per case, TEV 3/3 at 143 ms, Nimble 3/3 at 801 ms (Nimble partly on CPU).
- 12 Decision Bench cases: TEV 8/12 (two cases refused: prompts of 4,874 and 2,378 tokens exceeded its 2,050-token limit), Nimble 10/12.
- TEV measured 10-15 s per case when another model was loaded in Ollama at the same time; with the GPU free it was 143-400 ms. Lesson: one GPU job at a time.
- Laya (12 cases): English 8/12 on Decision Bench and 5/12 on JevBench choice; typed-decisions 6/12 and 9/12.
- **TEV, 40 cases** (the first 10 of banking77, decision-bench, jevbench-noul and jevbench-score; one run each, AC power, cool GPU): TEV1 0.8B: 26/40 correct (65.0%), 96 ms per case, 1 refused; TEV1 4B: 33/40 correct (82.5%), 279 ms per case, 1 refused. Both refused the same Decision Bench case (over the 2,050-token limit). This confirmed that TEV answers yes/no and score questions through Ollama. 40 cases give a 95% interval of roughly +/-14 points, so this is only a rough look; a full 3-run test of both TEV models (500 cases per dataset) was started and stopped by request before any model finished. Nimble 9B was left out by request. See `model-size-vs-performance.html` for how these sit against the Laya and Strands results.

## 4. Comparison and interpretation

- **Strands Decider is the stronger general model here** (66.7% against 63.3% and 45.8% on the full JevBench harness). It leads clearly on banking (88.7% against 37-39%), commerce (93% against 26-28%) and security-incident choice (84.8% against 3.0% for English). On the custom runner it also leads on every dataset except jevbench-score.
- **Laya typed-decisions is the specialist for structured operational tasks.** It beats Strands on customer-service choice (89.8% against 74.5%), invoice-processing choice (95.4% against 63.1%) and invoice-processing yes/no (95.5% against 80.5%), and it is much better at rating (score): QWK 0.773 against 0.551.
- **Laya English is weaker almost everywhere** and is mostly useful as a baseline. Its security-incident choice score of 3.0% is far below what random guessing would give (20-33% for 3 to 5 options) and points to a systematic mismatch with those inputs; its 512-token reading limit is a possible but unverified reason.
- **Speed:** Laya is roughly four times faster per case (about 27 ms against 113 ms on the harness) and about four times quicker over the whole run. Strands' time depends heavily on input length and option count: banking with 77 options took 288 ms per case and 86 s of the 333 s run.
- **Custom runner against harness:** the same model scores differently (for example Strands choice 80.6% on my first 500 cases against 77.1% on the full set; Laya typed-decisions banking 85.4% with 10 options against 37.3% with all 77 options). The difference is the case selection, the option cutting, and the way inputs are passed. The two are not interchangeable.
- **Yes/no accuracy needs context.** About 61% of the JevBench yes/no labels are "yes", and some topics are 95-99% "yes", so an always-"yes" answer already scores around 61%. (A baseline display for this was built and then removed at the user's request.)

## 5. Caveats, problems and corrections

1. **Battery and heat distort latency.** The laptop was on battery during part of the custom Laya runs, and the GPU was power-capped (35 W) with software thermal slowdown active. The same dataset took 20 ms, 36 ms and 94 ms per case in successive runs. On AC the GPU still reached 80 C and throttled from about 1,500-1,700 MHz at the start of a set to about 1,300-1,450 MHz during longer runs, so later runs of a set are slower (for example Strands 316 s, 339 s, 344 s). Treat latency as approximate; the fastest, coolest runs are the most representative. The custom-runner Strands 5-run latencies were steady (299-315 s per run), but I cannot confirm the power state they were taken in.
2. **Mislabelled Laya runs (my mistake, now fixed).** See section 3.7. The corrected runs name the checkpoint in every request, and the typed-decisions server loaded only that checkpoint.
3. **Script file-naming slip.** A variable in my run script was overwritten after the first run of each Laya set, so the output files of runs 2 and 3 got inconsistent names. In the archive (`reports/data/raw/harness/`) they are renamed to `run1` to `run3`. The numbers are unaffected.
4. **An unintended GPU run (my mistake).** While testing the new datasets page I started it with the default model, so a real Strands Decider run executed on the GPU after the user had asked me to wait. The GPU was idle at that moment, the run finished in about 25 s, and nothing collided, but I should have selected the fake test model first.
5. **Runs on a shared GPU.** Several early runs failed or were slow because the user's own server was using the GPU at the same time (out-of-memory errors; TEV at 10-15 s per case). The app now runs one job at a time and unloads other Ollama models first.
6. **Option cutting makes banking77 and clinc150 easier than the originals** (10 options instead of 77 and 150). Absolute accuracy there is flattering; compare models only against each other.
7. **Case selection:** the first 500 cases of fixed-seed shuffled files, not a random sample of the whole dataset.
8. **Unverified:** the exact Ollama request format for yes/no and score questions (I used the shared Jev wire format, which the Strands package and Laya also use, but never ran Nimble or TEV on those types). Laya's and Strands' formats were checked against their source code and tests, and the official harness's own `jev` backend uses the same format.
9. **Nimble and TEV were not benchmarked at scale.** Nimble (9.5 GB) does not fit the GPU fully and was left out by request. TEV rejects inputs over 2,050 tokens; both TEV models only have the 40-case check, because the full 3-run test was stopped by request before it finished.
10. **Single-author, new benchmark.** JevBench was created on 24 September 2026 by an individual developer, has had no outside review, and about 43% of its labels are "silver" (agreed by a panel of models). It is not affiliated with TypeSafe.

## 6. Background research notes (from web sources; not independently verified)

- **Jev (TypeSafe)** is a closed, hosted API (via `api.typesafe.ai` or OpenRouter) with no downloadable weights. The open "Jev" models (Laya, Kev, Von, mini-Jev and others) are reproductions that copy the interface, not the model.
- **TypeSafe's own benchmark** is private: 4 workflows (security incident response, agent-trace observability, invoice processing, customer service), scored against the averaged predictions of two frontier models. Reported: Jev 67.8% against 73-74% for the best frontier models, at about $0.0004 per case and 0.4 s. It is vendor-reported and not independently reproducible. JevBench's four operational topics mirror those workflows.
- **No widely accepted decision-model benchmark exists yet** (Jev launched on 15 September 2026). Independent efforts: JevBench (2,934 cases), OneWave AI's 150-record business-decision set, a 60-case agent tool-call risk test, and the Open-Jev dataset (synthetic, not evidence of general capability). Established datasets that apply to part of the problem: Banking77, CLINC150 and the Berkeley Function-Calling Leaderboard. Calibration (expected calibration error) is a standard extra axis that we have **not** measured.
- **Clef (Cloudflare)** has open Apache-2.0 weights in two sizes: `clef` 27B (Qwen3.8 base) and `clef-flash` 9B (Qwen3.5 base). Neither fits this 8 GB GPU at full or 8-bit precision; the 9B at 4-bit (about 5-6 GB, estimated) might. Cloudflare's published scores are self-reported.
- **Other small open models** noted but not run: TEV1 0.8B (Ollama), Von (about 395M), NanoJev (0.6B), Decider 0.8B-4B, Rizzo Flow 1.7-4B (via Ollaya or their own packages).

Sources: DataCamp on Jev (datacamp.com/blog/system-one-models-jev); OneWave AI benchmark (onewave-ai.com/blog/jev-vs-gpt-vs-claude-system-one-benchmark); Layer3 Labs (layer3labs.io/guides/jev-benchmarks); Open-Jev (huggingface.co/datasets/TypeSafeAI/Open-Jev); Ollaya (github.com/ollaya-dev/ollaya); Clef overview (flaviocopes.com/clef); JevBench repo (github.com/model-collapse/jev-bench).

## 7. What was built (code deliverables)

In `decision-benchmark/` (see the README for the full layout):

- `bench/` FastAPI app and runner: adapters for Strands Decider, Laya and Ollama decision models (Nimble, TEV); choice, yes/no and score question types; MAE and QWK for score tasks; one job at a time; automatic unloading of other GPU models; shared helpers for GPU temperature and AC-power checks (`hardware.py`) and run statistics (`summary.py`).
- `web/` two pages: `/` compares models on one dataset; `/datasets` runs one model across several datasets. Both show a live grid of green (correct) and orange (wrong) squares, a results table and column charts.
- `scripts/fetch_datasets.py` builds the datasets; `scripts/run_benchmark.py` runs our own benchmark repeatedly from the command line; `scripts/run_jev_harness.py` runs the official JevBench harness against local models (with `jev_harness_wrapper.py`); `scripts/make_report.py` regenerates this report; `scripts/make_size_chart.py` builds the model-size visual; `scripts/dev/` has fake servers for testing without a GPU.
- `tests/` 36 unit tests (all passing at the time of writing).
- `reports/` this report, its data and the archived raw results.

## 8. How to reproduce

```bash
cd decision-benchmark
uv sync
uv run python scripts/fetch_datasets.py                       # builds datasets/*.jsonl
uv run uvicorn bench.app:app --port 8000                      # the app (pages / and /datasets)

# Our own runner, 5 runs of 500 cases on every dataset (section 3.1 to 3.3):
uv run python scripts/run_benchmark.py strands-decision --runs 5 --cases 500
uv run python scripts/run_benchmark.py laya --laya-checkpoint typed-decisions --runs 5 --cases 500

# The official JevBench harness against local models (sections 3.4 to 3.7); the script starts and stops the servers:
uv run python scripts/run_jev_harness.py strands-decision laya-english laya-typed --runs 3

uv run python scripts/make_report.py                          # rebuild this report from reports/data/raw/
```

Before any timing run: connect AC power (`--require-ac`), set the performance power profile, let the GPU cool (`--cooldown-c`, default 60 C), and close other GPU programs.

## 9. Open items

- Re-time all models in one session on AC power with the GPU cooled, and with `causal-conv1d` if the CUDA 13 toolkit is installed, for a clean latency comparison.
- Measure calibration (expected calibration error), since all models return probabilities.
- Run the full test on both TEV models (500 cases x 6 datasets x 3 runs, about 80 minutes with TEV 4B) for a real four-size comparison; the study was stopped before it finished.
- Optional: a gold-labels-only mode, to compare with the JevBench leaderboard's 232-case sample.

## Appendix A: GPU state at the start of each harness run

Lines from the orchestration log (`data/harness_orchestration.log`), in order. Format: temperature in C, clock in MHz, whether thermal throttling was active. The run names `english` and `typed-decisions` are the corrected Laya runs (see section 5, item 3); `laya-english` and `laya-typed` are the invalid auto-routed runs.

The first three `strands-decider` lines are the earlier untimed Strands runs; the next three are the timed re-run.

```
== strands-decider run 1 start | gpu(temp,clock,throttle): 67,1500MHz,NotActive
== strands-decider run 2 start | gpu(temp,clock,throttle): 80,1335MHz,Active
== strands-decider run 3 start | gpu(temp,clock,throttle): 80,1410MHz,Active
== strands-decider run 1 start | gpu(temp,clock,throttle): 68,1500MHz,NotActive
== strands-decider run 2 start | gpu(temp,clock,throttle): 80,1335MHz,Active
== strands-decider run 3 start | gpu(temp,clock,throttle): 79,1305MHz,Active
== laya-english run 1 start | gpu(temp,clock,throttle): 67,1695MHz,NotActive
== laya-english run 2 start | gpu(temp,clock,throttle): 80,1545MHz,Active
== laya-english run 3 start | gpu(temp,clock,throttle): 79,1395MHz,Active
== laya-typed run 1 start | gpu(temp,clock,throttle): 66,1650MHz,NotActive
== laya-typed run 2 start | gpu(temp,clock,throttle): 79,1455MHz,Active
== laya-en-explicit run 1 start | gpu(temp,clock,throttle): 66,1665MHz,NotActive
== english run 2 start | gpu(temp,clock,throttle): 80,1590MHz,Active
== english run 3 start | gpu(temp,clock,throttle): 80,1650MHz,Active
== laya-td-explicit run 1 start | gpu(temp,clock,throttle): 66,1680MHz,NotActive
== typed-decisions run 2 start | gpu(temp,clock,throttle): 80,1665MHz,Active
== typed-decisions run 3 start | gpu(temp,clock,throttle): 80,1455MHz,Active
```

## Appendix B: Files

| Path | Contents |
|---|---|
| `reports/benchmark-report.md` | this report |
| `reports/model-size-vs-performance.html` (and `.png`) | how model size relates to accuracy and speed; built by `scripts/make_size_chart.py` |
| `reports/data/raw/custom_runner/` | every case and timing of our own runner: `strands-decision`, `laya-english`, `laya-typed-decisions` (5 runs each) and the first single pass (gzip JSON) |
| `reports/data/raw/harness/<model>/run1..3` | official-harness report (`.txt`), per-case predictions (`.json.gz`) and per-request timings (`.timing.json.gz`) for `strands-decision`, `laya-english`, `laya-typed-decisions` |
| `reports/data/raw/harness/strands-decision_untimed/` | the earlier, untimed Strands harness runs |
| `reports/data/raw/harness/invalid_autorouted_laya/` | the mislabelled Laya runs (reports only), kept for the record |
| `reports/data/raw/harness/orchestration.log` | log of the harness queue (GPU state at the start of each run) |
| `reports/data/raw/custom_runner/tev_smoke_40cases/` | the TEV 0.8B and 4B 40-case check (section 3.9) |
| `reports/data/raw/kernel_fix/` | before and after timings of the kernel fix (section 3.8) |
| `reports/data/harness_per_dataset.csv` | the per-dataset table (section 3.5) |
| `reports/data/summary_custom_runner.json` | aggregated statistics of our own runner |

The archived raw files are gzip-compressed JSON (about 1.4 MB in total). Server logs were not kept.

