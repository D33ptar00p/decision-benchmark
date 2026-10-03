# decision-benchmark

Benchmark small **decision models** (Strands Decider, Laya, Nimble, TEV) on your own machine.
A decision model reads some input and picks an answer from a closed set (a choice), answers yes/no
with a probability, or rates on an ordered scale. This project measures **accuracy and latency**
for those models, with a small web UI and a script that runs the official
[JevBench](https://github.com/model-collapse/jev-bench) harness locally.

Tested on an NVIDIA RTX 3070 Laptop GPU (8 GB). A full write-up of the results, caveats and
mistakes is in [`reports/benchmark-report.md`](reports/benchmark-report.md).

## Requirements

- Linux with an NVIDIA GPU (a CPU run works but is slow); Python 3.12+; [`uv`](https://docs.astral.sh/uv/); `git`
- [Ollama](https://ollama.com) **0.35+** only if you want Nimble or TEV (they are served through Ollama's `/v1/systemone` API)
- Network access on first use (Hugging Face models and datasets, the JevBench repo)

## Quick start

```bash
uv sync
uv run python scripts/fetch_datasets.py        # builds datasets/*.jsonl (needs network)
uv run uvicorn bench.app:app --port 8000       # then open http://localhost:8000
```

- `/` **Compare models**: pick one dataset and one or more models. A grid of squares fills in live
  (green = correct, orange = wrong), followed by a table and column charts (accuracy, average
  latency per case, total latency).
- `/datasets` **Compare datasets**: pick one model and several datasets.

Models that are not installed are greyed out with the reason. Only one run happens at a time, so
latency is not distorted by two jobs sharing the GPU; other Ollama models are unloaded first.

## Models

| Id | Model | How it runs |
|---|---|---|
| `strands-decision` | Strands Decider 2B (`StrandsAgents/strands-decider-2B-hobson-v19`) | in-process, `strands-decider` package |
| `laya` | Laya 421M (`convaiinnovations/laya`; set `LAYA_CHECKPOINT=typed-decisions` for the typed checkpoint) | in-process, `laya` package |
| `nimble` | Nimble 9B (`nimble`) | Ollama |
| `tev` / `tev-0.8b` | TEV1 4B / 0.8B (`tev1:latest`, `tev1:0.8b`) | Ollama |

Override names with `NIMBLE_MODEL`, `TEV_MODEL`, `TEV_SMALL_MODEL`, `STRANDS_DECIDER_CHECKPOINT`,
`LAYA_CHECKPOINT` and `OLLAMA_HOST`. Nimble (9.5 GB) does not fully fit an 8 GB GPU.

To add a model: write an adapter with `warmup()`, `decide(case)` and optionally `close()` in
`bench/adapters/`, then register it in `bench/config.py` (`FACTORIES`, `LABELS`, and
`MAX_OPTIONS` if the model limits options per case).

## Datasets and question types

`scripts/fetch_datasets.py` writes one JSONL file per dataset into `datasets/`. Any extra `.jsonl`
file in that folder shows up in the UI.

| Dataset | Type | Notes |
|---|---|---|
| `banking77`, `clinc150` | choice | intent classification; options cut to **10 per case** (Ollama allows at most 26) |
| `decision-bench` | choice (2 to 9 options) | needs `decision-bench-1071.jsonl` in `datasets/source/` (or set `DECISION_BENCH_PATH`); text-only cases |
| `jevbench-choice`, `jevbench-noul`, `jevbench-score` | choice, yes/no, score | from JevBench; the script clones the repo (or set `JEVBENCH_DIR`) |
| `sample` | choice | 3 cases, for smoke tests |

Case format: `{"id", "query", "options", "expected"}` plus optional `instructions`,
`descriptions`, `category`, `kind` (`choice`, `noul`, `score`) and `metric` (`exact`, `ordinal`).
Scoring is exact match. Score datasets also report MAE and QWK over their ordinal cases.

## Our own benchmark from the command line

The same runner as the web app, for repeated, multi-dataset runs:

```bash
uv run python scripts/run_benchmark.py strands-decision --runs 5 --cases 500
uv run python scripts/run_benchmark.py laya --laya-checkpoint typed-decisions --datasets banking77,jevbench-score
uv run python scripts/run_benchmark.py --list        # model ids and datasets
```

Each run reloads the model, runs the first `--cases` cases of every dataset, and times only the
model's decision call. Output goes to `results/benchmark/<timestamp>/`: every case and timing
(`<model>.json.gz`) plus `summary.md` and `summary.json` (accuracy, average latency per case, p95,
total time per dataset, spread across runs).

## The official JevBench harness

```bash
# models the script starts and stops itself: strands-decision, laya-english, laya-typed
uv run python scripts/run_jev_harness.py strands-decision laya-typed --runs 3

# a /v1/systemone server you started yourself (local URLs only)
uv run python scripts/run_jev_harness.py mymodel \
    --server mymodel=http://127.0.0.1:9000/v1/systemone --request-model mymodel=my-model-id

uv run python scripts/run_jev_harness.py laya-english --runs 1 --limit 50 --cooldown-c 0   # quick check
uv run python scripts/run_jev_harness.py --help
```

It runs JevBench's own `bench_eval.py` unmodified. That script normally calls TypeSafe's hosted API;
`scripts/jev_harness_wrapper.py` points it at a **local** server instead and refuses any other URL,
so no data leaves the machine. Per model it waits for the GPU to cool, starts the server, runs the
harness `--runs` times with per-request timing, and stops the server. Output goes to
`results/jev-harness/<timestamp>/`: the harness report, predictions and timings for every run, plus
`summary.md` and `summary.json` (accuracy by question type and by dataset, average latency per case,
total time per dataset, run times and GPU state).

For `--server` models set `--request-model` carefully: Laya only treats the exact names `english`
and `typed-decisions` as a checkpoint choice; any other value means automatic routing.

## Getting trustworthy timings

- Use **AC power** (the GPU is power-capped on battery; `--require-ac` waits for the charger) and the performance power profile.
- Let the GPU **cool down** between models (`--cooldown-c`, default 60). Laptop GPUs throttle at about 80 C, which makes later runs slower.
- Close other GPU programs. Accuracy is deterministic; only timings vary.

## Results and report

[`reports/benchmark-report.md`](reports/benchmark-report.md) has every result, the caveats and the mistakes
found along the way. The raw per-case results and timings of all runs are archived (gzip) in
`reports/data/raw/`; `uv run python scripts/make_report.py` rebuilds the report from them.

**Model size vs speed and accuracy:** open [`reports/model-size-vs-performance.html`](reports/model-size-vs-performance.html)
in a browser (a PNG copy is next to it). Rebuild it with `uv run python scripts/make_size_chart.py`.

## Trying things without a GPU

```bash
uv run python scripts/dev/fake_ui_server.py --port 8020       # the web app with an extra fake model
python scripts/dev/fake_jev_server.py --port 8777              # a fake /v1/systemone server
uv run python scripts/run_jev_harness.py fake --server fake=http://127.0.0.1:8777/v1/systemone --runs 1
```

## Tests

```bash
uv run pytest        # 36 tests: scoring, runner, API, adapters (with stand-ins), both scripts
```

## Layout

```
bench/            app.py (FastAPI), runner, scoring, metrics (MAE, QWK), summary (run statistics),
                  hardware (GPU temperature, AC power), config
bench/adapters/   Strands Decider, Laya and Ollama decision-model adapters, shared question format
web/              the two pages (index.html, datasets.html) plus shared css/js
scripts/          fetch_datasets.py, run_benchmark.py, run_jev_harness.py (+ jev_harness_wrapper.py),
                  make_report.py, make_size_chart.py
scripts/dev/      fake servers for trying things without a GPU
datasets/         generated JSONL datasets; datasets/source/ holds the Decision Bench source file
reports/          benchmark-report.md; reports/data/ has the CSV, summary JSON and raw/ (archived results)
results/          output of run_benchmark.py / run_jev_harness.py (created on demand, not tracked)
tests/            unit tests
```

## Known limits

- Jev (TypeSafe) is a closed hosted API and cannot be run locally. Clef (Cloudflare) is open but too big for an 8 GB GPU at full precision.
- TEV rejects inputs over 2,050 tokens; Laya reads about 512 tokens (1,024 for the typed checkpoint).
- Option cutting makes `banking77` and `clinc150` easier than the originals. Compare models against each other, not against published numbers.
- JevBench is a new, single-author benchmark and is not affiliated with TypeSafe.
