# Benchmark results

1 run(s), up to 10 cases per dataset. Latency is the model's decision time only (loading and warm-up excluded). +/- is the standard deviation across runs.

## tev-0.8b

| Dataset | Cases | Accuracy | Avg latency per case | Slowest 5% (p95) | Total time per dataset |
|---|---|---|---|---|---|
| banking77 | 10 | 80.0% | 74.0 ms (+/-0.0) | 78 ms | 0.7 s (+/-0.0) |
| decision-bench | 10 | 50.0% | 142.0 ms (+/-0.0) | 282 ms | 1.4 s (+/-0.0) |
| jevbench-noul | 10 | 70.0% | 82.3 ms (+/-0.0) | 100 ms | 0.8 s (+/-0.0) |
| jevbench-score | 10 | 60.0% (MAE 1.00, QWK 0.21) | 87.4 ms (+/-0.0) | 105 ms | 0.9 s (+/-0.0) |
| **All** | **40** | **65.0%** | **96.4 ms** | | **3.9 s** per run (+/-0.0) |

Same answers in every run: True. Errors per run: 1.

## tev

| Dataset | Cases | Accuracy | Avg latency per case | Slowest 5% (p95) | Total time per dataset |
|---|---|---|---|---|---|
| banking77 | 10 | 90.0% | 187.2 ms (+/-0.0) | 197 ms | 1.9 s (+/-0.0) |
| decision-bench | 10 | 70.0% | 430.6 ms (+/-0.0) | 903 ms | 4.3 s (+/-0.0) |
| jevbench-noul | 10 | 80.0% | 232.0 ms (+/-0.0) | 308 ms | 2.3 s (+/-0.0) |
| jevbench-score | 10 | 90.0% (MAE 0.14, QWK 0.86) | 265.4 ms (+/-0.0) | 357 ms | 2.7 s (+/-0.0) |
| **All** | **40** | **82.5%** | **278.8 ms** | | **11.2 s** per run (+/-0.0) |

Same answers in every run: True. Errors per run: 1.
