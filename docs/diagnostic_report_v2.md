# Diagnostic report v2 — log-return pipeline

Generated 2026-11-01 from local DuckDB/Parquet after implementing the v1 diagnostic recommendations. No observations were fabricated. LSTM/Transformers were not trained.

**Headline.** Switching the target from INR price levels to h-day log returns, reconstructing `price = close[t] × exp(ŷ)`, and scoring expanding-window regimes collapses the old 13× gap versus naive. Persistence remains the champion at 1d, 30d, and 90d. ARIMA is the 7d champion on a **1.66 INR (0.04%)** holdout edge plus 4/5 historical regimes. XGBoost and LightGBM still fail the promotion bar.

Promotion rule used: reconstructed-price MAE must beat naive on the latest 252-session holdout **and** on at least three of five expanding-window regimes. Naive persistence (zero log return) is always retained as the benchmark.

## What changed versus v1

1. **Target.** `y = log(close[t+h] / close[t])`. Live and holdout prices are reconstructed from the current close.
2. **Features.** Lagged closes and rolling min/mean/max/std of price were removed. Retained: return lags, realized vol, RSI/MACD, ETF gap/return, calendar, and first differences of macro series.
3. **ARIMA.** Fits daily log returns of the unshifted close, forecasts `h` trading days, and sums those steps. It no longer treats the already-shifted price target as a level series.
4. **Validation.** Expanding-window tests on 2014 taper, 2018 range, 2020 covid, 2022 hikes, and 2024–2026 bull, plus the latest chronological holdout.
5. **Real yields.** Treasury.gov 10-year par real yield curve: **4,210** daily prints, 2010-01-04 → 2026-10-30, **0%** missing on the feature matrix (FRED `DFII10` previously 100% empty).
6. **Naive.** Still zero log return ⇒ `ŷ_price = close[t]`.
7. **Tracking.** Experiments are no longer deleted. `pipeline_version=v2-logret` is stored next to `v1-price-level` rows. 24 new experiments vs 36 retained v1 rows.

LSTM and TFT remain untrained.

## Champions (v2)

| Horizon | Champion | Holdout MAE (INR/10g) | Holdout MAPE | Regimes beat |
|---|---|---|---|---|
| 1d | **naive** | 1,782 | 1.32% | — |
| 7d | **arima** | 3,942 | 2.92% | 4/5 |
| 30d | **naive** | 7,827 | 5.81% | — |
| 90d | **naive** | 14,522 | 10.81% | — |

The 7d ARIMA holdout MAE is 3,941.59 vs naive 3,943.26. The signed residual mean drops from +163 (naive) to +15, which is a small positive drift, not a structurally different forecast. Treat this promotion as a **near-tie that met the written rule**, not as evidence of a large skill edge.

## Latest holdout (252 sessions through 2026-10-30)

Reconstructed INR prices. `mae_vs_naive` < 1 means better than persistence.

| Horizon | naive | ARIMA | XGBoost | LightGBM | Ensemble |
|---|---|---|---|---|---|
| 1d | 1,782 | **1,779 (0.999×)** | 1,827 (1.03×) | 1,876 (1.05×) | 1,775 (0.996×) |
| 7d | 3,943 | **3,942 (1.000×)** | 4,394 (1.11×) | 4,515 (1.15×) | 4,018 (1.02×) |
| 30d | 7,827 | 7,827 (1.000×) | **7,111 (0.91×)** | 7,162 (0.92×) | **7,065 (0.90×)** |
| 90d | **14,522** | 14,541 (1.001×) | 17,423 (1.20×) | 18,036 (1.24×) | 16,331 (1.12×) |

Trees beat naive on the 30d holdout (the 2025–26 bull/pullback window) but only win **1 of 5** historical regimes (that same bull window). Ensemble 1d also beats holdout (1,775 vs 1,782) but only 2/5 regimes. Neither is promoted.

## Expanding-window regimes (train on all history before the test window)

ARIMA vs naive, reconstructed MAE ratio (values < 1 beat naive):

| Regime | 1d | 7d | 30d | 90d |
|---|---|---|---|---|
| 2014 taper | **0.996** | **0.999** | 1.000 | **0.999** |
| 2018 range | 1.002 | **0.999** | **0.993** | **0.988** |
| 2020 covid | 1.037 | 1.011 | **0.986** | **0.981** |
| 2022 hikes | 1.008 | **0.993** | **0.988** | **0.991** |
| 2024–26 bull | 1.008 | **0.991** | **0.987** | 1.011 |
| **Regimes beat** | 1/5 | **4/5** | 4/5* | 4/5* |

\*30d and 90d ARIMA meet the regime bar but **lose the latest holdout**, so they stay unpromoted.

XGBoost 1d beats naive only in 2018 and 2020 (2/5). At 30d it is 0.77× naive on the 2026 bull test and worse than naive in every earlier regime.

## Versus v1 price-level models

Naive MAE is unchanged at every horizon (same persistence definition after reconstruction). Non-naive 1d MAE vs the stored v1 run:

| Model | v1 MAE | v2 MAE | v2 / v1 |
|---|---|---|---|
| XGBoost 1d | 23,590 | 1,827 | 0.077 |
| LightGBM 1d | 20,181 | 1,876 | 0.093 |
| ARIMA 1d | 20,510 | 1,779 | 0.087 |
| Ensemble 1d | 18,085 | 1,775 | 0.098 |

The v1 failure mode (level models stuck near ~110k while the holdout traded 113–157k, residual ACF ≈ 0.97) is gone. 1d residual ACF is now −0.05 for naive/ARIMA and 0.13–0.19 for trees.

## Features and leakage

- 4,164 rows × 49 stored columns. Tree matrix uses 35 return/vol/macro-change/calendar columns; `close` is the reconstruction anchor only.
- No `close_lag_*`, `roll_mean_*`, `roll_min_*`, or `roll_max_*` columns remain.
- Real-yield *changes* are in the model matrix (`us_real_yield_10y_chg`).
- Still missing: India call money rate (FRED timeout; repo rate is the substitute), MCX futures, IBJA prints. India CPI is annual World Bank (16 points).
- XGBoost 1d importances are diffuse (top gain 0.05 on `is_trading_day`). At 30d/90d, `india_cpi_chg` and `gold_import_duty_chg` dominate — sparse step functions, not a daily signal.

## Forecasts from 2026-10-30 close (INR / 10g, international parity)

| Horizon | Model | Point | 10–90% residual interval |
|---|---|---|---|
| 1d | naive | 119,276 | 116,091 – 122,461 |
| 7d | arima | 119,390 | 112,790 – 125,991 |
| 30d | naive | 119,276 | 107,064 – 131,488 |
| 90d | naive | 119,276 | 96,886 – 141,666 |

Target remains the COMEX×USDINR parity proxy, not MCX or IBJA.

## Conclusions

1. The v1 underperformance was the non-stationary level target, not missing TIPS or MCX.
2. After the target fix, trees are competitive only on the most recent 30d holdout and do not generalize across regimes.
3. Corrected ARIMA is essentially naive plus a small estimated drift. That drift was enough to clear the written 7d promotion rule; it is not a large economic edge.
4. Real yields are now in the feature store and do not overturn persistence.
5. Do not introduce LSTM/Transformers until a model beats naive on the latest holdout **and** at least three regimes by a margin that survives a different cutoff.

Artifacts: `reports/output/diagnostics_v2.json`, experiment table in DuckDB (`pipeline_version=v2-logret`), training run `train_20261101T194101_e9e3bd27`.
