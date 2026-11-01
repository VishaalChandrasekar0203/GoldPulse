# Stage-3 research report: forecast reliability and scenarios

**Date.** 2026-11-01  
**Sample.** International gold parity `gold_inr_10g`, 2010-01-04 → 2026-10-30.  
**Rolling-origin grid.** 56 cutoffs, 2012-01-03 → 2025-10-21, step 63 sessions, 126-session test window. Train-only scales: constant σ and empirical λ stop at origin `i − h` so `y[t] = log(close[t+h]/close[t])` never uses post-cutoff prices. EWMA variance at t uses daily returns through t only.  
**Not trained.** LSTM, Transformers. No point-forecast champion was changed.

**Headline.** Causal EWMA × √h is a better 1-day 80/90/95% interval than frozen train residual RMSE: mean 80% coverage 81.4% vs 87.2% (less overcover), |coverage error| 2.6pp vs 9.0pp, and a slightly better Winkler score. That is the only production change. 7d/30d/90d √h bands are wider than they are better; 90d √h is a poor variance-additivity assumption (empirical λ ≈ 0.44, not 0.50). No point model is promoted. Scenarios are identity and sample-elasticity what-ifs, labelled separately from forecasts. MCX and IBJA remain unavailable licensed series — parity is not substituted for them.

Production point champions remain naive (1d/30d/90d) and ARIMA (7d, 0.04% near-tie).

Artifacts: `reports/stage3_interval_validation.py`, `reports/output/stage3_interval_validation.json`, `reports/output/stage3_coverage_{1,7,30,90}d.png`, `reports/output/indian_market_data_sources.md`, `reports/output/latest_scenario.csv`. CLI: `gold scenario`.

---

## 1. Volatility and interval validation

Point forecast held at naive (ŷ = 0). Intervals are `close[t] × exp(± z σ̂)`. Methods:

| Method | σ̂ at origin t |
|---|---|
| `const_train` | Train-only std of completed h-day log returns |
| `ewma_sqrt_h` | Causal daily EWMA (λ=0.94) × √h_trading |
| `ewma_emp_h` | Same EWMA × h^λ̂ with λ̂ from train RMS only |
| `ewma_completed_h` | Causal EWMA of completed h-day returns (no √h) |

z-scores: 80% 1.282, 90% 1.645, 95% 1.960. λ ∈ {0.90, 0.94, 0.97} is a sensitivity sweep, not a production tuner.

### √h scaling

| Horizon | h trading | √h | Median empirical λ | √h overstates? |
|---|---|---|---|---|
| 1d | 1 | 1.00 | 0.50 (identity) | No |
| 7d | 5 | 2.24 | **0.42** | Yes |
| 30d | 21 | 4.58 | **0.44** | Yes |
| 90d | 63 | 7.94 | **0.44** | Yes |

Gold’s h-day return RMS grows slower than √h. Using √h at 30d/90d inflates width. The 7d gap is already material (0.42 vs 0.50).

### 80% intervals (mean across 56 windows)

| Horizon | Method | Coverage (target 80%) | \|err\| | Mean width (INR) | Median Winkler |
|---|---|---|---|---|---|
| 1d | const_train | 0.872 | 0.090 | 1,266 | 1,431 |
| 1d | **ewma_sqrt_h** | **0.814** | **0.026** | **1,200** | **1,407** |
| 7d | const_train | 0.841 | 0.099 | 2,469 | **2,792** |
| 7d | ewma_sqrt_h | 0.832 | 0.054 | 2,684 | 2,934 |
| 7d | ewma_emp_h | 0.779 | 0.050 | 2,353 | 2,823 |
| 7d | **ewma_completed_h** | 0.785 | **0.029** | 2,450 | 2,927 |
| 30d | const_train | 0.828 | 0.118 | **4,792** | **5,443** |
| 30d | ewma_sqrt_h | 0.824 | 0.087 | 5,506 | 5,750 |
| 90d | const_train | 0.751 | 0.200 | **7,471** | **9,039** |
| 90d | ewma_sqrt_h | 0.813 | 0.166 | 9,558 | 9,699 |
| 90d | ewma_completed_h | 0.693 | 0.153 | 8,587 | 9,719 |

89% of 1d EWMA windows are within 5 coverage points of 80%, versus 21% for const_train. Const_train **overcovers** on average because a pooled historical σ is too wide in quiet windows and still misses 2025–26 (last-window 1d coverage 0.64). EWMA tracks the local vol level.

### 90% and 95% (1d and 90d)

| Horizon | Level | const cov | EWMA×√h cov | const Winkler | EWMA Winkler |
|---|---|---|---|---|---|
| 1d | 90% | 0.929 | **0.896** | 1,838 | **1,757** |
| 1d | 95% | 0.958 | **0.935** | 2,081 | **2,024** |
| 90d | 90% | 0.837 | 0.887 | **10,730** | 11,690 |
| 90d | 95% | 0.888 | 0.926 | **12,255** | 13,199 |

EWMA×√h wins 1d at all three nominal levels on both calibration and Winkler. At 90d it buys coverage with a worse interval score — the √h width penalty dominates.

λ=0.90 is slightly better calibrated at 1d 80% (|err| 1.9pp vs 2.6pp) but worse on median Winkler. Production keeps RiskMetrics λ=0.94 rather than fitting decay on this grid.

**Pre-registered production gate.** Replace residual RMSE with EWMA×√h only if coverage error improves by ≥2pp **and** median Winkler is not >5% worse, at both 80% and 90%, and only for 1d/7d. **1d passes. 7d fails** (Winkler ratio 1.051). 30d/90d were never in scope for a swap.

Charts: `reports/output/stage3_coverage_1d.png` (and 7/30/90d).

---

## 2. Indian market data (MCX and IBJA)

Full assessment: [`reports/output/indian_market_data_sources.md`](indian_market_data_sources.md).

Neither series is in DuckDB. Parity is **not** used as a stand-in print.

| Source | Frequency | History | Licence / cost (indicative, ex-tax) | Redistribution |
|---|---|---|---|---|
| MCX EOD bhavcopy | Daily OHLC/OI | Up to 3 months excluding the last 12 months may be free; paid EOD listed | ₹2,00,000 / year (MCX domestic vendor list, 1 Jul 2024) | Agreement required; vendor redistribution needs prior MCX approval |
| MCX historical trade book | Tick | Minimum **one year**; HDD delivery | Gold-only ₹50,000 / year; all commodities ₹2,50,000 | Same |
| MCX historical trade+order | Tick + book | Minimum one year; HDD | Gold-only ₹1,00,000 / year; all commodities ₹5,00,000 | Same |
| Live MCX L1–L3 | Realtime | n/a | ₹15L–₹54L / year plus terminal fees; connectivity ₹1–1.5L | Vendor list |
| IBJA Rates API | AM ~12:30 / PM ~18:30 IST, weekdays | API is latest session; persist locally | Quote by email (nagaraj.iyer@ibja.in); min 1 year, one company | No resale/publish without written consent |
| IBJA historical | Daily AM/PM | Up to **20 years**, **.xls on request** | Quote | Retain internally after termination; no redistribute |

Contacts: MCX `datafeed@mcxindia.com`; IBJA-IGM benchmark https://ibja-igm.com/. Scraping exchange sites is disallowed.

GOLDBEES−parity (median gap about −3.4% as of 2026-10-30) remains an ETF proxy, not IBJA.

---

## 3. Scenario engine (not a forecast)

`gold scenario` and `gold.scenarios.engine`. Snapshot is as-of the last **gold** session so FX cannot leak a later calendar date (USDINR printed 2026-11-01 while COMEX last is 2026-10-30).

**As-of 2026-10-30**

| Input | Value |
|---|---|
| XAU/USD (GC=F) | 3,898.90 |
| USD/INR | 95.1524 |
| International parity | **119,275.95 INR/10g** (matches stored `gold_inr_10g`) |
| DXY | 101.96 |
| US 10y / TIPS 10y | 4.929% / 2.60% |
| Import duty (curated) | 6.0% |
| GOLDBEES vs parity | −3.39% |

Identity: `P = XAU × USDINR × (10/31.1034768)`, so `dP/P ≈ dXAU/XAU + dUSDINR/USDINR`.

Sample elasticities are **contemporaneous OLS** on daily returns (n≈4,230), not Diebold–Mariano-validated forecasts. Do not add a DXY row to an XAU identity shock (double-counts USD gold).

| Shock | Kind | Δ INR/10g | Δ% | Note |
|---|---|---|---|---|
| XAU/USD +1% | identity | +1,193 | +1.00% | USDINR held |
| USD/INR +1% | identity | +1,193 | +1.00% | XAU held |
| Both +1% | identity | +2,397 | +2.01% | Log-additive |
| DXY +1% | elasticity β=−0.88 | −1,047 | −0.88% | Not a forecast |
| US 10y +25 bp | elasticity β=−0.046 / pp | −1,365 | −1.14% | Contemporaneous |
| TIPS 10y +25 bp | elasticity β=−0.025 / pp | −746 | −0.63% | Contemporaneous |
| Duty +1 pp | onshore add-on | +1,229 | +0.94% of onshore | **Parity unchanged.** GST 3% config assumption |
| ETF gap +1 pp | onshore add-on | +1,193 | vs ETF, not IBJA | **Parity unchanged** |

Default CLI: `gold scenario --xau 1 --usdinr 1 --dxy 1 --yield-bp 25 --duty-pp 1 --premium-pp 1`. CSV: `reports/output/latest_scenario.csv`.

---

## 4. Research validation (point forecasts)

Promotion bar unchanged from Stage 2: reconstructed-price MAE >2% better than naive **and** Diebold–Mariano p<0.05 on absolute errors (HAC lags = horizon) on **at least three** cutoff windows, including one after 2025-10-28. Stage-2 Ridge/XGB/regime splits still fail that bar. This stage did not re-estimate those models.

Rolling-origin **naive** reconstructed MAE (mean across the 56 windows; INR levels were much lower in 2012–18, so these averages are not comparable to the 2026 holdout):

| Horizon | Mean naive MAE (INR/10g) | Directional accuracy |
|---|---|---|
| 1d | 359 | 0 by construction |
| 7d | 785 | 0 |
| 30d | 1,586 | 0 |
| 90d | 3,237 | 0 |

Latest chronological holdout (pipeline `v2-logret`, unchanged): naive 1d MAE 1,782 / MAPE 1.32%; 7d ARIMA 3,942 / 2.92% (1.66 INR edge); 30d 7,827 / 5.81%; 90d 14,522 / 10.81%.

Experiment history: training run `research-stage3-intervals` recorded in `training_runs` (interval scores only, `is_champion` untouched). Previous v2 experiments were not deleted.

---

## Limitations

1. Target is international parity, not MCX or IBJA.
2. Overlapping 30d/90d returns make coverage estimates dependent; 56 windows with step 63 reduce but do not remove that.
3. Intervals are centered on naive. A biased mean would still mis-center a well-scaled band.
4. Empirical λ is a ratio of RMS, not a GARCH term-structure model.
5. Scenario DXY/yield rows are in-sample elasticities. Duty/GST is an illustration, not a tax engine.
6. USDINR can print on days COMEX does not; the snapshot now stops at the last gold session.

---

## Production recommendations

1. **Point forecasts.** Keep naive at 1d/30d/90d and ARIMA at 7d. Do not promote trees, Ridge, regime splits, or sequence models.
2. **1d intervals.** Use causal EWMA × √h (λ=0.94) around the champion point. Config: `forecast.interval_method_by_horizon.1: ewma_sqrt_h`. Residual RMSE remains the default for other horizons.
3. **7d/30d/90d intervals.** Keep residual RMSE. √h is rejected at 7d by the Winkler gate and at 30d/90d by both Winkler and λ̂≈0.44. Optional research: a dedicated 90d scale (completed-h EWMA or empirical λ) — not production.
4. **Scenarios.** Ship `gold scenario` as a labelled overlay. Never write scenario rows into `forecasts` or mark them champion.
5. **MCX/IBJA.** Buy licensed history only if the product is onshore transacted price. Do not backfill that target with parity.
6. **Next model.** Still requires material, DM-significant OOS MAE versus naive on a post-2025-10-28 window **and** at least two earlier windows. The 2024–25 bull window remains banned as sole evidence.
