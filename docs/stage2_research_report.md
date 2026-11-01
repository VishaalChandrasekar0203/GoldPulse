# Stage-2 research report: gold return predictability

**Date.** 2026-11-01  
**Sample.** International gold parity `gold_inr_10g`, 2010-01-04 → 2026-10-30, leak-free as-of features.  
**Rule.** No production promotion. A result counts as a *reproducible out-of-sample gain* only if reconstructed-price MAE beats naive by **more than 2%** *and* a Diebold–Mariano test on absolute price errors (HAC lags = horizon) is significant at 5% on **at least three of six** cutoff windows.  
**Not trained.** LSTM, Transformers, and other sequence models.

**Headline.** There are economically familiar *contemporaneous* links (USD gold vs DXY, INR gold vs USD/INR). They do **not** translate into reproducible forecasts of h-day INR gold returns. No linear, interaction, shallow-tree, or regime-split model meets the reproducible-gain bar. The only operationally useful finding is interval calibration: a causal EWMA volatility scale around a **naive** point forecast is better calibrated at 1d/7d/30d than a full-sample residual RMSE band. US real yields, once ingested, add no incremental OOS skill. GOLDBEES–parity gaps mean-revert but do not replace IBJA/MCX and do not survive the latest holdout as a return signal.

Production champions are unchanged: naive at 1d/30d/90d, ARIMA at 7d on a previously documented 0.04% near-tie.

## Design (leakage control)

- Target is `y = log(close[t+h]/close[t])`. Prices scored as `close[t]×exp(ŷ)`.
- Predictors are known at session close t (as-of joins, publication lags). Same-day gold–DXY correlations are labeled contemporaneous and are **not** used as if they were h-step forecasts.
- Naive benchmark is zero log return (price persistence).
- Cutoff windows train through year-end 2016/2018/2020/2022/2024 plus the current 252-session holdout (train through 2025-10-28).
- Diebold–Mariano uses Newey–West on MAE loss; Clark–West is reported for nested mean-zero vs linear models. Overlapping 30d/90d returns make unadjusted MAE comparisons too optimistic; DM is the gate.

Artifacts: `reports/output/stage2_research.json`, `reports/stage2_research.py`.

---

## 1. Predictable relationships?

### Contemporaneous (same-day, descriptive)

| Pair | Pearson | n |
|---|---|---|
| USD gold return vs DXY return | **−0.35** | 4,162 |
| INR gold return vs USD/INR return | **+0.41** | 4,162 |
| USD gold return vs USD/INR return | 0.00 | 4,162 |
| USD gold return vs 10y real-yield change | −0.11 | 4,162 |

INR gold *contains* USD/INR by construction, so the +0.41 with FX is partly mechanical. USD gold’s negative correlation with DXY is the textbook dollar-gold link. Real yields comove more weakly.

### Predictive (feature at t → h-day INR gold return)

HAC OLS on the 1d target (n = 4,163). Coefficients are for a one-unit change in the named series.

| Variable | Coef | HAC p |
|---|---|---|
| usd_inr_chg | −0.0023 | 0.034 |
| dxy_chg | +0.0012 | 0.022 |
| us_real_yield_10y_chg | +0.0009 | 0.87 |
| us_10y_yield_chg | −0.0192 | <0.001 |
| log_return (today) | −0.106 | <0.001 |
| ret_lag_5 | −0.042 | 0.035 |
| roll_vol_21 | +0.0015 | 0.65 |

Predictive correlations are small and decay with horizon:

| Predictor | 1d r | 7d r | 30d r | 90d r |
|---|---|---|---|---|
| usd_inr_chg | −0.10 | −0.04 | −0.02 | −0.01 |
| dxy_chg | +0.06 | +0.04 | +0.01 | +0.01 |
| us_real_yield_10y_chg | +0.01 | +0.01 | 0.00 | +0.01 |
| gold momentum (today) | −0.12 | −0.06 | −0.05 | −0.03 |
| etf_parity_gap_pct | +0.01 | −0.04 | −0.13 | −0.23 |

**Interpretation.** Short-horizon weak mean reversion in gold returns is detectable in-sample (negative coefficient on today’s return). Nominal yield *changes* beat TIPS real yields as a linear predictor. Real yields are now in the database (4,210 Treasury prints) but are not the missing ingredient. None of these relationships is large enough, on its own, to beat persistence out of sample (section 5).

Era-split 1d correlations flip sign: real-yield vs next-day gold is +0.20 in 2022 hikes and −0.16 in the 2026 bull window. A pooled linear model cannot be stable across regimes.

---

## 2. Interactions and nonlinearity

In-sample HAC OLS with products `mom×real_yield`, `mom×vol`, `inr×dxy`, `real_yield×vol`, `gap×festival` does **not** produce significant interaction coefficients at 1d. Remaining significant terms are the same main effects (DXY, nominal 10y change, short lags, duty change).

Latest-holdout reconstructed MAE vs naive:

| Model | 1d | 7d | 30d | 90d |
|---|---|---|---|---|
| Ridge, core | 1.004× | 0.995× | 1.010× | 1.038× |
| Ridge + interactions | 1.014× | 0.994× | 1.008× | 1.032× |
| Shallow XGB (depth 2, 80 trees) | 0.990× | 0.981× | **0.956×** | 1.214× |

The 30d shallow tree is a **4.4% MAE gain with DM p = 0.57**. That is a noisy single-window win, not a reproducible gain. Interactions never help enough to matter. There is no evidence that a deeper nonlinear sequence model is the next step: 1d naive residuals already have ACF ≈ −0.05.

---

## 3. Volatility and bull/bear splits

Causal labels: `vol_high` = `roll_vol_21` above the expanding 252-day median; `bull` = trailing 63-day log return > 0.

In-sample, high-vol periods show a slightly stronger 1d mean-reversion (momentum r = −0.14 vs −0.09) and a significant USD/INR coefficient that is absent in low-vol periods. Real-yield correlations stay near zero in both buckets.

Out-of-sample (2026 holdout), **separate Ridge models per regime do not beat naive** (MAE ratios 1.00–1.02, all DM p > 0.5). The vol split is degenerate in this window: 244 high-vol vs 8 low-vol origins, because 2025–26 volatility sits above the expanding historical median. A two-regime model cannot be identified where almost every test day is “high vol.”

---

## 4. Returns vs volatility for prediction intervals

Point forecast held at naive (ŷ = 0). 80% bands compared on the latest holdout.

| Horizon | Method | Coverage (target 80%) | Mean width (INR) | Winkler 80 (lower better) |
|---|---|---|---|---|
| 1d | Constant residual RMSE | 0.67 | 3,918 | 9,205 |
| 1d | Causal EWMA × √h | **0.83** | 6,140 | **8,718** |
| 1d | HAR on \|h-day return\| | 0.67 | 4,080 | 8,957 |
| 7d | Constant | 0.58 | 7,838 | 20,214 |
| 7d | Causal EWMA × √h | **0.83** | 13,811 | **19,298** |
| 30d | Constant | 0.47 | 15,143 | 40,752 |
| 30d | Causal EWMA × √h | **0.80** | 28,920 | 37,198 |
| 90d | Constant | 0.58 | 25,146 | **66,153** |
| 90d | Causal EWMA × √h | 0.83 | 51,790 | 69,129 |

Full-sample residual bands **undercover** in 2025–26 because realized vol rose. A leak-free EWMA of daily returns, scaled by √h, restores coverage near 80% and improves the 1d/7d/30d interval score. HAR using rolling-vol features does not. At 90d, √h scaling is a poor variance-additivity assumption (overlapping, mean-reverting returns); frozen train-end EWMA has the best 90d Winkler but is still a wide band.

This is an interval-construction result, not a point-forecast result. It does not justify promoting ARIMA/XGB.

---

## 5. Statistical significance vs naive

Reproducible-gain rollup (Ridge/XGB on core features, six cutoff windows):

| Horizon | Model | Windows with >2% MAE gain | Windows with gain **and** DM p<0.05 | Meets bar (3/6) |
|---|---|---|---|---|
| 1d | ridge_core | 0 | 0 | No |
| 1d | xgb_shallow_core | 0 | 0 | No |
| 7d | ridge_core | 1 | 1 | No |
| 30d | ridge_core | 1 | 1 | No |
| 90d | ridge_core | 5 | 1 | No |

Almost every “win” concentrates in the **2024 cutoff** (test year 2025, still inside the bull). Examples: 30d ridge 0.91× naive, DM p≈0; 90d ridge+domestic 0.68× naive, DM p≈0. The same specifications **lose** on the subsequent 2026 holdout (30d 1.01×, 90d 1.04–1.26×). That is the opposite of reproducibility.

90d MAE is lower than naive in five of six windows, but Diebold–Mariano with HAC lags = 63 is significant in only one. Overlapping 63-session returns inflate apparent MAE skill; the test is doing its job.

Clark–West tests on the latest holdout do not rescue any nested linear model.

The previously promoted 7d ARIMA (0.04% MAE edge) would not pass this research bar. It should stay documented as a near-tie, not as evidence of genuine predictability.

---

## 6. Sensitivity to cutoffs and windows

Ridge on `{usd_inr_chg, dxy_chg, real_yield_chg, nominal_yield_chg, return lags, roll_vol_21}`:

| Cutoff (test year) | 1d vs naive | 7d | 30d | 90d |
|---|---|---|---|---|
| 2016 → 2017 | 1.021 | 0.991 | 1.010 | 0.891† |
| 2018 → 2019 | 1.013 | 1.002 | 0.998 | 0.925† |
| 2020 → 2021 | 0.986 | 0.993 | 1.028 | 0.961† |
| 2022 → 2023 | 1.041* | 1.000 | 1.007 | 0.841† |
| 2024 → 2025 | 0.987 | **0.973*** | **0.912*** | **0.858*** |
| Holdout 2026 | 1.004 | 0.995 | 1.010 | 1.038 |

\* DM p<0.05 and material MAE gain.  
† Material MAE gain but DM not significant.

No specification is better than naive on both the 2024 window *and* the 2026 holdout at the same horizon. Any model tuned on 2024–25 will look skilled and then fail.

---

## 7. Indian premiums and domestic variables

**Unavailable (not fabricated):** MCX Gold Futures, IBJA rupee prints.

**Available proxy:** NSE GOLDBEES scaled to INR/10g vs international parity.

| Statistic | Value |
|---|---|
| Median ETF − parity gap | −1.7% |
| 10th / 90th percentile | −10.7% / +2.1% |
| 1d corr(gap, future parity return) | +0.01 |
| 90d corr(gap, future parity return) | −0.23 |
| 1d corr(gap, future gap change) | −0.13 |
| HAC p, gap → 1d parity return | 0.999 |
| HAC p, gap → next gap change | <0.001 (mean reversion) |

The gap **mean-reverts**. It does **not** forecast next-day parity returns. The 30d/90d negative correlations with future parity returns are large in-sample and explode in the 2024 test window when bundled with duty/festival features (90d MAE 0.68× naive), then reverse on the 2026 holdout (1.26× naive). GOLDBEES also embeds corporate actions; the gap is not a clean local-premium series.

Duty changes have a small in-sample 1d coefficient (p = 0.008) and do not produce OOS MAE gains. Festival-window flags are not significant at 5%. RBI repo changes are not significant.

**What would actually test Indian premia.** Licensed MCX and IBJA histories, or a documented dealer-premium series with a stable definition. Until then, domestic variables should stay as scenario/context features, not as a reason to promote a model.

---

## Limitations

1. Target is international parity, not the rupee price an Indian buyer pays (duty, GST, making charges, local premium).
2. 30d/90d evaluation uses overlapping returns; DM/HAC reduces but does not eliminate that dependence.
3. Expanding vol-regime labels collapse in 2025–26 (almost all test days “high vol”).
4. India CPI is annual (16 points). Call-money rates are empty (FRED timeout).
5. Shallow XGB is a nonlinearity probe, not a tuned production model. A larger tree would overfit more, not less.
6. Interval scores use a naive mean. Combining a biased mean with a good vol model can still mis-center bands.
7. Six cutoff windows are a sparse sensitivity grid, not a full walk-forward of every origin.

---

## Recommendations

1. **Do not promote** Ridge, interaction, regime-split, or shallow-tree models. None meet the reproducible OOS bar.
2. **Keep naive** as the 1d/30d/90d point forecast. Treat 7d ARIMA as optional documentation of a 0.04% near-tie, not as a research success.
3. **Do not introduce LSTM/Transformers.** Residual serial correlation at 1d is already near zero; the failure mode is unstable macro/premium relationships across regimes, which attention layers will not fix without a better target.
4. **Interval follow-up (not a champion change):** replace full-sample residual RMSE bands with causal EWMA × √h around the naive point for 1d and 7d, and re-check coverage on a new cutoff before shipping. Do not use HAR-on-absolute-return as implemented here.
5. **Do not treat TIPS real yields as a blocker.** They are populated and still insignificant next to nominal yield *changes* and gold’s own short lag.
6. **Do not treat GOLDBEES−parity as IBJA.** If local-premium forecasting is the product goal, acquire licensed MCX/IBJA data; otherwise keep parity labeled as parity.
7. **Any future model** must beat naive on a cutoff *after* 2025-10-28 **and** on at least two earlier windows under Diebold–Mariano, with relative MAE gain > 2%. The 2024–25 bull window is banned as the sole evidence.

No production experiment was marked champion from this study.
