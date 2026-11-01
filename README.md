# GoldPulse

**v1.0.0** — local gold intelligence for Indian INR / 10g parity.

GoldPulse downloads public market data, stores it on your machine, and produces 1-day, 7-day, 30-day, and 90-day forecasts of **international gold converted to rupees per 10 grams**. You can use it from the terminal, an optional local browser dashboard, or a scheduled daily job.

It does **not** scrape MCX or IBJA. Those prints are licensed and are left empty on purpose. It is **not** trading advice.

Repository: [github.com/VishaalChandrasekar0203/GoldPulse](https://github.com/VishaalChandrasekar0203/GoldPulse)

---

## What you get

| You want | Command |
|---|---|
| Set up once | `gold init` then `gold data update` then `gold train --all` then `gold forecast --horizon all` |
| Daily numbers in the terminal | `gold brief` |
| Local web dashboard | `gold ui` |
| Hands-off daily refresh | `gold run` |
| What-if INR impact | `gold scenario --pack` |

The modelled series is:

```
gold_inr_10g  =  COMEX gold (USD/oz)  ×  USD/INR  ×  (10 / 31.1034768)
```

That is **international parity**. It excludes Indian import duty, GST, making charges, and local premium. GOLDBEES is stored as an onshore ETF proxy, labelled separately, and is **not** IBJA.

---

## Requirements

- macOS or Linux
- Python **3.11 or newer**
- Internet for the first data download (Yahoo Finance, US Treasury, World Bank)
- About 1 GB of disk after history is stored locally

Windows works if you use the same commands in Git Bash or PowerShell, using `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

---

## Install (first time)

Do **not** put `# comments` on the same line as a command. The CLI will treat them as extra arguments.

```bash
git clone https://github.com/VishaalChandrasekar0203/GoldPulse.git
cd GoldPulse

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e ".[dev,ui]"
```

`[ui]` installs Streamlit for the local browser app. Skip it if you only want the terminal:

```bash
pip install -e ".[dev]"
```

---

## First run (about 5–15 minutes)

From the project folder, with the virtualenv active:

```bash
gold init
gold data update
gold features build
gold train --all
gold forecast --horizon all
gold brief
```

What each step does:

1. **`gold init`** — creates `data/gold.duckdb` and the series catalog. MCX and IBJA are marked unavailable.
2. **`gold data update`** — downloads public history (gold, USD/INR, DXY, yields, WTI, GOLDBEES, CPI, policy tables).
3. **`gold features build`** — leak-free feature matrix. No future prints are joined.
4. **`gold train --all`** — walk-forward training. A model is promoted only if it beats naive on the latest holdout **and** on at least 3 of 5 historical regimes.
5. **`gold forecast --horizon all`** — writes 1d / 7d / 30d / 90d points and intervals into DuckDB.
6. **`gold brief`** — prints the daily summary: parity, forecasts, 1d EWMA band, drivers, scenario packs, alerts.

Then open the dashboard (optional):

```bash
gold ui
```

Your browser should open [http://127.0.0.1:8501](http://127.0.0.1:8501). Leave that terminal running. Use another terminal for `gold brief` or `gold run`. Press Ctrl+C to stop the UI only.

If port 8501 is busy:

```bash
gold ui --port 8502
```

The UI **does not poll**. After you run `gold data update` or `gold forecast` in another terminal, click **Reload from DuckDB**. The sidebar timestamp updates even when the numbers are unchanged.

---

## Everyday use

### Morning check

```bash
cd GoldPulse
source .venv/bin/activate
gold brief
```

### Refresh data and forecasts (recommended daily command)

```bash
gold run
```

`gold run` will:

1. Take an exclusive lock so two jobs cannot collide
2. Back up DuckDB and parquet
3. Ingest new prints with retries
4. Skip the forecast if gold, USD/INR, or parity is missing or stale
5. Reuse yesterday’s live book if the cutoff has not moved
6. Score forecasts whose target dates have now printed
7. Write an audit row with a unique run id

It **does not retrain models**. Champions stay frozen until you explicitly run `gold train --all`.

### What-if calculator (not a forecast)

```bash
gold scenario --xau 1 --usdinr 1 --dxy 1 --yield-bp 25 --duty-pp 1 --premium-pp 1
gold scenario --pack
```

`--pack` prints named Bull / Bear / Stress identity shocks. Outputs are labelled **SCENARIO, NOT A FORECAST**.

---

## Command reference

| Command | What it does |
|---|---|
| `gold init` | Create local folders, DuckDB schema, and catalog |
| `gold data update` | Incremental download of public series |
| `gold data update --full` | Re-download from 2010 instead of incremental |
| `gold data history` | Coverage table and restricted sources |
| `gold data validate` | Gaps, flags, anomalies (anomalies are kept unless out of range) |
| `gold features build` | Rebuild the leak-free feature matrix |
| `gold train --all` | Train, compare, promote. **Only way to change champions** |
| `gold backtest` | Chronological holdout comparison |
| `gold compare` | Same as `gold backtest` |
| `gold forecast --horizon 1d` | Write one horizon |
| `gold forecast --horizon 7d` | Write the 7-day horizon |
| `gold forecast --horizon all` | Write 1d, 7d, 30d, 90d |
| `gold brief` | Daily intelligence summary |
| `gold monitor` | Score realized outcomes, rolling MAE / coverage |
| `gold scenario` | Custom identity / elasticity shocks |
| `gold scenario --pack` | Bull, bear, stress packs |
| `gold status` | DuckDB counts, last ingest, source health |
| `gold health` | Freshness gate for critical series |
| `gold backup` | Copy DuckDB + parquet to `data/backups/` |
| `gold restore --from data/backups/<stamp> --yes` | Restore a backup (stop `gold ui` first) |
| `gold report` | Write a markdown report under `reports/output/` |
| `gold run` | Daily ops loop (no retrain) |
| `gold ui` | Local Streamlit app on 127.0.0.1:8501 |
| `gold tui` | Optional Textual terminal dashboard |

### Useful options

```bash
gold forecast --horizon all
gold ui --port 8502
gold restore --from data/backups/20261101T120000Z --yes
```

`gold restore` refuses to overwrite unless you pass `--yes`.

---

## Schedule it (optional)

Weekday mornings, Asia/Kolkata:

```cron
15 7 * * 1-5 cd /path/to/GoldPulse && /path/to/GoldPulse/.venv/bin/gold run >> data/logs/cron.log 2>&1
```

Or use the bundled script:

```bash
chmod +x scripts/daily.sh
```

Point launchd / cron at `scripts/daily.sh`. The app stays on your machine. There is no cloud deploy.

---

## Production forecasts (v1.0)

Chronological holdout, pipeline `v2-logret`. Not a random train/test split.

| Horizon | Champion | Holdout MAPE | Holdout MAE (INR / 10g) | Interval |
|---|---|---|---|---|
| 1 calendar day | naive | 1.32% | 1,782 | causal EWMA × √h |
| 7 calendar days | ARIMA | 2.92% | 3,942 | residual RMSE |
| 30 calendar days | naive | 5.81% | 7,827 | residual RMSE |
| 90 calendar days | naive | 10.81% | 14,522 | residual RMSE |

Naive means “tomorrow’s price equals today’s close” in log-return space (predicted change is zero). ARIMA at 7d is a 0.04% near-tie that still cleared the written promotion rule (holdout **and** 3 of 5 regimes). XGBoost and LightGBM are not production champions.

---

## Data

Stored locally in **DuckDB + Parquet** under `data/` (gitignored).

| Series | Source | Role |
|---|---|---|
| `gold_usd_oz` | Yahoo `GC=F` | COMEX gold |
| `usd_inr` | Yahoo `INR=X` | FX |
| `gold_inr_10g` | derived | Primary target (parity) |
| `dxy` | Yahoo | Dollar index |
| `us_10y_yield` | Yahoo `^TNX` | Nominal 10y |
| `us_real_yield_10y` | US Treasury TIPS | Real 10y |
| `wti_usd` | Yahoo `CL=F` | Oil |
| `gold_etf_nse` / `gold_etf_inr_10g` | Yahoo `GOLDBEES.NS` | Onshore ETF proxy |
| `india_cpi` | World Bank | Macro, lagged |
| import duty, RBI repo, festivals | curated tables in `config/` | Policy / calendar |
| `mcx_gold_futures` | unavailable | Not fabricated |
| `ibja_gold` | unavailable | Not fabricated |

FRED CSV is a fail-fast fallback and is often empty. Empty fallbacks are not treated as stale gold prints.

---

## How look-ahead is prevented

- Exogenous series join on `available_at <= feature_date` (publication lag for CPI and rates).
- Splits are chronological. Rows are never shuffled.
- Models forecast **h-day log returns** and reconstruct `price = close[t] × exp(ŷ)`.
- Daily `gold run` will not forecast if `gold_usd_oz`, `usd_inr`, or `gold_inr_10g` is missing, invalid, or stale.

---

## Project layout

```
config/          sources, duty/repo/festival tables
src/gold/        ingestion, DuckDB, features, models, forecast, ops, CLI, UI
tests/           unit and integration tests
scripts/daily.sh local cron/launchd helper
reports/output/  generated reports (local; not required to clone)
data/            DuckDB, parquet, backups, logs (created on your machine)
```

---

## Tests

```bash
source .venv/bin/activate
pytest
```

---

## Troubleshooting

**`Got unexpected extra argument(s)`**  
You pasted a comment on the same line, for example `gold ui # http://127.0.0.1:8501`. Run the command alone.

**`Streamlit is not installed`**  
`pip install -e ".[ui]"` then `gold ui`.

**Reload in the UI does nothing visible**  
The catalog did not change. Check the sidebar **Loaded … UTC** timestamp. Run `gold data update` or `gold forecast --horizon all` in another terminal, then click Reload.

**Forecast still shows an old naive 7d row after `gold train --all`**  
`gold forecast` will not overwrite a live book for the same cutoff. A new gold session (new cutoff) writes a new book. Daily `gold run` is the intended path.

**`gold run` says locked**  
Another `gold run` is in progress. Wait, or delete `data/gold.run.lock` only if that process is dead.

**Restore**  
Stop `gold ui` first, then:

```bash
gold restore --from data/backups/<timestamp> --yes
```

---

## Disclaimer

GoldPulse is a research and monitoring tool. Forecasts can be wrong. Intervals are not guaranteed coverage. Scenarios are identity overlays, not statistically validated predictions. Official MCX and IBJA prints are not in this database. Do not treat output as investment, trading, or tax advice.
