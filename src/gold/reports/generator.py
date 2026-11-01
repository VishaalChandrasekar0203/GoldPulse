from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gold-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from gold.config import AppConfig
from gold.database.store import GoldStore
from gold.logging import get_logger

log = get_logger("reports")


class ReportGenerator:
    def __init__(self, cfg: AppConfig, store: GoldStore):
        self.cfg = cfg
        self.store = store
        self.out_dir = cfg.reports_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write(self) -> Path:
        now = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.out_dir / f"report_{now}.md"
        coverage = self.store.coverage()
        catalog = self.store.catalog()
        forecasts = self.store.latest_forecasts()
        champs = self.store.champions()
        exps = self.store.experiments()
        target = self.store.load_series(self.cfg.primary_series)

        chart_path = self.out_dir / f"price_{now}.png"
        if not target.empty:
            t = target.copy()
            t["obs_date"] = pd.to_datetime(t["obs_date"])
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t["obs_date"], t["value"], color="#c9a227", lw=1.2)
            ax.set_title("International gold parity (INR / 10g) — not MCX/IBJA official")
            ax.set_ylabel("INR per 10g")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(chart_path, dpi=120)
            plt.close(fig)

        lines = [
            f"# Gold Forecasting Report ({now})",
            "",
            "This report uses **documented public series**. Official MCX Gold Futures and IBJA historical prints are **not** in the database. The primary target is international gold converted to INR per 10g (`gold_inr_10g`). Models forecast h-day log returns and reconstruct prices as close[t]×exp(ŷ).",
            "",
            "## Data coverage",
            "",
            coverage.to_string(index=False) if not coverage.empty else "_No observations stored._",
            "",
            "## Restricted / unavailable sources",
            "",
        ]
        restricted = catalog[catalog["status"].isin(["unavailable", "restricted"])] if not catalog.empty else catalog
        if restricted.empty:
            lines.append("_None recorded._")
        else:
            for _, r in restricted.iterrows():
                lines.append(f"- **{r['series_id']}**: {r['notes']}")
        lines += ["", "## Champions (out-of-sample holdout)", ""]
        if champs.empty:
            lines.append("No champion has been promoted. Run `gold train --all`.")
        else:
            lines.append(champs[["model_name", "horizon", "cutoff_date", "experiment_id", "notes"]].to_string(index=False))
        lines += ["", "## Latest forecasts", ""]
        if forecasts.empty:
            lines.append("No forecasts stored. Run `gold forecast --horizon all`.")
        else:
            show = forecasts[
                [
                    "horizon_calendar_days",
                    "horizon_trading_days",
                    "target_date",
                    "target_is_trading_day",
                    "point",
                    "lower",
                    "upper",
                    "model_name",
                    "model_version",
                    "cutoff_date",
                ]
            ]
            lines.append(show.to_string(index=False))
        from gold.evaluation.monitor import rolling_metrics

        outcomes = self.store.forecast_outcomes()
        lines += ["", "## Forecast monitor", ""]
        if outcomes.empty:
            lines.append("No scored outcomes yet. Run `gold monitor`.")
        else:
            roll = rolling_metrics(outcomes)
            lines.append(roll.to_string(index=False) if not roll.empty else "_empty_")
            lines.append(
                "Monitor backfill uses the production policy (naive point; 1d EWMA × √h; "
                "longer horizons train-only σ). Live 7d champion remains ARIMA. "
                "Charts: monitor_mae_*d.png, monitor_coverage_*d.png."
            )
        lines += [
            "",
            "## Accuracy disclaimer",
            "",
            "Models forecast h-day log returns and reconstruct INR prices as close[t]×exp(ŷ). A non-naive model is promoted only when reconstructed-price MAE beats naive persistence on the latest chronological holdout **and** on at least three of five expanding-window historical regimes. Metrics are chronological out-of-sample, not random-split. Accuracy is **not claimed** beyond the stored experiment table.",
            "",
            f"Price chart: `{chart_path.name}`" if chart_path.exists() else "Price chart: unavailable (no target series).",
            "",
            f"Generated at {datetime.now(tz=timezone.utc).isoformat()}",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        log.info("wrote report %s", path)
        return path
