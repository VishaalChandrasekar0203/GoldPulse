from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Header, Log, Static, TabbedContent, TabPane

from gold.config import load_config
from gold.database.store import GoldStore
from gold.terminal.brief import build_brief


class GoldTUI(App):
    """Daily gold intelligence terminal."""

    CSS = """
    Screen { background: #111111; }
    #hero { height: 7; padding: 1; background: #1b1408; color: #f3d38a; }
    DataTable { height: 1fr; }
    Log { height: 1fr; border: solid #3d2e12; }
    #brief-body { padding: 1; }
    """
    TITLE = "Gold Intelligence"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="hero")
        with TabbedContent():
            with TabPane("Brief", id="tab-brief"):
                yield Static(id="brief-body")
            with TabPane("Market", id="tab-market"):
                yield DataTable(id="market")
            with TabPane("Forecasts", id="tab-fc"):
                yield DataTable(id="forecasts")
            with TabPane("Monitor", id="tab-mon"):
                yield DataTable(id="monitor")
            with TabPane("Scenarios", id="tab-scen"):
                yield DataTable(id="scenarios")
            with TabPane("Audit", id="tab-models"):
                yield DataTable(id="models")
            with TabPane("Health", id="tab-health"):
                with Horizontal():
                    yield DataTable(id="coverage")
                    yield Log(id="logs")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_data()

    def action_refresh(self) -> None:
        self.refresh_data()

    def refresh_data(self) -> None:
        cfg = load_config()
        store = GoldStore(cfg)
        brief = build_brief(cfg, store, run_monitor=False)
        snap = brief.get("snapshot")
        vol = brief.get("parity_vol_ewma")
        vol_txt = f"{vol * 100:.2f}%" if vol else "n/a"
        hero = self.query_one("#hero", Static)
        if snap is not None:
            hero.update(
                f"[b]{snap.parity_inr_10g:,.0f} INR/10g parity[/b]  as-of {snap.as_of.date()}  "
                f"vol {vol_txt} EWMA\n"
                f"XAU {snap.gold_usd_oz:,.2f}  USDINR {snap.usd_inr:,.4f}  "
                f"DXY {snap.dxy if snap.dxy is not None else 'n/a'}\n"
                "International parity — not MCX/IBJA. Scenarios are not forecasts. "
                "Champions unchanged (naive 1/30/90d, ARIMA 7d; 1d EWMA interval).  r refresh  q quit"
            )
        else:
            hero.update("[b]Gold intelligence[/b] — run gold data update. r refresh, q quit.")

        body = self.query_one("#brief-body", Static)
        lines = [brief.get("target_note") or ""]
        fc1 = brief.get("forecast_1d")
        if fc1:
            lines.append(
                f"1d forecast {float(fc1['point']):,.2f}  "
                f"interval [{float(fc1['lower']):,.2f}, {float(fc1['upper']):,.2f}]  "
                f"{fc1.get('model_name')} / {fc1.get('model_version')}"
            )
        alerts = brief.get("alerts")
        if alerts is not None and not alerts.empty:
            lines.append("Alerts: " + " | ".join(alerts["message"].astype(str).head(4).tolist()))
        else:
            lines.append("No monitor alerts.")
        body.update("\n".join(lines))

        intel = brief.get("intelligence") or {}
        import pandas as pd

        cards = pd.DataFrame(intel.get("cards") or [])
        self._fill_table("market", cards)
        self._fill_table("forecasts", brief.get("forecasts"))
        self._fill_table("monitor", brief.get("rolling"))
        packs = brief.get("scenario_packs")
        if packs is not None and not packs.empty:
            self._fill_table(
                "scenarios",
                packs[packs["kind"] == "identity_combined"][
                    ["pack_label", "delta_inr_10g", "delta_pct", "kind"]
                ],
            )
        else:
            self._fill_table("scenarios", pd.DataFrame())
        champs = brief.get("champions")
        self._fill_table("models", champs)
        self._fill_table("coverage", store.coverage())
        logw = self.query_one("#logs", Log)
        logw.clear()
        log_path = cfg.log_dir / "gold.log"
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]:
                logw.write_line(line)
        else:
            logw.write_line("No log file yet. Run gold data update.")
        store.close()

    def _fill_table(self, widget_id: str, df) -> None:
        table = self.query_one(f"#{widget_id}", DataTable)
        table.clear(columns=True)
        if df is None or getattr(df, "empty", True):
            table.add_columns("status")
            table.add_row("empty")
            return
        cols = [str(c) for c in df.columns[:10]]
        table.add_columns(*cols)
        for _, row in df.head(50).iterrows():
            table.add_row(*[str(row[c])[:40] if row[c] is not None else "" for c in cols])
