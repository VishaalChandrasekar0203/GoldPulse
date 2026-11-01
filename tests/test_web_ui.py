from __future__ import annotations

from gold.web import APP_PATH, streamlit_command


def test_streamlit_app_file_exists():
    assert APP_PATH.exists()
    assert APP_PATH.name == "app.py"


def test_streamlit_command_binds_localhost():
    cmd = streamlit_command(APP_PATH, host="127.0.0.1", port=8501, python="python")
    assert cmd[:4] == ["python", "-m", "streamlit", "run"]
    assert str(APP_PATH) in cmd
    assert cmd[cmd.index("--server.address") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--server.port") + 1] == "8501"
    assert cmd[cmd.index("--browser.gatherUsageStats") + 1] == "false"


def test_ui_payload_includes_loaded_at(store, cfg):
    from gold.web.load import load_ui_payload

    payload = load_ui_payload(cfg=cfg, store=store)
    assert payload.get("loaded_at")
    assert payload["forecast_book"] == payload["brief"]["forecast_book"]


def test_web_load_module_does_not_import_streamlit():
    import gold.web.load as load

    assert not hasattr(load, "st")
    from gold.data.quality import SERIES_KIND

    assert "mcx_gold_futures" in SERIES_KIND
    assert "MCX" in SERIES_KIND["gold_inr_10g"]
    assert "IBJA" in SERIES_KIND["ibja_gold"]
    assert "unavailable" in SERIES_KIND["ibja_gold"].lower()
