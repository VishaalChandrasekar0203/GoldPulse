from __future__ import annotations

from typer.testing import CliRunner

from gold.cli import app

runner = CliRunner()


def test_help_lists_required_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    text = result.stdout
    for cmd in ["init", "data", "features", "train", "backtest", "forecast", "compare", "status", "report"]:
        assert cmd in text


def test_forecast_rejects_bad_horizon():
    result = runner.invoke(app, ["forecast", "--horizon", "2y"])
    assert result.exit_code != 0


def test_data_help():
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    assert "update" in result.stdout
    assert "history" in result.stdout
    assert "validate" in result.stdout
