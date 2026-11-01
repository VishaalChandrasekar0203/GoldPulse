from __future__ import annotations

from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "app.py"


def streamlit_command(
    app_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8501,
    python: str | None = None,
) -> list[str]:
    """Local-only Streamlit launch. Binds to localhost; no cloud flags."""
    import sys

    exe = python or sys.executable
    path = str(app_path or APP_PATH)
    return [
        exe,
        "-m",
        "streamlit",
        "run",
        path,
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "false",
    ]


__all__ = ["APP_PATH", "streamlit_command"]
