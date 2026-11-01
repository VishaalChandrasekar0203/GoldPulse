"""Indian gold price forecasting system."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gold-forecasting")
except PackageNotFoundError:
    __version__ = "1.0.0"

__all__ = ["__version__"]
