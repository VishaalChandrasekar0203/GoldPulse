class GoldError(Exception):
    """Base error for the forecasting system."""


class ConfigError(GoldError):
    pass


class IngestionError(GoldError):
    pass


class SourceUnavailable(IngestionError):
    """A requested market source is restricted or unreachable. Data is not fabricated."""


class ValidationError(GoldError):
    pass


class FeatureError(GoldError):
    pass


class ModelError(GoldError):
    pass


class InsufficientDataError(GoldError):
    pass


class LockBusy(GoldError):
    """Another gold run holds the daily lock."""


class BackupError(GoldError):
    pass
