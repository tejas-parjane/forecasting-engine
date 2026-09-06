"""Custom exceptions for the Forecasting Engine."""


class ForecastingError(Exception):
    """Base exception for the forecasting engine."""


class DataValidationError(ForecastingError):
    """Raised when input data fails validation."""


class DataLoadError(ForecastingError):
    """Raised when input data cannot be loaded."""


class FrequencyDetectionError(ForecastingError):
    """Raised when the data frequency cannot be determined."""


class ModelTrainingError(ForecastingError):
    """Raised when a model fails to train."""


class InsufficientDataError(ForecastingError):
    """Raised when there is not enough data to perform an operation."""


class ModelNotFoundError(ForecastingError):
    """Raised when a requested model does not exist."""


class AIProviderError(ForecastingError):
    """Raised when the AI provider fails or is unavailable."""
