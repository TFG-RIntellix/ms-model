"""Application settings and configuration management.

Uses ``pydantic-settings`` for environment variable support with
type-safe defaults.  Settings are cached via :func:`get_settings`
so that only one instance exists per process.
"""

from importlib.metadata import version, PackageNotFoundError
from functools import lru_cache
import logging

from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Resolve package version from pyproject.toml metadata (single source of truth)
# ---------------------------------------------------------------------------
try:
    _PKG_VERSION = version("ms-model")
except PackageNotFoundError:
    _PKG_VERSION = "1.1.0"  # fallback when running without pip install -e


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Attributes:
        APP_NAME: Human-readable service name.
        APP_VERSION: Semantic version string (auto-read from ``pyproject.toml``).
        DEBUG: Enable debug mode (extra logging).
        LOG_LEVEL: Python log level name (e.g. ``INFO``, ``DEBUG``).
        HOST: Bind address for uvicorn.
        PORT: Bind port for uvicorn.
        MODEL_PATH: Path to the trained XGBoost model JSON.
        ENCODER_PATH: Path to the pickled categorical encoder.
        SCALER_PATH: Path to the pickled ``StandardScaler``.
        API_V1_PREFIX: URL prefix for versioned API routes.
        RISK_THRESHOLD_LOW: PD below this value is classified as *Low* risk.
        RISK_THRESHOLD_HIGH: PD at or above this value is classified as *High* risk.
        CORS_ORIGINS: Allowed CORS origins.
    """

    # Application info
    APP_NAME: str = "Credit Risk Engine"
    APP_VERSION: str = _PKG_VERSION
    DEBUG: bool = False

    # Logging
    LOG_LEVEL: str = "INFO"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model paths
    MODEL_PATH: str = "app/ml_artifacts/loans_model.json"
    ENCODER_PATH: str = "app/ml_artifacts/encoder.pkl"
    SCALER_PATH: str = "app/ml_artifacts/scaler.pkl"

    # Credit card model paths
    CREDIT_CARD_MODEL_PATH: str = "app/ml_artifacts/credit_card_model.json"
    CREDIT_CARD_ENCODER_PATH: str = "app/ml_artifacts/credit_card_encoder.pkl"
    CREDIT_CARD_SCALER_PATH: str = "app/ml_artifacts/credit_card_scaler.pkl"

    # API settings
    API_V1_PREFIX: str = "/api/v1"

    # Risk classification thresholds
    RISK_THRESHOLD_LOW: float = 0.15
    RISK_THRESHOLD_HIGH: float = 0.35

    # CORS — MUST be overridden in production with specific origins.
    # An empty default forces operators to configure allowed origins explicitly.
    CORS_ORIGINS: list[str] = []

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance.

    The cache ensures only one ``Settings`` object exists per process,
    avoiding repeated env-file reads.

    Returns:
        Singleton ``Settings`` instance.
    """
    return Settings()


def configure_logging(log_level: str = "INFO") -> None:
    """Configure root logger with a consistent format.

    Args:
        log_level: Python log-level name (case-insensitive).
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )
