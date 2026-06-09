"""FastAPI application lifespan context manager.

Handles model loading on startup and cleanup on shutdown.
"""

import logging
import pickle
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import xgboost as xgb
from fastapi import FastAPI

from app.core.settings import get_settings

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Artifact set — groups related model/encoder/scaler/explainer
# ------------------------------------------------------------------

@dataclass
class ArtifactSet:
    """Holds a single model's runtime artifacts and loaded-flags."""

    model: xgb.Booster | None = None
    encoder: object = None
    scaler: object = None
    explainer: object = None
    model_loaded: bool = False
    encoder_loaded: bool = False
    scaler_loaded: bool = False

    def reset(self) -> None:
        """Release all artifacts and reset flags."""
        self.model = None
        self.encoder = None
        self.scaler = None
        self.explainer = None
        self.model_loaded = False
        self.encoder_loaded = False
        self.scaler_loaded = False


def _load_artifact_set(
    model_path: str,
    encoder_path: str,
    scaler_path: str,
    label: str,
) -> ArtifactSet:
    """Load an XGBoost model, encoder, scaler, and SHAP explainer from disk.

    Each component is loaded independently so a missing optional file
    does not block the rest.

    Args:
        model_path: Path to the XGBoost model JSON file.
        encoder_path: Path to the pickled categorical encoder.
        scaler_path: Path to the pickled ``StandardScaler``.
        label: Human-readable name for log messages (e.g. ``"loan"``).

    Returns:
        Populated ``ArtifactSet``.

    """
    artifacts = ArtifactSet()

    # --- XGBoost model ---
    mp = Path(model_path)
    if mp.exists():
        artifacts.model = xgb.Booster()
        artifacts.model.load_model(str(mp))
        artifacts.model_loaded = True
        logger.info("✓ Loaded %s XGBoost model from %s", label, mp)
        logger.info("  Model has %d boosting rounds", artifacts.model.num_boosted_rounds())
    else:
        logger.warning("⚠ %s model file not found: %s", label.capitalize(), mp)
        if label == "loan":
            logger.warning("  The microservice will run in PLACEHOLDER mode")
            logger.warning("  Please train the model using: python train_model.py")

    # --- Encoder ---
    ep = Path(encoder_path)
    if ep.exists():
        with open(ep, "rb") as f:
            artifacts.encoder = pickle.load(f)
        artifacts.encoder_loaded = True
        logger.info("✓ Loaded %s encoder from %s", label, ep)
        if hasattr(artifacts.encoder, "encoders"):
            logger.info("  Encoder has %d categorical features", len(artifacts.encoder.encoders))
    else:
        logger.info(
            "  %s encoder not found — using built-in categorical encoding",
            label.capitalize(),
        )

    # --- Scaler ---
    sp = Path(scaler_path)
    if sp.exists():
        with open(sp, "rb") as f:
            artifacts.scaler = pickle.load(f)
        artifacts.scaler_loaded = True
        logger.info("✓ Loaded %s scaler from %s", label, sp)
    else:
        logger.info("  No %s scaler found — numeric features will not be scaled", label)

    # --- SHAP Explainer ---
    if artifacts.model is not None:
        try:
            import shap
            artifacts.explainer = shap.TreeExplainer(artifacts.model)
            logger.info("✓ %s SHAP TreeExplainer pre-created at startup", label.capitalize())
        except ImportError as import_err:
            logger.warning(
                "⚠ Could not pre-create %s SHAP explainer (missing package): %s",
                label, import_err,
            )
        except Exception as shap_exc:
            logger.warning(
                "⚠ Could not pre-create %s SHAP explainer: %s", label, shap_exc
            )

    return artifacts


class ModelManager:
    """Manages ML model lifecycle and state.

    Attributes:
        loan_model: Loaded XGBoost Booster for loans, or ``None``.
        credit_card_model: Loaded XGBoost Booster for credit cards, or ``None``.
        encoder: Loaded ``TrainingEncoder`` for loans, or ``None``.
        credit_card_encoder: Loaded ``TrainingEncoder`` for credit cards, or ``None``.
        scaler: Loaded ``StandardScaler`` for loans, or ``None``.
        credit_card_scaler: Loaded ``StandardScaler`` for credit cards, or ``None``.
        explainer: Pre-built SHAP ``TreeExplainer`` for loans, or ``None``.
        credit_card_explainer: Pre-built SHAP ``TreeExplainer`` for credit cards, or ``None``.
        model_loaded: Whether the loan model file was loaded successfully.
        credit_card_model_loaded: Whether the credit card model file was loaded successfully.
        encoder_loaded: Whether the encoder file was loaded successfully.
        credit_card_encoder_loaded: Whether the credit card encoder was loaded successfully.
        scaler_loaded: Whether the scaler file was loaded successfully.
        credit_card_scaler_loaded: Whether the credit card scaler was loaded successfully.

    """

    def __init__(self) -> None:
        """Initialise empty artifact sets for loan and credit card models."""
        self._loan = ArtifactSet()
        self._credit_card = ArtifactSet()

    # Preserve the original attribute interface via properties so that
    # existing code (InferenceService, risk_router, tests) keeps working.

    # --- Loan properties ---
    @property
    def loan_model(self):  # noqa: D102
        return self._loan.model

    @property
    def encoder(self):  # noqa: D102
        return self._loan.encoder

    @property
    def scaler(self):  # noqa: D102
        return self._loan.scaler

    @property
    def explainer(self):  # noqa: D102
        return self._loan.explainer

    @property
    def model_loaded(self):  # noqa: D102
        return self._loan.model_loaded

    @property
    def encoder_loaded(self):  # noqa: D102
        return self._loan.encoder_loaded

    @property
    def scaler_loaded(self):  # noqa: D102
        return self._loan.scaler_loaded

    # --- Credit card properties ---
    @property
    def credit_card_model(self):  # noqa: D102
        return self._credit_card.model

    @property
    def credit_card_encoder(self):  # noqa: D102
        return self._credit_card.encoder

    @property
    def credit_card_scaler(self):  # noqa: D102
        return self._credit_card.scaler

    @property
    def credit_card_explainer(self):  # noqa: D102
        return self._credit_card.explainer

    @property
    def credit_card_model_loaded(self):  # noqa: D102
        return self._credit_card.model_loaded

    @property
    def credit_card_encoder_loaded(self):  # noqa: D102
        return self._credit_card.encoder_loaded

    @property
    def credit_card_scaler_loaded(self):  # noqa: D102
        return self._credit_card.scaler_loaded

    async def load_models(self) -> None:
        """Load XGBoost model, encoder and scaler from disk.

        Raises:
            Exception: If a critical file cannot be loaded.

        """
        settings = get_settings()

        try:
            self._loan = _load_artifact_set(
                model_path=settings.MODEL_PATH,
                encoder_path=settings.ENCODER_PATH,
                scaler_path=settings.SCALER_PATH,
                label="loan",
            )

            self._credit_card = _load_artifact_set(
                model_path=settings.CREDIT_CARD_MODEL_PATH,
                encoder_path=settings.CREDIT_CARD_ENCODER_PATH,
                scaler_path=settings.CREDIT_CARD_SCALER_PATH,
                label="credit card",
            )

        except Exception as e:
            logger.error("✗ Error loading models: %s", str(e), exc_info=True)
            raise

    async def cleanup(self) -> None:
        """Release all loaded artefacts."""
        self._loan.reset()
        self._credit_card.reset()
        logger.info("✓ Models unloaded")


# Global model manager instance
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager.

    Manages the full lifecycle of the application:

    - **Startup**: Load ML model, encoder, and scaler; pre-create SHAP
      explainer; store ``ModelManager`` on ``app.state``.
    - **Shutdown**: Release all loaded artefacts and free memory.

    Args:
        app: The running ``FastAPI`` application instance.

    """
    # --- Startup ---
    logger.info("=" * 60)
    logger.info("Starting %s...", app.title)
    logger.info("=" * 60)

    await model_manager.load_models()

    # Expose model_manager on app.state for dependency injection
    app.state.model_manager = model_manager

    if model_manager.model_loaded:
        logger.info("✓ %s started successfully (production mode)", app.title)
    else:
        logger.warning("⚠ %s started in PLACEHOLDER mode", app.title)
    logger.info("=" * 60)

    yield

    # --- Shutdown ---
    logger.info("=" * 60)
    logger.info("Shutting down %s...", app.title)
    logger.info("=" * 60)
    await model_manager.cleanup()
    logger.info("✓ %s shut down cleanly", app.title)
