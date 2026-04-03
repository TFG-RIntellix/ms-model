"""
FastAPI application lifespan context manager.
Handles model loading on startup and cleanup on shutdown.
"""

import logging
from contextlib import asynccontextmanager
import pickle

import xgboost as xgb
from fastapi import FastAPI
from pathlib import Path

from app.core.settings import get_settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ML model lifecycle and state.

    Attributes:
        loan_model: Loaded XGBoost Booster, or ``None``.
        encoder: Loaded ``TrainingEncoder``, or ``None``.
        scaler: Loaded ``StandardScaler``, or ``None``.
        explainer: Pre-built SHAP ``TreeExplainer``, or ``None``.
        model_loaded: Whether the model file was loaded successfully.
        encoder_loaded: Whether the encoder file was loaded successfully.
        scaler_loaded: Whether the scaler file was loaded successfully.
    """

    def __init__(self) -> None:
        self.loan_model = None
        self.encoder = None
        self.scaler = None
        self.explainer = None
        self.model_loaded = False
        self.encoder_loaded = False
        self.scaler_loaded = False

    async def load_models(self) -> None:
        """Load XGBoost model, encoder and scaler from disk.

        Raises:
            Exception: If a critical file cannot be loaded.
        """
        settings = get_settings()

        try:
            # --- XGBoost model ---
            model_path = Path(settings.MODEL_PATH)
            if model_path.exists():
                self.loan_model = xgb.Booster()
                self.loan_model.load_model(str(model_path))
                self.model_loaded = True
                logger.info("✓ Loaded XGBoost model from %s", model_path)
                logger.info("  Model has %d boosting rounds",
                            self.loan_model.num_boosted_rounds())
            else:
                logger.warning("⚠ Model file not found: %s", model_path)
                logger.warning("  The microservice will run in PLACEHOLDER mode")
                logger.warning("  Please train the model using: python train_model.py")
                self.model_loaded = False

            # --- Encoder ---
            encoder_path = Path(settings.ENCODER_PATH)
            if encoder_path.exists():
                with open(encoder_path, "rb") as f:
                    self.encoder = pickle.load(f)
                self.encoder_loaded = True
                logger.info("✓ Loaded encoder from %s", encoder_path)
                if hasattr(self.encoder, "encoders"):
                    logger.info("  Encoder has %d categorical features",
                                len(self.encoder.encoders))
            else:
                logger.info("  Using built-in categorical encoding (no pickle encoder)")
                self.encoder_loaded = False

            # --- Scaler ---
            scaler_path = Path(settings.SCALER_PATH)
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                self.scaler_loaded = True
                logger.info("✓ Loaded scaler from %s", scaler_path)
            else:
                logger.info("  No scaler found — numeric features will not be scaled")
                self.scaler_loaded = False

            # --- SHAP Explainer (pre-created to avoid per-request cold start) ---
            if self.loan_model is not None:
                try:
                    import shap  # lazy import — heavy, only needed when model is available
                    self.explainer = shap.TreeExplainer(self.loan_model)
                    logger.info("✓ SHAP TreeExplainer pre-created at startup")
                except ImportError as import_err:
                    logger.warning(
                        "⚠ Could not pre-create SHAP explainer (missing package): %s", 
                        import_err
                    )
                except Exception as shap_exc:
                    logger.warning(
                        "⚠ Could not pre-create SHAP explainer: %s", shap_exc
                    )

        except Exception as e:
            logger.error("✗ Error loading models: %s", str(e), exc_info=True)
            raise

    async def cleanup(self) -> None:
        """Release all loaded artefacts."""
        self.loan_model = None
        self.encoder = None
        self.scaler = None
        self.model_loaded = False
        self.encoder_loaded = False
        self.scaler_loaded = False
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
