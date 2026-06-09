"""Unit tests to improve coverage for app/core/lifespan.py.

Targets model-not-found branches, cleanup, and SHAP init failures
(coverage was 81 %).
"""

from unittest.mock import Mock, patch

import pytest

from app.core.lifespan import ModelManager


class TestModelManagerInit:
    """Tests for ``ModelManager`` initial state."""

    def test_initial_state(self):
        """All attributes are None/False after construction."""
        mgr = ModelManager()

        assert mgr.loan_model is None
        assert mgr.credit_card_model is None
        assert mgr.model_loaded is False
        assert mgr.credit_card_model_loaded is False
        assert mgr.encoder_loaded is False
        assert mgr.credit_card_encoder_loaded is False
        assert mgr.scaler_loaded is False
        assert mgr.credit_card_scaler_loaded is False


class TestModelManagerLoadMissingFiles:
    """Test load_models when artifact files do not exist."""

    @pytest.mark.asyncio
    async def test_load_models_missing_all_artifacts(self, tmp_path):
        """All paths point to non-existent files → degraded / placeholder mode."""
        with patch("app.core.lifespan.get_settings") as mock_settings:
            s = mock_settings.return_value
            s.MODEL_PATH = str(tmp_path / "missing_model.json")
            s.ENCODER_PATH = str(tmp_path / "missing_encoder.pkl")
            s.SCALER_PATH = str(tmp_path / "missing_scaler.pkl")
            s.CREDIT_CARD_MODEL_PATH = str(tmp_path / "missing_cc_model.json")
            s.CREDIT_CARD_ENCODER_PATH = str(tmp_path / "missing_cc_encoder.pkl")
            s.CREDIT_CARD_SCALER_PATH = str(tmp_path / "missing_cc_scaler.pkl")

            mgr = ModelManager()
            await mgr.load_models()

            assert mgr.model_loaded is False
            assert mgr.loan_model is None
            assert mgr.encoder_loaded is False
            assert mgr.scaler_loaded is False
            assert mgr.credit_card_model_loaded is False
            assert mgr.credit_card_encoder_loaded is False
            assert mgr.credit_card_scaler_loaded is False


class TestModelManagerCleanup:
    """Tests for ``ModelManager.cleanup``."""

    @pytest.mark.asyncio
    async def test_cleanup_resets_state(self):
        """All attributes are reset to None/False after cleanup."""
        mgr = ModelManager()

        # Simulate loaded state
        mgr.loan_model = Mock()
        mgr.encoder = Mock()
        mgr.scaler = Mock()
        mgr.explainer = Mock()
        mgr.model_loaded = True
        mgr.encoder_loaded = True
        mgr.scaler_loaded = True

        mgr.credit_card_model = Mock()
        mgr.credit_card_encoder = Mock()
        mgr.credit_card_scaler = Mock()
        mgr.credit_card_explainer = Mock()
        mgr.credit_card_model_loaded = True
        mgr.credit_card_encoder_loaded = True
        mgr.credit_card_scaler_loaded = True

        await mgr.cleanup()

        # Loan
        assert mgr.loan_model is None
        assert mgr.encoder is None
        assert mgr.scaler is None
        assert mgr.explainer is None
        assert mgr.model_loaded is False
        assert mgr.encoder_loaded is False
        assert mgr.scaler_loaded is False

        # Credit card
        assert mgr.credit_card_model is None
        assert mgr.credit_card_encoder is None
        assert mgr.credit_card_scaler is None
        assert mgr.credit_card_explainer is None
        assert mgr.credit_card_model_loaded is False
        assert mgr.credit_card_encoder_loaded is False
        assert mgr.credit_card_scaler_loaded is False


class TestModelManagerLoadWithSHAPFailure:
    """Test SHAP explainer failure branches."""

    @pytest.mark.asyncio
    async def test_shap_import_error(self, tmp_path):
        """When shap import fails, model still loads but explainer is None."""
        model_path = tmp_path / "loans_model.json"

        # Create a minimal valid XGBoost model file
        import numpy as np
        import xgboost as xgb

        dtrain = xgb.DMatrix(np.zeros((2, 3)), label=[0, 1])
        bst = xgb.train({"max_depth": 1, "objective": "binary:logistic"}, dtrain, 1)
        bst.save_model(str(model_path))

        with patch("app.core.lifespan.get_settings") as mock_settings:
            s = mock_settings.return_value
            s.MODEL_PATH = str(model_path)
            s.ENCODER_PATH = str(tmp_path / "missing_encoder.pkl")
            s.SCALER_PATH = str(tmp_path / "missing_scaler.pkl")
            s.CREDIT_CARD_MODEL_PATH = str(tmp_path / "missing_cc_model.json")
            s.CREDIT_CARD_ENCODER_PATH = str(tmp_path / "missing_cc_encoder.pkl")
            s.CREDIT_CARD_SCALER_PATH = str(tmp_path / "missing_cc_scaler.pkl")

            mgr = ModelManager()

            # Patch shap import to raise ImportError
            with patch.dict("sys.modules", {"shap": None}):
                await mgr.load_models()

            assert mgr.model_loaded is True
            assert mgr.explainer is None
