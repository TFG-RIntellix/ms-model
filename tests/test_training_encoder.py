"""Unit tests for TrainingEncoder (app/services/training_encoder.py).

Covers fit, transform, save/load round-trip, and edge cases that were
previously untested (coverage was 52 %).
"""

import os
import tempfile

import pandas as pd
import pytest

from app.services.training_encoder import TrainingEncoder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_df() -> pd.DataFrame:
    """Return a small DataFrame with categorical + numeric columns."""
    return pd.DataFrame(
        {
            "gender": ["Mujer", "Hombre", "Otro", "Mujer"],
            "education": ["Grado", "Primaria", "Secundaria", "Posgrado"],
            "age": [25, 40, 33, 55],
            "income": [30000.0, 50000.0, 40000.0, 60000.0],
        }
    )


CATEGORICAL_COLS = ["gender", "education"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainingEncoderFit:
    """Tests for the fit() method."""

    def test_fit_stores_encoders(self):
        """Encoders dict contains all categorical features after fit."""
        encoder = TrainingEncoder()
        df = _sample_df()
        encoder.fit(df, CATEGORICAL_COLS)

        assert set(encoder.encoders.keys()) == set(CATEGORICAL_COLS)
        assert encoder.categorical_features == CATEGORICAL_COLS

    def test_fit_returns_self(self):
        """fit() returns self for method chaining."""
        encoder = TrainingEncoder()
        result = encoder.fit(_sample_df(), CATEGORICAL_COLS)
        assert result is encoder

    def test_fit_skips_missing_columns(self):
        """Columns not in the DataFrame are silently skipped."""
        encoder = TrainingEncoder()
        df = _sample_df()
        encoder.fit(df, ["gender", "nonexistent_column"])

        assert "gender" in encoder.encoders
        assert "nonexistent_column" not in encoder.encoders

    def test_fit_deterministic_sorted_order(self):
        """LabelEncoder is fitted on sorted unique values."""
        encoder = TrainingEncoder()
        df = _sample_df()
        encoder.fit(df, ["gender"])

        classes = list(encoder.encoders["gender"].classes_)
        assert classes == sorted(classes)


class TestTrainingEncoderTransform:
    """Tests for the transform() method."""

    def test_transform_encodes_categoricals(self):
        """Categorical columns become integers; numeric columns pass through."""
        encoder = TrainingEncoder()
        df = _sample_df()
        encoder.fit(df, CATEGORICAL_COLS)

        transformed = encoder.transform(df)

        # Categorical columns should now be integers
        assert transformed["gender"].dtype in ("int64", "int32", "int8")
        assert transformed["education"].dtype in ("int64", "int32", "int8")

        # Numeric columns should pass through unchanged
        assert list(transformed["age"]) == list(df["age"])
        assert list(transformed["income"]) == list(df["income"])

    def test_transform_does_not_mutate_original(self):
        """The original DataFrame is not modified."""
        encoder = TrainingEncoder()
        df = _sample_df()
        encoder.fit(df, CATEGORICAL_COLS)

        original_gender = list(df["gender"])
        encoder.transform(df)

        assert list(df["gender"]) == original_gender

    def test_transform_consistent_mapping(self):
        """Same value always maps to the same integer."""
        encoder = TrainingEncoder()
        df = _sample_df()
        encoder.fit(df, ["gender"])

        t1 = encoder.transform(df)
        t2 = encoder.transform(df)
        assert list(t1["gender"]) == list(t2["gender"])


class TestTrainingEncoderPersistence:
    """Tests for save() and load() round-trip."""

    def test_save_and_load_roundtrip(self):
        """Save → load preserves encoders and produces identical transforms."""
        encoder = TrainingEncoder()
        df = _sample_df()
        encoder.fit(df, CATEGORICAL_COLS)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "encoder.pkl")
            encoder.save(path)

            loaded = TrainingEncoder.load(path)

            assert loaded.categorical_features == encoder.categorical_features
            assert set(loaded.encoders.keys()) == set(encoder.encoders.keys())

            # Transforms should be identical
            orig_transformed = encoder.transform(df)
            loaded_transformed = loaded.transform(df)
            pd.testing.assert_frame_equal(orig_transformed, loaded_transformed)

    def test_save_creates_parent_dirs(self):
        """save() creates intermediate directories."""
        encoder = TrainingEncoder()
        encoder.fit(_sample_df(), CATEGORICAL_COLS)

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c", "encoder.pkl")
            encoder.save(nested)
            assert os.path.isfile(nested)

    def test_load_nonexistent_raises(self):
        """Loading from a nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TrainingEncoder.load("/tmp/nonexistent_encoder_file.pkl")
