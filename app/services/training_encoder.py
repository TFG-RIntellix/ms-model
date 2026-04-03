"""Training encoder for categorical feature transformation.

Provides consistent label encoding between the training pipeline and
inference service, ensuring that each categorical value maps to the
same integer at both stages.

Typical usage::

    encoder = TrainingEncoder()
    encoder.fit(df, categorical_features=list(CATEGORICAL_FEATURES.keys()))
    encoder.save("app/ml_artifacts/encoder.pkl")
"""

import pickle
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class TrainingEncoder:
    """Categorical encoder for training and inference consistency.

    Wraps ``sklearn.preprocessing.LabelEncoder`` for each categorical
    feature.  Encoding is kept consistent between training and inference
    by sorting unique values before fitting, so the mapping is
    deterministic regardless of row order.

    Attributes:
        encoders: Mapping from feature name to its fitted ``LabelEncoder``.
        categorical_features: Feature names that were fitted.
    """

    def __init__(self) -> None:
        self.encoders: Dict[str, LabelEncoder] = {}
        self.categorical_features: List[str] = []

    def fit(self, df: pd.DataFrame, categorical_features: List[str]) -> "TrainingEncoder":
        """Fit label encoders on training data.

        For each categorical feature, sorts the unique values before
        fitting to guarantee a deterministic integer mapping.

        Args:
            df: Training ``DataFrame`` containing all feature columns.
            categorical_features: Names of columns to encode.

        Returns:
            ``self`` — allows method chaining.
        """
        self.categorical_features = categorical_features

        for feature in categorical_features:
            if feature in df.columns:
                encoder = LabelEncoder()
                unique_vals = sorted(df[feature].unique())
                encoder.fit(unique_vals)
                self.encoders[feature] = encoder

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted label encodings to a ``DataFrame``.

        Only columns present in both ``categorical_features`` and
        ``self.encoders`` are transformed; all other columns are
        passed through unchanged.

        Args:
            df: ``DataFrame`` to encode (may contain non-categorical columns).

        Returns:
            Copy of ``df`` with categorical columns replaced by integer codes.
        """
        df_encoded = df.copy()

        for feature in self.categorical_features:
            if feature in df_encoded.columns and feature in self.encoders:
                df_encoded[feature] = self.encoders[feature].transform(
                    df_encoded[feature]
                )

        return df_encoded

    def save(self, path: str) -> None:
        """Persist the fitted encoder to disk via pickle.

        Creates parent directories if they do not exist.

        Args:
            path: Destination file path (e.g. ``app/ml_artifacts/encoder.pkl``).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "TrainingEncoder":
        """Load a previously saved encoder from disk.

        Args:
            path: Path to the pickle file produced by :meth:`save`.

        Returns:
            Deserialised ``TrainingEncoder`` instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        with open(path, "rb") as f:
            encoder = pickle.load(f)
        return encoder
