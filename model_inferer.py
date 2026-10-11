"""Inference for the saved healthy-reference bearing anomaly model.

Inputs must use the same feature extraction and units as Q2_model.ipynb.
An anomaly flag indicates departure from healthy operation, not a confirmed fault.
Only load trusted joblib artifacts.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "artifacts" / "bearing_anomaly_model.joblib"
)


class ModelInferer:
    """Load once and score one feature dictionary or a batch of named features."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        bundle = joblib.load(model_path)
        if not isinstance(bundle, dict) or bundle.get("format_version") != 1:
            raise ValueError("Unsupported model bundle format.")
        if bundle.get("score_method") != "negative_score_samples":
            raise ValueError("Unsupported anomaly score convention.")
        if bundle.get("decision_rule") != "anomaly_score > threshold":
            raise ValueError("Unsupported anomaly decision rule.")

        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.feature_columns = list(bundle["feature_columns"])
        if not np.isfinite(self.threshold):
            raise ValueError("Model threshold must be finite.")
        if self.feature_columns != list(self.model.feature_names_in_):
            raise ValueError("Saved feature names do not match the fitted model.")

    def predict(
        self,
        features: Mapping | Sequence[Mapping] | pd.DataFrame,
    ) -> pd.DataFrame:
        """Return anomaly_score, threshold, and anomaly for each supplied row.

        Required columns are ordered from the saved bundle; extra columns are
        ignored. Missing, nonnumeric, and non-finite values raise ValueError.
        DataFrame row indexes are preserved. No feature extraction or fitting
        occurs here.
        """
        if isinstance(features, pd.DataFrame):
            data = features.copy()
        elif isinstance(features, Mapping):
            data = pd.DataFrame([features])
        else:
            data = pd.DataFrame(features)

        if data.empty:
            raise ValueError("Provide at least one feature row.")
        if not data.columns.is_unique:
            raise ValueError("Feature columns must be unique.")
        missing = [name for name in self.feature_columns if name not in data]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        values = data.loc[:, self.feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        if not finite.all():
            invalid = [
                name for name, valid in zip(self.feature_columns, finite.all(axis=0))
                if not valid
            ]
            raise ValueError(f"Features must contain finite numeric values: {invalid}")

        scores = -self.model.score_samples(values)
        return pd.DataFrame(
            {
                "anomaly_score": scores,
                "threshold": self.threshold,
                "anomaly": scores > self.threshold,
            },
            index=data.index,
        )


def predict_features(
    features: Mapping | Sequence[Mapping] | pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Load the saved model and score features; reuse ModelInferer for many calls."""
    return ModelInferer(model_path).predict(features)
