"""Shared SHAP utility for tree-based MLModel implementations.

Wraps `shap.TreeExplainer` for XGBoost / CATBoost / RandomForest. Models that
need a different explainer (e.g., kernel SHAP for non-tree models) can call
shap directly.
"""

from typing import Any

import numpy as np
import shap


def compute_shap_values(model: Any, X: np.ndarray) -> np.ndarray:
    """Compute SHAP values for a tree model.

    Returns an array shaped (n_samples, n_features) — for binary classifiers
    these are values for the positive class (so positive contribution pushes
    the prediction toward class 1).
    """
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)

    # XGBoost binary: returns (n_samples, n_features) directly.
    # Some sklearn classifiers: returns a list per class.
    if isinstance(values, list):
        return np.array(values[1] if len(values) > 1 else values[0])
    return np.array(values)
