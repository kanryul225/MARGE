"""XGBoost classifier on the sklearn breast cancer (Wisconsin Diagnostic) dataset.

Binary task: malignant (1) vs benign (0) from 30 fine-needle aspiration features.
This is the thin-slice dummy model — same shape as production models will follow.

Train the artifact: `python -m packages.ml_training.train_breast_cancer`
"""

from pathlib import Path

import joblib
import numpy as np
from pydantic import BaseModel, Field, create_model
from sklearn.datasets import load_breast_cancer

from packages.schemas.prediction import ModelMetadata, Prediction, XAIScore
from services.ml_mcp_server.explainers.shap_wrapper import compute_shap_values
from services.ml_mcp_server.models._base import MLModel


_DATASET = load_breast_cancer()
_FEATURE_NAMES: list[str] = list(_DATASET.feature_names)
_ARTIFACT_PATH = Path(__file__).parent.parent / "artifacts" / "breast_cancer_xgb.joblib"


def _sanitize(name: str) -> str:
    """sklearn feature names contain spaces; Pydantic field names cannot."""
    return name.replace(" ", "_")


# Build input schema dynamically from the real feature names.
BreastCancerInputs: type[BaseModel] = create_model(
    "BreastCancerInputs",
    **{
        _sanitize(name): (float, Field(description=f"FNA measurement: {name}"))
        for name in _FEATURE_NAMES
    },
)


class BreastCancerXGB(MLModel):
    """XGBoost binary classifier for breast tumor malignancy."""

    @property
    def name(self) -> str:
        return "predict_breast_cancer_malignancy"

    def __init__(self) -> None:
        if not _ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact missing: {_ARTIFACT_PATH}. "
                f"Run: python -m packages.ml_training.train_breast_cancer"
            )
        bundle = joblib.load(_ARTIFACT_PATH)
        self._model = bundle["model"]
        self._test_accuracy: float = bundle["test_accuracy"]
        self._test_f1: float | None = bundle.get("test_f1")

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self.name,
            version="0.1.0",
            description=(
                "Binary classifier for breast tumor malignancy from 30 fine-needle "
                "aspiration features (cell nucleus measurements: radius, texture, "
                "perimeter, area, smoothness, compactness, concavity, etc., aggregated "
                "as mean / standard error / worst). Trained on the Wisconsin Diagnostic "
                "Breast Cancer dataset (n=569)."
            ),
            trained_on="sklearn.datasets.load_breast_cancer (Wisconsin Diagnostic, n=569)",
            test_accuracy=self._test_accuracy,
            test_f1=self._test_f1,
            feature_count=len(_FEATURE_NAMES),
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return BreastCancerInputs

    def predict(self, inputs: BaseModel) -> Prediction:
        data = inputs.model_dump()
        X = np.array([[data[_sanitize(name)] for name in _FEATURE_NAMES]])

        # sklearn breast_cancer convention: target=0 is malignant, target=1 is benign.
        proba = self._model.predict_proba(X)[0]
        p_malignant = float(proba[0])
        p_benign = float(proba[1])
        predicted_class = "malignant" if p_malignant >= 0.5 else "benign"
        confidence = max(p_malignant, p_benign)

        # SHAP — wrapper returns positive-class values.
        shap_vals = compute_shap_values(self._model, X)
        all_scores = [
            XAIScore(
                feature_name=_FEATURE_NAMES[i],
                contribution=float(shap_vals[0, i]),
                feature_value=float(X[0, i]),
            )
            for i in range(len(_FEATURE_NAMES))
        ]
        all_scores.sort(key=lambda s: abs(s.contribution), reverse=True)

        return Prediction(
            model_name=self.name,
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities={"malignant": p_malignant, "benign": p_benign},
            xai_scores=all_scores[:5],
            metadata=self.metadata,
        )
