"""Schemas for ML model predictions and metadata.

Used by every MLModel implementation, the MCP server that exposes them, and
the orchestrator that consumes their results.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class XAIScore(BaseModel):
    """A single feature's contribution to a prediction.

    For tree models we use SHAP values: positive contribution pushes the
    prediction toward the positive class (e.g., malignant), negative pushes
    toward the negative class. The orchestrator can compare contributions
    across features to explain *why* a model decided what it did.
    """

    feature_name: str
    contribution: float = Field(description="Signed SHAP value for this feature")
    feature_value: float | int | str | None = Field(
        default=None, description="The actual input value of this feature for the patient"
    )


class ModelMetadata(BaseModel):
    """What the orchestrator needs to know about a model before calling it.

    The orchestrator reads this to decide *whether* to call the model and how
    much to weight its result against other models and the medical expert.
    """

    name: str
    version: str
    description: str
    trained_on: str = Field(description="Dataset name + brief description")
    test_accuracy: float = Field(ge=0.0, le=1.0)
    test_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    feature_count: int


class Prediction(BaseModel):
    """A single ML model's output, including XAI explanation and self-described metadata.

    Returned by every MLModel.predict() call. The MCP server serializes this
    as the tool's response payload.
    """

    model_name: str
    predicted_class: str | None = Field(
        default=None, description="For classifiers: the predicted class label"
    )
    predicted_value: float | None = Field(
        default=None, description="For regressors: the predicted scalar value"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="For classifiers: probability of the predicted class. For regressors: model-specific (e.g., inverse of prediction interval width).",
    )
    class_probabilities: dict[str, float] | None = Field(
        default=None, description="For classifiers: full distribution over classes"
    )
    xai_scores: list[XAIScore] = Field(
        description="Top features by absolute contribution. Order matters — most influential first."
    )
    metadata: ModelMetadata
    predicted_at: datetime = Field(default_factory=datetime.utcnow)
