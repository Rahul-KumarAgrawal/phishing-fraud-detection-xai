"""
schemas.py
==========
Pydantic request/response schemas for the FastAPI backend.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ── Requests ─────────────────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10_000, description="Raw message text.")

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


class UrlRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=4_096, description="URL to analyze.")

    @field_validator("url")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("url must not be blank")
        return v


# ── Sub-schemas ───────────────────────────────────────────────────────────────

class FeatureContribution(BaseModel):
    feature: str
    # SHAP contribution to `explained_output`. Positive values push towards the
    # positive class (phishing / smishing). The old payload reused a field named
    # `value` for this, which read as "the feature's value".
    shap_value: float
    # The feature's actual value for this input: the numeric feature value for a
    # URL, the TF-IDF weight for a message token. Never a SHAP value.
    feature_value: Any = None
    direction: str   # "increases_risk" | "decreases_risk"
    description: Optional[str] = None


class ExplanationSchema(BaseModel):
    explanation_type: str
    # What the contributions add up to, and in which units. `units` is
    # "probability" when the explainer is anchored to the displayed score, and
    # "log_odds" for the exact LinearExplainer on a logistic model.
    units: str = "unknown"                      # "probability" | "log_odds"
    explained_output: str = ""                  # e.g. "P(phishing)", "P(smishing)"
    base_value: Optional[float] = None          # expected value of explained_output
    additivity_gap: Optional[float] = None      # |base + sum(shap) - explained_output|
    top_positive_features: list[FeatureContribution]
    top_negative_features: list[FeatureContribution]
    disclaimer: str


class UrlAnalysisResult(BaseModel):
    url: str
    label: int
    label_name: str          # "phishing" | "legitimate"
    score: float             # probability of phishing (if calibrated)
    threshold: float
    features: Optional[dict[str, Any]] = None
    explanation: Optional[ExplanationSchema] = None


# ── Responses ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    message_model_loaded: bool
    url_model_loaded: bool
    # Provenance of the data the message model was trained on. "real" means the
    # Mendeley SMS/Smishing dataset; "synthetic" means the 65-message
    # development placeholder, whose metrics must not be reported as results.
    message_dataset_provenance: str = "unavailable"
    # True when any loaded model was trained on non-real data.
    prototype: bool = False


class MessageAnalysisResponse(BaseModel):
    message_label: int
    message_label_name: str     # "smishing" | "ham"
    message_score: float
    message_threshold: float
    explanation: Optional[ExplanationSchema] = None
    detected_urls: list[UrlAnalysisResult] = Field(default_factory=list)
    disclaimer: str = (
        "This result is a model-based risk assessment, not a guarantee that a "
        "message or website is safe. Verify important requests through official channels."
    )


class UrlAnalysisResponse(BaseModel):
    url: str
    label: int
    label_name: str
    score: float
    threshold: float
    features: Optional[dict[str, Any]] = None
    explanation: Optional[ExplanationSchema] = None
    disclaimer: str = (
        "This result is a model-based risk assessment, not a guarantee that a "
        "website is safe. Verify important links through official channels."
    )
