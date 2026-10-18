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
    value: Any
    shap_value: Optional[float] = None
    direction: str   # "increases_risk" | "decreases_risk"
    description: Optional[str] = None


class ExplanationSchema(BaseModel):
    explanation_type: str
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
