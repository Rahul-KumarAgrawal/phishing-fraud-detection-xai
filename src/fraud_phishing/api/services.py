"""
services.py
===========
Business logic layer for the FastAPI backend.
Loads models once at startup and provides inference + explanation services.
"""

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from fraud_phishing.explain.message_explainer import explain_message
from fraud_phishing.explain.url_explainer import explain_url
from fraud_phishing.features.url_features import extract_urls_from_text
from fraud_phishing.models.message_model import load_model as load_message_model
from fraud_phishing.models.url_model import load_model as load_url_model

from .schemas import (
    ExplanationSchema,
    FeatureContribution,
    MessageAnalysisResponse,
    UrlAnalysisResponse,
    UrlAnalysisResult,
)


class ModelRegistry:
    """Holds loaded model pipelines. Singleton instantiated at app startup."""

    def __init__(self):
        self._message_pipeline = None
        self._message_metadata: dict = {}
        self._url_pipeline = None
        self._url_metadata: dict = {}
        self._message_loaded = False
        self._url_loaded = False

    def load_models(self) -> None:
        """Attempt to load both models. Errors are logged but not raised."""
        try:
            self._message_pipeline, self._message_metadata = load_message_model()
            self._message_loaded = True
            print("[services] Message model loaded.")
        except FileNotFoundError as exc:
            print(f"[services] Message model not available: {exc}")
        except Exception as exc:
            print(f"[services] Message model load error: {exc}")

        try:
            self._url_pipeline, self._url_metadata = load_url_model()
            self._url_loaded = True
            print("[services] URL model loaded.")
        except FileNotFoundError as exc:
            print(f"[services] URL model not available: {exc}")
        except Exception as exc:
            print(f"[services] URL model load error: {exc}")

    @property
    def message_loaded(self) -> bool:
        return self._message_loaded

    @property
    def url_loaded(self) -> bool:
        return self._url_loaded

    # ── Message analysis ──────────────────────────────────────────────────

    def analyze_message(self, message: str) -> MessageAnalysisResponse:
        if not self._message_loaded:
            raise RuntimeError(
                "Message model is not loaded. "
                "Run scripts/train_message_model.py first."
            )

        # Message-level SHAP explanation
        result = explain_message(
            text=message,
            pipeline=self._message_pipeline,
            metadata=self._message_metadata,
        )

        msg_explanation = _build_explanation(result)

        # URL analysis (independent pipeline)
        url_results: list[UrlAnalysisResult] = []
        urls = extract_urls_from_text(message)
        for url in urls[:10]:  # limit to 10 URLs per message
            url_result = self._analyze_single_url(url)
            if url_result is not None:
                url_results.append(url_result)

        return MessageAnalysisResponse(
            message_label=result["label"],
            message_label_name=result["label_name"],
            message_score=result["score"],
            message_threshold=result["threshold"],
            explanation=msg_explanation,
            detected_urls=url_results,
        )

    # ── URL analysis ──────────────────────────────────────────────────────

    def analyze_url(self, url: str) -> UrlAnalysisResponse:
        if not self._url_loaded:
            raise RuntimeError(
                "URL model is not loaded. "
                "Run scripts/train_url_model.py first."
            )
        result = self._analyze_single_url_full(url)
        if "error" in result:
            raise ValueError(result["error"])
        return UrlAnalysisResponse(**result)

    def _analyze_single_url(self, url: str) -> UrlAnalysisResult | None:
        """Used internally when extracting URLs from a message."""
        if not self._url_loaded:
            return None
        try:
            result = explain_url(
                url=url,
                pipeline=self._url_pipeline,
                metadata=self._url_metadata,
            )
            if "error" in result:
                return None
            explanation = _build_explanation(result)
            return UrlAnalysisResult(
                url=url,
                label=result["label"],
                label_name=result["label_name"],
                score=result["score"],
                threshold=result["threshold"],
                features=result.get("features"),
                explanation=explanation,
            )
        except Exception:
            return None

    def _analyze_single_url_full(self, url: str) -> dict[str, Any]:
        result = explain_url(
            url=url,
            pipeline=self._url_pipeline,
            metadata=self._url_metadata,
        )
        if "error" in result:
            return result
        explanation = _build_explanation(result)
        return {
            "url": url,
            "label": result["label"],
            "label_name": result["label_name"],
            "score": result["score"],
            "threshold": result["threshold"],
            "features": result.get("features"),
            "explanation": explanation,
        }


def _build_explanation(result: dict) -> ExplanationSchema | None:
    exp_type = result.get("explanation_type", "unavailable")
    if exp_type == "unavailable":
        return None

    def _to_fc(item: dict) -> FeatureContribution:
        return FeatureContribution(
            feature=item["feature"],
            value=item.get("value", item.get("shap_value")),
            shap_value=item.get("shap_value"),
            direction=item.get("direction", "unknown"),
            description=item.get("description"),
        )

    pos = [_to_fc(i) for i in result.get("top_positive_features", [])]
    neg = [_to_fc(i) for i in result.get("top_negative_features", [])]

    return ExplanationSchema(
        explanation_type=exp_type,
        top_positive_features=pos,
        top_negative_features=neg,
        disclaimer=result.get("disclaimer", ""),
    )


# Global singleton
registry = ModelRegistry()
