"""
main.py
=======
FastAPI application entry point.

Start with:
    uvicorn fraud_phishing.api.main:app --reload --host 0.0.0.0 --port 8000
or from repo root:
    python -m uvicorn src.fraud_phishing.api.main:app --reload --port 8000
"""

import pathlib
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from fraud_phishing.api.schemas import (
    HealthResponse,
    MessageAnalysisResponse,
    MessageRequest,
    UrlAnalysisResponse,
    UrlRequest,
)
from fraud_phishing.api.services import registry

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load_models()
    yield


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud & Phishing Detection API",
    description=(
        "College-project prototype for fraud and phishing analysis with "
        "Explainable AI (XAI). It is an educational demonstration, not a "
        "production security control. Course: CSET312, Bennett University."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS – allow only the local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev
        "http://localhost:3000",   # CRA dev
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)




@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """
    Returns the application status and whether both trained pipelines
    have loaded successfully.
    """
    return HealthResponse(
        status="ok",
        message_model_loaded=registry.message_loaded,
        url_model_loaded=registry.url_loaded,
        message_dataset_provenance=registry.message_dataset_provenance,
        prototype=registry.prototype,
    )


@app.post(
    "/api/analyze/message",
    response_model=MessageAnalysisResponse,
    tags=["Analysis"],
    summary="Analyze a text message for phishing / fraud",
)
def analyze_message(request: MessageRequest) -> MessageAnalysisResponse:
    """
    Accepts a raw text message.

    Returns:
    - Message-level phishing classification and SHAP explanation.
    - Independent URL-level analysis for any URLs detected in the message.

    Message and URL predictions are kept strictly separate.
    The submitted message is not stored.
    """
    if not registry.message_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Message model is not loaded. "
                "Run scripts/train_message_model.py to train the model first."
            ),
        )
    try:
        return registry.analyze_message(request.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@app.post(
    "/api/analyze/url",
    response_model=UrlAnalysisResponse,
    tags=["Analysis"],
    summary="Analyze a URL for phishing",
)
def analyze_url(request: UrlRequest) -> UrlAnalysisResponse:
    """
    Accepts a raw URL string.

    Returns:
    - Phishing classification.
    - Extracted URL features (33 numeric/boolean features).
    - SHAP explanation of the model's decision.

    The URL destination is NEVER visited, fetched, or resolved.
    The submitted URL is not stored.
    """
    if not registry.url_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "URL model is not loaded. "
                "Run scripts/train_url_model.py to train the model first."
            ),
        )
    try:
        return registry.analyze_url(request.url)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")
