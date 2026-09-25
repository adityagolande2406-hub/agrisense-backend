"""
AIInferenceService — Model-Agnostic Inference Abstraction.

This is the ONLY entry point for AI inference in AgriSense.
The rest of the system never calls a model directly.

Architecture:
    AIInferenceService (abstract interface)
    ├── MockInferenceService     — development / demo (USE_MOCK_INFERENCE=true)
    └── HttpInferenceService     — production: calls a separate inference microservice

The inference microservice (served separately) will run the actual model.
This design means:
  - The model can be YOLO, ResNet, EfficientNet, or any architecture
  - The model can be updated (v1 → v2) without changing the FastAPI backend
  - The mobile app never changes when the model changes
  - Segmentation support can be added by updating the inference service only

Usage:
    service = get_inference_service()
    result: PredictionResult = await service.predict(image_bytes, crop_id, crop_stage)
"""
from __future__ import annotations

import abc
import logging
from typing import Optional

from app.config import settings
from app.schemas import PredictionResult

logger = logging.getLogger(__name__)


# ── Abstract Interface ─────────────────────────────────────────────────────────

class AIInferenceService(abc.ABC):
    """
    Abstract base class for all AI inference implementations.

    Implementations must be stateless — all context is passed per-call.
    """

    @abc.abstractmethod
    async def predict(
        self,
        image_bytes: bytes,
        crop_id: str,
        crop_stage: Optional[str] = None,
    ) -> PredictionResult:
        """
        Run inference on the provided image.

        Args:
            image_bytes: Raw image bytes (JPEG or PNG).
            crop_id:     Crop identifier (e.g. 'wheat').
            crop_stage:  Optional growth stage (e.g. 'flowering').

        Returns:
            PredictionResult with disease, confidence, affected_region, model_version.

        Raises:
            InferenceError: if the inference service fails.
        """
        ...

    @property
    @abc.abstractmethod
    def mode(self) -> str:
        """Returns a string identifying the inference mode: 'mock' or 'http'."""
        ...


# ── Factory ───────────────────────────────────────────────────────────────────

def get_inference_service() -> AIInferenceService:
    """
    Returns the correct inference service based on configuration.

    When USE_MOCK_INFERENCE=true   → MockInferenceService (no model needed)
    When USE_EMBEDDED_INFERENCE=true → EmbeddedInferenceService (model in-process, for cloud)
    When USE_MOCK_INFERENCE=false  → HttpInferenceService (calls real model endpoint, for local dev)
    """
    if settings.USE_MOCK_INFERENCE:
        from app.services.mock_inference import MockInferenceService
        logger.warning(
            "[MOCK MODE] Using mock AI inference. "
            "Set USE_MOCK_INFERENCE=false to use a real model."
        )
        return MockInferenceService()
    elif settings.USE_EMBEDDED_INFERENCE:
        from app.services.embedded_inference import EmbeddedInferenceService
        logger.info("Using EMBEDDED inference service (in-process PyTorch model).")
        return EmbeddedInferenceService()
    else:
        from app.services.http_inference import HttpInferenceService
        logger.info(
            "Using HTTP inference service at: %s", settings.INFERENCE_SERVICE_URL
        )
        return HttpInferenceService(
            service_url=settings.INFERENCE_SERVICE_URL,
            timeout=settings.INFERENCE_TIMEOUT_SECONDS,
        )


# ── Custom Exceptions ──────────────────────────────────────────────────────────

class InferenceError(Exception):
    """Raised when the inference service fails."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class InferenceTimeoutError(InferenceError):
    """Raised when the inference service times out."""
    def __init__(self):
        super().__init__("Inference service timed out", status_code=504)


class InferenceUnavailableError(InferenceError):
    """Raised when the inference service cannot be reached."""
    def __init__(self, url: str):
        super().__init__(f"Inference service unavailable at {url}", status_code=503)
