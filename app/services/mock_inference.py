"""
MockInferenceService — development/demo implementation of AIInferenceService.

IMPORTANT:
  - Only active when USE_MOCK_INFERENCE=true in .env
  - Does NOT look at the image bytes — returns a plausible random result
  - Results are clearly flagged with is_mock=True
  - Disease set is intentionally generic — NOT a real crop/disease scope
  - The real disease set will be defined when the model is selected

This class is the ONLY place mock AI behaviour lives in the backend.
"""
from __future__ import annotations

import logging
import random
from typing import Optional

from app.schemas import PredictionResult
from app.services.ai_inference import AIInferenceService

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Placeholder disease registry
# These are NOT the final disease classes — they exist only to demonstrate
# the data contract until the real model and crop scope is chosen.
# Replace this entire list once the real model is selected.
# ─────────────────────────────────────────────────────────────────────────────
_MOCK_DISEASE_POOL = [
    {
        "disease": "healthy",
        "disease_label": "Healthy",
        "confidence_range": (0.88, 0.99),
    },
    {
        "disease": "disease_class_a",
        "disease_label": "Disease Class A (Placeholder)",
        "confidence_range": (0.70, 0.95),
    },
    {
        "disease": "disease_class_b",
        "disease_label": "Disease Class B (Placeholder)",
        "confidence_range": (0.65, 0.90),
    },
    {
        "disease": "disease_class_c",
        "disease_label": "Disease Class C (Placeholder)",
        "confidence_range": (0.60, 0.85),
    },
]


class MockInferenceService(AIInferenceService):
    """
    Mock implementation — returns plausible random predictions.
    Used for development and demonstration when no real model is available.
    """

    @property
    def mode(self) -> str:
        return "mock"

    async def predict(
        self,
        image_bytes: bytes,
        crop_id: str,
        crop_stage: Optional[str] = None,
    ) -> PredictionResult:
        """
        Returns a mock prediction.
        The image is not examined — result is random from the mock pool.
        """
        logger.warning(
            "[MOCK INFERENCE] Returning mock result for crop='%s' stage='%s'. "
            "Image was NOT analysed. Set USE_MOCK_INFERENCE=false for real detection.",
            crop_id,
            crop_stage,
        )

        chosen = random.choice(_MOCK_DISEASE_POOL)
        lo, hi = chosen["confidence_range"]
        confidence = round(random.uniform(lo, hi), 4)

        return PredictionResult(
            disease=chosen["disease"],
            disease_label=chosen["disease_label"],
            confidence=confidence,
            affected_region=None,  # mock doesn't produce segmentation
            model_version="v0-mock",
            is_mock=True,
        )
