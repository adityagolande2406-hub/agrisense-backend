"""
SeverityEngine — Computes crop disease severity independently of AI confidence.

DESIGN PRINCIPLES:
  - Severity (0–100) is NOT the same as AI confidence (0–1).
  - Thresholds are loaded from config — never hardcoded here.
  - The calculation method is recorded for auditability.
  - Multiple methods are supported; the correct one is chosen based on
    what data the prediction provides.

CURRENT METHODS (Phase 1 — pre-real-model):
  1. 'mock'               — returns a plausible mock severity for demo
  2. 'confidence_proxy'   — approximates severity from model confidence
                            (temporary; replace with validated method after model selection)
  3. 'segmentation_area'  — uses affected_area_percent from segmentation model
                            (will be available when a segmentation model is used)

FUTURE METHODS (to be added after model and crop scope selection):
  - 'validated_threshold' — crop+disease specific validated thresholds
  - 'expert_review'       — severity confirmed by agronomist

The method used is recorded in SeverityRecord.calculation_method so
historical records remain auditable even after the engine is upgraded.
"""
from __future__ import annotations

import logging
import random
from typing import Optional

from app.config import settings
from app.schemas import PredictionResult, SeverityResult

logger = logging.getLogger(__name__)


class SeverityEngine:
    """
    Calculates severity from prediction output.
    Does not depend on any specific model architecture.
    """

    def __init__(self):
        self._high_threshold = settings.SEVERITY_HIGH_THRESHOLD
        self._medium_threshold = settings.SEVERITY_MEDIUM_THRESHOLD

    def compute(
        self,
        prediction: PredictionResult,
        crop_id: Optional[str] = None,
        crop_stage: Optional[str] = None,
    ) -> SeverityResult:
        """
        Compute severity from a PredictionResult.

        Selection logic:
          - If prediction is mock → use mock method
          - If affected_region with area_percent → use segmentation_area method
          - Otherwise → use confidence_proxy (temporary until validated thresholds exist)
        """
        if prediction.is_mock:
            return self._mock_severity(prediction)

        if (
            prediction.affected_region is not None
            and prediction.affected_region.affected_area_percent is not None
        ):
            return self._segmentation_area_severity(prediction)

        # Fallback: confidence proxy
        # TODO: Replace with validated agronomic thresholds once model + crop scope is decided
        return self._confidence_proxy_severity(prediction)

    def _mock_severity(self, prediction: PredictionResult) -> SeverityResult:
        """
        Returns a plausible mock severity for development/demo.
        Healthy crops get low severity; others get random mid-range.
        """
        if prediction.disease.lower() == "healthy":
            severity = round(random.uniform(0.0, 8.0), 1)
        else:
            severity = round(random.uniform(15.0, 65.0), 1)

        return SeverityResult(severity=severity, calculation_method="mock")

    def _segmentation_area_severity(self, prediction: PredictionResult) -> SeverityResult:
        """
        Uses affected_area_percent from segmentation model as severity proxy.
        This is a direct mapping — 1% affected area = 1 severity point.
        TODO: Validate this mapping with agronomists once the real model is selected.
        """
        area = prediction.affected_region.affected_area_percent  # type: ignore[union-attr]
        severity = max(0.0, min(100.0, float(area)))
        return SeverityResult(severity=round(severity, 1), calculation_method="segmentation_area")

    def _confidence_proxy_severity(self, prediction: PredictionResult) -> SeverityResult:
        """
        Approximates severity from confidence for non-segmentation models.

        WARNING: This is a TEMPORARY method. Confidence and severity are
        fundamentally different concepts. This proxy exists only until
        validated agronomic severity thresholds are established for
        the selected crop+disease scope.

        Mapping:
          healthy     → always low severity regardless of confidence
          non-healthy → severity = confidence * 70 (caps at 70% until validated)
        """
        logger.warning(
            "[SEVERITY] Using confidence_proxy method — "
            "replace with validated thresholds after crop/model selection. "
            "disease=%s confidence=%.3f",
            prediction.disease,
            prediction.confidence,
        )

        if prediction.disease.lower() == "healthy":
            severity = round(prediction.confidence * 5.0, 1)  # max ~5% for healthy
        else:
            # Cap at 70 — intentionally conservative until real thresholds are defined
            severity = round(min(prediction.confidence * 70.0, 70.0), 1)

        return SeverityResult(severity=severity, calculation_method="confidence_proxy")

    def risk_from_severity(self, severity: float) -> str:
        """
        Converts severity score to risk level using configurable thresholds.
        Thresholds come from settings — NOT hardcoded.
        """
        if severity >= self._high_threshold:
            return "HIGH"
        if severity >= self._medium_threshold:
            return "MEDIUM"
        return "LOW"
