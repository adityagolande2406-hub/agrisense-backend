"""
DecisionEngine — Deterministic, explainable risk assessment.

DESIGN PRINCIPLES:
  - Decisions are deterministic and testable (no LLM, no randomness)
  - Every decision has a recorded reason (decision_reason field)
  - Rules are NOT hardcoded agricultural facts — they are stubs
    waiting for validated agronomic rules after model+crop scope selection
  - The engine accepts: disease, confidence, severity, crop, crop_stage
  - Context (weather, historical trend) can be added later as optional inputs
  - Output: risk level (LOW | MEDIUM | HIGH) + decision reason

CURRENT STATE (Phase 1):
  Rule set is intentionally minimal — severity thresholds from config.
  Do NOT add agricultural rules that are not validated for the selected crop.

FUTURE STATE (after crop + disease scope selected):
  Add crop-specific, disease-specific rules validated by agronomists.
  Example structure shown in _load_rules() below.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.config import settings
from app.schemas import PredictionResult, SeverityResult, DecisionResult

logger = logging.getLogger(__name__)


@dataclass
class DecisionContext:
    """
    All inputs to the DecisionEngine.
    Add new context fields here as they become available.
    """
    prediction: PredictionResult
    severity: SeverityResult
    crop_id: Optional[str] = None
    crop_stage: Optional[str] = None
    # Reserved for future context inputs (weather, soil, historical trend)
    extra_context: dict = field(default_factory=dict)


class DecisionEngine:
    """
    Produces a risk level and decision reason from prediction + severity.

    Rules:
      1. If disease == 'healthy' → always LOW regardless of severity
      2. If severity >= HIGH_THRESHOLD → HIGH
      3. If severity >= MEDIUM_THRESHOLD → MEDIUM
      4. Otherwise → LOW

    These rules will be extended with crop+disease specific logic
    after the model and scope is selected.
    """

    def __init__(self):
        self._high_threshold = settings.SEVERITY_HIGH_THRESHOLD
        self._medium_threshold = settings.SEVERITY_MEDIUM_THRESHOLD

    def decide(self, context: DecisionContext) -> DecisionResult:
        """
        Applies decision rules and returns a risk level + explanation.
        Always produces a deterministic, explainable result.
        """
        prediction = context.prediction
        severity = context.severity.severity
        crop_id = context.crop_id
        crop_stage = context.crop_stage

        # Rule 1: Healthy prediction → always LOW risk
        if prediction.disease.lower() == "healthy":
            return DecisionResult(
                risk="LOW",
                decision_reason=(
                    f"Model predicted 'healthy' with confidence {prediction.confidence:.0%}. "
                    f"Severity score: {severity:.1f}. No intervention required."
                ),
                recommended_action_key="monitor_routine",
                monitoring_interval_days=14,
            )

        # Rule 2: Low confidence → downgrade risk by one level
        # Rationale: a low-confidence diseased prediction should not trigger HIGH alert
        confidence_penalty = prediction.confidence < 0.60

        # Rule 3: Severity-based risk (using configurable thresholds)
        if severity >= self._high_threshold:
            base_risk = "HIGH"
        elif severity >= self._medium_threshold:
            base_risk = "MEDIUM"
        else:
            base_risk = "LOW"

        # Apply confidence penalty
        effective_risk = self._downgrade_risk(base_risk) if confidence_penalty else base_risk

        reason = self._build_reason(
            disease=prediction.disease,
            confidence=prediction.confidence,
            severity=severity,
            base_risk=base_risk,
            effective_risk=effective_risk,
            confidence_penalty=confidence_penalty,
            crop_id=crop_id,
            crop_stage=crop_stage,
        )

        return DecisionResult(
            risk=effective_risk,
            decision_reason=reason,
            recommended_action_key=self._action_key(effective_risk),
            monitoring_interval_days=self._monitoring_days(effective_risk),
        )

    def _downgrade_risk(self, risk: str) -> str:
        """Reduces risk by one level due to low model confidence."""
        order = ["LOW", "MEDIUM", "HIGH"]
        idx = order.index(risk)
        return order[max(0, idx - 1)]

    def _action_key(self, risk: str) -> str:
        return {
            "HIGH": "act_immediately",
            "MEDIUM": "monitor_closely",
            "LOW": "monitor_routine",
        }[risk]

    def _monitoring_days(self, risk: str) -> int:
        # Placeholder intervals — to be validated per crop+disease
        return {"HIGH": 2, "MEDIUM": 5, "LOW": 14}[risk]

    def _build_reason(
        self,
        disease: str,
        confidence: float,
        severity: float,
        base_risk: str,
        effective_risk: str,
        confidence_penalty: bool,
        crop_id: Optional[str],
        crop_stage: Optional[str],
    ) -> str:
        parts = [
            f"Disease detected: '{disease}'.",
            f"Model confidence: {confidence:.0%}.",
            f"Severity score: {severity:.1f}/100.",
            f"Base risk from severity: {base_risk}.",
        ]
        if confidence_penalty:
            parts.append(
                f"Risk downgraded from {base_risk} to {effective_risk} "
                f"due to low confidence ({confidence:.0%} < 60%)."
            )
        if crop_id:
            parts.append(f"Crop: {crop_id}.")
        if crop_stage:
            parts.append(f"Stage: {crop_stage}.")
        parts.append(
            "NOTE: Rules are generic (Phase 1). "
            "Crop-specific validated rules will be added after model selection."
        )
        return " ".join(parts)
