"""
/predict router — the core AgriSense inference endpoint.

Flow:
  POST /predict  (multipart/form-data: image + context)
    ↓
  Image validation + compression
    ↓
  AIInferenceService.predict()  →  PredictionResult
    ↓
  SeverityEngine.compute()      →  SeverityResult
    ↓
  TrendEngine.compute()         →  trend string (from field history)
    ↓
  DecisionEngine.decide()       →  DecisionResult (risk + reason)
    ↓
  AdvisoryEngine.generate()     →  AdvisoryResult
    ↓
  PredictResponse.assemble()    →  JSON response to Flutter

All engines are called in sequence. Each is independently testable.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.config import settings
from app.schemas import PredictResponse, PredictionResult
from app.services import (
    get_inference_service,
    SeverityEngine,
    DecisionEngine,
    DecisionContext,
    AdvisoryEngine,
    TrendEngine,
)

router = APIRouter(prefix="/predict", tags=["Prediction"])
logger = logging.getLogger(__name__)

# Max image size: 10 MB
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


async def _validate_and_read_image(image: UploadFile) -> bytes:
    """Validates image type and size, returns raw bytes."""
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid image type: {image.content_type}. Allowed: JPEG, PNG.",
        )
    image_bytes = await image.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image too large. Maximum size is 10 MB.",
        )
    # Verify it's actually a valid image
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File is not a valid image.",
        )
    return image_bytes


async def _get_field_severity_history(
    db: AsyncSession, field_id: int
) -> list[float]:
    """Fetches recent severity values for trend calculation."""
    from sqlalchemy import select, desc
    from app.models import Observation

    result = await db.execute(
        select(Observation.severity)
        .where(Observation.field_id == field_id)
        .where(Observation.severity.isnot(None))
        .order_by(desc(Observation.observed_at))
        .limit(5)
    )
    rows = result.scalars().all()
    return list(reversed(rows))  # oldest first for trend calculation


@router.post(
    "",
    response_model=PredictResponse,
    summary="Analyse a crop image and return disease prediction + risk assessment",
    description="""
    Accepts a crop image and context (farm, field, crop).
    Returns: disease prediction, confidence, severity, risk, advisory, trend.

    When USE_MOCK_INFERENCE=true (development):
      - Image is accepted but NOT analysed
      - A mock prediction is returned
      - Response includes is_mock=true

    When USE_MOCK_INFERENCE=false (production):
      - Image is sent to the configured inference microservice
      - Real model output is processed through the engine pipeline
    """,
)
async def predict(
    image: UploadFile = File(..., description="Crop image (JPEG or PNG, max 10 MB)"),
    farm_id: int = Form(...),
    field_id: int = Form(...),
    crop_id: str = Form(...),
    crop_stage: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
) -> PredictResponse:

    logger.info(
        "POST /predict | farm=%d field=%d crop=%s stage=%s | mock=%s",
        farm_id, field_id, crop_id, crop_stage, settings.USE_MOCK_INFERENCE,
    )

    # 1. Validate image
    image_bytes = await _validate_and_read_image(image)

    # 2. AI Inference
    inference_service = get_inference_service()
    try:
        prediction: PredictionResult = await inference_service.predict(
            image_bytes=image_bytes,
            crop_id=crop_id,
            crop_stage=crop_stage,
        )
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI inference failed: {exc}",
        )

    # 3. Severity Engine
    severity_engine = SeverityEngine()
    severity_result = severity_engine.compute(
        prediction=prediction,
        crop_id=crop_id,
        crop_stage=crop_stage,
    )

    # 4. Trend (from observation history)
    severity_history = await _get_field_severity_history(db, field_id)
    severity_history.append(severity_result.severity)  # include current
    trend = TrendEngine.compute(severity_history)

    # 5. Decision Engine
    decision_engine = DecisionEngine()
    decision_ctx = DecisionContext(
        prediction=prediction,
        severity=severity_result,
        crop_id=crop_id,
        crop_stage=crop_stage,
    )
    decision_result = decision_engine.decide(decision_ctx)

    # 6. Advisory Engine
    advisory_engine = AdvisoryEngine()
    advisory_result = advisory_engine.generate(
        decision=decision_result,
        prediction=prediction,
        severity=severity_result,
        crop_id=crop_id,
        crop_stage=crop_stage,
    )

    # 7. Assemble response
    response = PredictResponse.assemble(
        prediction=prediction,
        severity=severity_result,
        decision=decision_result,
        advisory=advisory_result,
        trend=trend,
    )

    logger.info(
        "Prediction complete | disease=%s confidence=%.3f severity=%.1f risk=%s trend=%s is_mock=%s",
        response.disease, response.confidence, response.severity,
        response.risk, response.trend, response.is_mock,
    )

    return response
