"""
HttpInferenceService — production implementation of AIInferenceService.

Calls a SEPARATE inference microservice that hosts the actual ML model.
This design means:
  - The model (YOLO / ResNet / EfficientNet / etc.) lives in its own process
  - The model can be upgraded without touching this backend
  - The model can run on a GPU server while this backend runs on CPU
  - The mobile app never needs to change when the model changes

Expected inference microservice contract (POST /infer):
  Request:  multipart/form-data
    image       — image bytes
    crop_id     — crop identifier
    crop_stage  — (optional) crop growth stage

  Response: application/json
    {
      "disease": "...",
      "disease_label": "...",
      "confidence": 0.91,
      "affected_region": null | { "type": "segmentation", ... },
      "model_version": "v1"
    }

The inference microservice is responsible for:
  - Loading and running the model
  - Image preprocessing
  - Returning the raw model output

It is NOT responsible for:
  - Severity calculation
  - Risk assessment
  - Advisory generation
  - Database storage

Those responsibilities belong to this backend.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.schemas import PredictionResult, AffectedRegionSegmentation
from app.services.ai_inference import (
    AIInferenceService,
    InferenceError,
    InferenceTimeoutError,
    InferenceUnavailableError,
)

logger = logging.getLogger(__name__)


class HttpInferenceService(AIInferenceService):
    """
    Production inference service — delegates to a remote inference microservice.
    """

    def __init__(self, service_url: str, timeout: int = 60):
        self._service_url = service_url.rstrip("/")
        self._timeout = timeout

    @property
    def mode(self) -> str:
        return "http"

    async def predict(
        self,
        image_bytes: bytes,
        crop_id: str,
        crop_stage: Optional[str] = None,
    ) -> PredictionResult:
        """
        Sends image to the inference microservice and returns a PredictionResult.
        """
        logger.info(
            "[HTTP INFERENCE] Calling inference service: %s | crop=%s stage=%s",
            self._service_url,
            crop_id,
            crop_stage,
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                files = {"image": ("crop_image.jpg", image_bytes, "image/jpeg")}
                data: dict = {"crop_id": crop_id}
                if crop_stage:
                    data["crop_stage"] = crop_stage

                response = await client.post(
                    f"{self._service_url}/infer",
                    files=files,
                    data=data,
                )

                if response.status_code != 200:
                    raise InferenceError(
                        f"Inference service returned HTTP {response.status_code}: {response.text}",
                        status_code=response.status_code,
                    )

                payload = response.json()
                return self._parse_response(payload)

        except httpx.TimeoutException:
            raise InferenceTimeoutError()
        except httpx.ConnectError:
            raise InferenceUnavailableError(self._service_url)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"Unexpected inference error: {exc}") from exc

    def _parse_response(self, payload: dict) -> PredictionResult:
        """
        Parses the inference microservice response into a PredictionResult.
        Validates required fields and handles optional affected_region.
        """
        try:
            affected_region = None
            raw_region = payload.get("affected_region")
            if raw_region is not None:
                affected_region = AffectedRegionSegmentation(**raw_region)

            return PredictionResult(
                disease=payload["disease"],
                disease_label=payload.get("disease_label"),
                confidence=float(payload["confidence"]),
                affected_region=affected_region,
                model_version=payload.get("model_version", "unknown"),
                is_mock=False,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InferenceError(
                f"Invalid response from inference service: {exc}"
            ) from exc
