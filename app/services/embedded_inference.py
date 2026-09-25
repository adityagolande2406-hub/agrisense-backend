"""
EmbeddedInferenceService — runs the PyTorch ResNet18 model directly inside
the FastAPI process. No separate microservice needed for cloud deployment.

This implementation is used when INFERENCE_SERVICE_URL starts with 'embedded://'
or when USE_EMBEDDED_INFERENCE=true environment variable is set.

Architecture (Cloud):
    FastAPI backend
    └── EmbeddedInferenceService (same process)
        └── ResNet18 model (grape_disease_model.pth)

vs. local dev:
    FastAPI backend  ←HTTP→  Inference Microservice (port 9000)
"""
from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

from app.schemas import PredictionResult, AffectedRegionSegmentation
from app.services.ai_inference import AIInferenceService, InferenceError

logger = logging.getLogger(__name__)


class EmbeddedInferenceService(AIInferenceService):
    """
    Loads the PyTorch model in-process and runs inference directly.
    Used for cloud deployments where running a separate microservice
    is not feasible or desired.
    """

    _model = None
    _classes = None
    _transform = None
    _image_size = 224
    _model_version = "grape-v1"
    _is_ready = False
    _load_error = None

    DISEASE_LABELS: dict[str, str] = {
        "Black Rot": "Grape Black Rot",
        "ESCA": "Grape ESCA (Esca / Black Measles)",
        "Healthy": "Healthy Grape",
        "Leaf Blight": "Grape Leaf Blight (Isariopsis Leaf Spot)",
    }

    @property
    def mode(self) -> str:
        return "embedded"

    def load_model(self) -> None:
        """Load the model from disk. Called once at app startup."""
        model_path = Path(
            os.environ.get("MODEL_PATH", "model/grape_disease_model.pth")
        )

        if not model_path.exists():
            EmbeddedInferenceService._load_error = f"Model file not found: {model_path.resolve()}"
            logger.error(EmbeddedInferenceService._load_error)
            return

        try:
            import torch
            import torch.nn.functional as F
            from torchvision import models, transforms

            logger.info("Loading embedded model from: %s", model_path.resolve())
            t0 = time.time()
            device = torch.device("cpu")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            classes: list = checkpoint["classes"]
            image_size: int = checkpoint.get("image_size", 224)

            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            model.to(device)

            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            EmbeddedInferenceService._model = model
            EmbeddedInferenceService._classes = classes
            EmbeddedInferenceService._transform = transform
            EmbeddedInferenceService._image_size = image_size
            EmbeddedInferenceService._is_ready = True

            logger.info(
                "Embedded model ready in %.2fs | classes=%s | image_size=%d",
                time.time() - t0,
                classes,
                image_size,
            )
        except Exception as exc:
            EmbeddedInferenceService._load_error = f"Failed to load model: {exc}"
            logger.error(EmbeddedInferenceService._load_error, exc_info=True)

    async def predict(
        self,
        image_bytes: bytes,
        crop_id: str,
        crop_stage: Optional[str] = None,
    ) -> PredictionResult:
        if not EmbeddedInferenceService._is_ready:
            raise InferenceError(
                f"Embedded model not ready: {EmbeddedInferenceService._load_error}",
                status_code=503,
            )

        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image, UnidentifiedImageError

            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            t0 = time.time()

            input_tensor = self._transform(pil_image).unsqueeze(0)
            with torch.no_grad():
                output = self._model(input_tensor)
                probabilities = F.softmax(output, dim=1)[0]

            predicted_idx = int(torch.argmax(probabilities).item())
            predicted_class = self._classes[predicted_idx]
            confidence = float(probabilities[predicted_idx].item())

            logger.info(
                "Embedded inference | disease=%s confidence=%.4f | %.1fms | crop=%s",
                predicted_class,
                confidence,
                (time.time() - t0) * 1000,
                crop_id,
            )

            return PredictionResult(
                disease=predicted_class,
                disease_label=self.DISEASE_LABELS.get(predicted_class, predicted_class),
                confidence=round(confidence, 6),
                affected_region=None,
                model_version=self._model_version,
                is_mock=False,
            )

        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"Embedded inference failed: {exc}") from exc
