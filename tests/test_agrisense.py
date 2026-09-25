"""
AgriSense Backend Tests.

Tests cover:
  - AIInferenceService (mock and http modes)
  - PredictionResult schema validation
  - SeverityEngine (all calculation methods)
  - DecisionEngine (all risk rules)
  - AdvisoryEngine (all risk levels)
  - TrendEngine
  - /predict API endpoint (mock mode)
  - /health endpoint

Run:
  cd backend
  pytest tests/ -v
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.schemas import (
    PredictionResult,
    SeverityResult,
    DecisionResult,
    AffectedRegionSegmentation,
)
from app.services.severity_engine import SeverityEngine
from app.services.decision_engine import DecisionEngine, DecisionContext
from app.services.advisory_engine import AdvisoryEngine
from app.services.trend_engine import TrendEngine
from app.services.mock_inference import MockInferenceService
from app.services.http_inference import HttpInferenceService
from app.services.ai_inference import InferenceError


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_prediction(
    disease: str = "disease_class_a",
    confidence: float = 0.85,
    is_mock: bool = True,
    affected_region=None,
) -> PredictionResult:
    return PredictionResult(
        disease=disease,
        disease_label=disease.replace("_", " ").title(),
        confidence=confidence,
        affected_region=affected_region,
        model_version="v0-mock",
        is_mock=is_mock,
    )


def make_severity(severity: float = 45.0, method: str = "mock") -> SeverityResult:
    return SeverityResult(severity=severity, calculation_method=method)


# ── PredictionResult Schema Tests ─────────────────────────────────────────────

class TestPredictionResult:
    def test_valid_prediction(self):
        p = make_prediction()
        assert p.disease == "disease_class_a"
        assert 0.0 <= p.confidence <= 1.0
        assert p.model_version == "v0-mock"
        assert p.is_mock is True

    def test_healthy_prediction(self):
        p = make_prediction(disease="healthy", confidence=0.95)
        assert p.disease == "healthy"

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            PredictionResult(
                disease="x", confidence=1.5,
                model_version="v1", is_mock=False
            )

    def test_affected_region_optional(self):
        p = make_prediction(affected_region=None)
        assert p.affected_region is None

    def test_affected_region_segmentation(self):
        region = AffectedRegionSegmentation(
            type="segmentation",
            mask_url="http://example.com/mask.png",
            affected_area_percent=32.5,
        )
        p = make_prediction(affected_region=region)
        assert p.affected_region.affected_area_percent == 32.5


# ── MockInferenceService Tests ────────────────────────────────────────────────

class TestMockInferenceService:
    @pytest.mark.asyncio
    async def test_returns_prediction_result(self):
        service = MockInferenceService()
        result = await service.predict(
            image_bytes=b"fake_image_data",
            crop_id="wheat",
            crop_stage="flowering",
        )
        assert isinstance(result, PredictionResult)
        assert result.is_mock is True
        assert result.model_version == "v0-mock"
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_mock_does_not_care_about_image(self):
        service = MockInferenceService()
        # Empty bytes — mock should still work
        result = await service.predict(image_bytes=b"", crop_id="rice")
        assert result.is_mock is True

    def test_mode(self):
        assert MockInferenceService().mode == "mock"


# ── HttpInferenceService Tests ────────────────────────────────────────────────

class TestHttpInferenceService:
    def test_mode(self):
        service = HttpInferenceService("http://localhost:9000")
        assert service.mode == "http"

    def test_parse_response_success(self):
        service = HttpInferenceService("http://localhost:9000")
        payload = {
            "disease": "Black Rot",
            "disease_label": "Grape Black Rot",
            "confidence": 0.88,
            "model_version": "grape-v1",
            "affected_region": None,
        }
        res = service._parse_response(payload)
        assert res.disease == "Black Rot"
        assert res.disease_label == "Grape Black Rot"
        assert res.confidence == 0.88
        assert res.model_version == "grape-v1"
        assert res.is_mock is False

    def test_parse_response_with_region(self):
        service = HttpInferenceService("http://localhost:9000")
        payload = {
            "disease": "Leaf Blight",
            "confidence": 0.76,
            "model_version": "grape-v1",
            "affected_region": {
                "type": "segmentation",
                "affected_area_percent": 15.2,
            },
        }
        res = service._parse_response(payload)
        assert res.affected_region is not None
        assert res.affected_region.affected_area_percent == 15.2

    def test_parse_response_invalid_raises(self):
        service = HttpInferenceService("http://localhost:9000")
        with pytest.raises(InferenceError):
            service._parse_response({"confidence": "not-a-number"})


# ── SeverityEngine Tests ──────────────────────────────────────────────────────

class TestSeverityEngine:
    def test_mock_healthy_is_low(self):
        engine = SeverityEngine()
        p = make_prediction(disease="healthy", is_mock=True)
        result = engine.compute(p)
        assert result.severity <= 8.0
        assert result.calculation_method == "mock"

    def test_mock_disease_is_mid_range(self):
        engine = SeverityEngine()
        p = make_prediction(disease="disease_class_a", is_mock=True)
        result = engine.compute(p)
        assert 15.0 <= result.severity <= 65.0
        assert result.calculation_method == "mock"

    def test_segmentation_area_method(self):
        engine = SeverityEngine()
        region = AffectedRegionSegmentation(affected_area_percent=42.0)
        p = make_prediction(affected_region=region, is_mock=False)
        result = engine.compute(p)
        assert result.severity == 42.0
        assert result.calculation_method == "segmentation_area"

    def test_confidence_proxy_method(self):
        engine = SeverityEngine()
        p = make_prediction(disease="disease_class_b", confidence=0.80, is_mock=False)
        result = engine.compute(p)
        # 0.80 * 70 = 56.0
        assert result.severity == pytest.approx(56.0, abs=0.1)
        assert result.calculation_method == "confidence_proxy"

    def test_confidence_proxy_healthy_caps_low(self):
        engine = SeverityEngine()
        p = make_prediction(disease="healthy", confidence=0.95, is_mock=False)
        result = engine.compute(p)
        assert result.severity <= 5.0

    def test_confidence_proxy_healthy_capitalized(self):
        engine = SeverityEngine()
        p = make_prediction(disease="Healthy", confidence=0.95, is_mock=False)
        result = engine.compute(p)
        assert result.severity <= 5.0

    def test_severity_capped_at_70_for_proxy(self):
        engine = SeverityEngine()
        p = make_prediction(disease="disease_class_a", confidence=1.0, is_mock=False)
        result = engine.compute(p)
        assert result.severity <= 70.0

    def test_risk_from_severity_high(self):
        engine = SeverityEngine()
        assert engine.risk_from_severity(70.0) == "HIGH"

    def test_risk_from_severity_medium(self):
        engine = SeverityEngine()
        assert engine.risk_from_severity(40.0) == "MEDIUM"

    def test_risk_from_severity_low(self):
        engine = SeverityEngine()
        assert engine.risk_from_severity(10.0) == "LOW"


# ── DecisionEngine Tests ──────────────────────────────────────────────────────

class TestDecisionEngine:
    def test_healthy_always_low(self):
        engine = DecisionEngine()
        ctx = DecisionContext(
            prediction=make_prediction(disease="healthy", confidence=0.95),
            severity=make_severity(severity=3.0),
        )
        result = engine.decide(ctx)
        assert result.risk == "LOW"
        assert "healthy" in result.decision_reason

    def test_healthy_capitalized_always_low(self):
        engine = DecisionEngine()
        ctx = DecisionContext(
            prediction=make_prediction(disease="Healthy", confidence=0.95),
            severity=make_severity(severity=3.0),
        )
        result = engine.decide(ctx)
        assert result.risk == "LOW"
        assert "healthy" in result.decision_reason.lower()

    def test_high_severity_gives_high_risk(self):
        engine = DecisionEngine()
        ctx = DecisionContext(
            prediction=make_prediction(confidence=0.90),
            severity=make_severity(severity=70.0),
        )
        result = engine.decide(ctx)
        assert result.risk == "HIGH"

    def test_medium_severity_gives_medium_risk(self):
        engine = DecisionEngine()
        ctx = DecisionContext(
            prediction=make_prediction(confidence=0.80),
            severity=make_severity(severity=40.0),
        )
        result = engine.decide(ctx)
        assert result.risk == "MEDIUM"

    def test_low_severity_gives_low_risk(self):
        engine = DecisionEngine()
        ctx = DecisionContext(
            prediction=make_prediction(confidence=0.75),
            severity=make_severity(severity=10.0),
        )
        result = engine.decide(ctx)
        assert result.risk == "LOW"

    def test_low_confidence_downgrades_risk(self):
        engine = DecisionEngine()
        ctx = DecisionContext(
            prediction=make_prediction(confidence=0.50),  # < 0.60 threshold
            severity=make_severity(severity=65.0),        # would be HIGH
        )
        result = engine.decide(ctx)
        # HIGH → MEDIUM due to low confidence
        assert result.risk == "MEDIUM"
        assert "downgraded" in result.decision_reason.lower()

    def test_decision_reason_always_present(self):
        engine = DecisionEngine()
        ctx = DecisionContext(
            prediction=make_prediction(confidence=0.80),
            severity=make_severity(severity=50.0),
        )
        result = engine.decide(ctx)
        assert result.decision_reason
        assert len(result.decision_reason) > 20


# ── AdvisoryEngine Tests ──────────────────────────────────────────────────────

class TestAdvisoryEngine:
    def _make_decision(self, risk: str) -> DecisionResult:
        return DecisionResult(
            risk=risk,
            decision_reason="test",
            monitoring_interval_days={"HIGH": 2, "MEDIUM": 5, "LOW": 14}[risk],
        )

    def test_low_risk_advisory(self):
        engine = AdvisoryEngine()
        result = engine.generate(
            decision=self._make_decision("LOW"),
            prediction=make_prediction(disease="healthy"),
            severity=make_severity(3.0),
        )
        assert result.advisory_text
        assert result.next_observation_days == 14

    def test_medium_risk_advisory(self):
        engine = AdvisoryEngine()
        result = engine.generate(
            decision=self._make_decision("MEDIUM"),
            prediction=make_prediction(),
            severity=make_severity(40.0),
        )
        assert "agronomist" in result.advisory_text.lower()
        assert result.next_observation_days == 5

    def test_high_risk_advisory(self):
        engine = AdvisoryEngine()
        result = engine.generate(
            decision=self._make_decision("HIGH"),
            prediction=make_prediction(),
            severity=make_severity(70.0),
        )
        # 'Immediate action is recommended' is in advisory_text; 'immediately' is in recommended_action
        assert "immediate" in result.advisory_text.lower()
        assert result.next_observation_days == 2

    def test_mock_source_label(self):
        engine = AdvisoryEngine()
        result = engine.generate(
            decision=self._make_decision("LOW"),
            prediction=make_prediction(is_mock=True),
            severity=make_severity(5.0),
        )
        assert result.advisory_source == "mock"

    def test_real_source_label(self):
        engine = AdvisoryEngine()
        result = engine.generate(
            decision=self._make_decision("MEDIUM"),
            prediction=make_prediction(is_mock=False),
            severity=make_severity(40.0),
        )
        assert result.advisory_source == "rule_based_generic"


# ── TrendEngine Tests ──────────────────────────────────────────────────────────

class TestTrendEngine:
    def test_stable_single_value(self):
        assert TrendEngine.compute([30.0]) == "STABLE"

    def test_stable_no_change(self):
        assert TrendEngine.compute([30.0, 31.0]) == "STABLE"

    def test_increasing(self):
        assert TrendEngine.compute([10.0, 20.0, 35.0]) == "INCREASING"

    def test_improving(self):
        assert TrendEngine.compute([50.0, 35.0, 20.0]) == "IMPROVING"

    def test_uses_last_3(self):
        # First two values push it toward increasing but last 3 are stable
        history = [5.0, 60.0, 30.0, 32.0, 33.0]
        assert TrendEngine.compute(history) == "STABLE"


# ── API Integration Tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "mock_inference" in data


@pytest.mark.asyncio
async def test_predict_endpoint_mock_mode():
    """
    Tests /predict endpoint end-to-end in mock mode.
    DB is SQLite (injected by conftest.py).
    Uses a minimal valid JPEG to pass image validation.
    """
    import io
    from PIL import Image

    # Create minimal valid JPEG bytes
    img = Image.new("RGB", (10, 10), color=(100, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/predict",
            files={"image": ("test.jpg", jpeg_bytes, "image/jpeg")},
            data={
                "farm_id": "1",
                "field_id": "1",
                "crop_id": "wheat",
                "crop_stage": "flowering",
            },
        )

    assert resp.status_code == 200
    data = resp.json()

    # Validate full response contract
    assert "disease" in data
    assert "confidence" in data
    assert "severity" in data
    assert "risk" in data
    assert "advisory" in data
    assert "model_version" in data
    assert data["is_mock"] is True
    assert data["risk"] in ("LOW", "MEDIUM", "HIGH")
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["severity"] <= 100.0


@pytest.mark.asyncio
async def test_predict_rejects_invalid_image():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/predict",
            files={"image": ("test.txt", b"not an image", "text/plain")},
            data={"farm_id": "1", "field_id": "1", "crop_id": "wheat"},
        )
    assert resp.status_code == 422
