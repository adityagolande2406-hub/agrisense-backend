"""
AgriSense ORM Models — SQLAlchemy 2.x declarative style.

Tables:
  users, farms, fields, crops, observations, predictions,
  severity_records, alerts, advisories

Design decisions:
  - Prediction is a separate table from Observation (clean AI boundary).
  - Severity is stored separately so the engine can be replaced independently.
  - Advisory is stored separately from prediction.
  - model_version is stored on every prediction row for future model upgrades.
  - affected_region_json allows segmentation metadata to be stored as JSON.
"""
import datetime
from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey,
    Integer, String, Text, JSON, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


# SQLite requires INTEGER PRIMARY KEY for AUTOINCREMENT; PostgreSQL uses BIGINT/BIGSERIAL.
PK_INT = BigInteger().with_variant(Integer, "sqlite")


# ── Timestamp mixin ──────────────────────────────────────────────────────────

class TimestampMixin:
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False
    )


# ── User ──────────────────────────────────────────────────────────────────────

class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(PK_INT, primary_key=True, autoincrement=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    mobile_number: Mapped[str | None] = mapped_column(String(20), nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    preferred_language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    farms: Mapped[list["Farm"]] = relationship("Farm", back_populates="owner", cascade="all, delete-orphan")


# ── Crop ──────────────────────────────────────────────────────────────────────

class Crop(Base):
    """
    Crop master data.
    Populated by backend; mobile app reads from /crops endpoint.
    The crop list is determined by the ML scope decision — do not hardcode.
    """
    __tablename__ = "crops"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # e.g. 'wheat'
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    scientific_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    fields: Mapped[list["Field"]] = relationship("Field", back_populates="crop")
    observations: Mapped[list["Observation"]] = relationship("Observation", back_populates="crop")


# ── Farm ──────────────────────────────────────────────────────────────────────

class Farm(TimestampMixin, Base):
    __tablename__ = "farms"

    id: Mapped[int] = mapped_column(PK_INT, primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    location: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)

    owner: Mapped["User"] = relationship("User", back_populates="farms")
    fields: Mapped[list["Field"]] = relationship("Field", back_populates="farm", cascade="all, delete-orphan")

    @property
    def field_count(self) -> int:
        return len(self.fields)

    @property
    def main_crop(self) -> str | None:
        if not self.fields:
            return None
        return self.fields[0].crop_id


# ── Field ──────────────────────────────────────────────────────────────────────

class Field(TimestampMixin, Base):
    __tablename__ = "fields"

    id: Mapped[int] = mapped_column(PK_INT, primary_key=True, autoincrement=True)
    farm_id: Mapped[int] = mapped_column(ForeignKey("farms.id", ondelete="CASCADE"), nullable=False)
    crop_id: Mapped[str] = mapped_column(ForeignKey("crops.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    crop_variety: Mapped[str | None] = mapped_column(String(128), nullable=True)
    crop_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)

    farm: Mapped["Farm"] = relationship("Farm", back_populates="fields")
    crop: Mapped["Crop"] = relationship("Crop", back_populates="fields")
    observations: Mapped[list["Observation"]] = relationship(
        "Observation", back_populates="field", cascade="all, delete-orphan",
        order_by="Observation.observed_at"
    )

    @property
    def current_risk(self) -> str | None:
        """Risk from the latest observation."""
        if not self.observations:
            return None
        latest = max(self.observations, key=lambda o: o.observed_at or datetime.datetime.min)
        return latest.risk

    @property
    def latest_severity(self) -> float | None:
        if not self.observations:
            return None
        latest = max(self.observations, key=lambda o: o.observed_at or datetime.datetime.min)
        return latest.severity


# ── Prediction ────────────────────────────────────────────────────────────────
# Raw AI model output — stored independently of severity/risk/advisory.
# This is what the AIInferenceService returns.

class Prediction(TimestampMixin, Base):
    """
    Stores the raw output from the AI inference service.
    Deliberately kept separate from business-logic outputs (severity, risk, advisory).

    Fields:
      disease         — disease key returned by the model (e.g. 'powdery_mildew')
      confidence      — model confidence 0.0–1.0
      affected_region — nullable JSON; supports segmentation metadata later:
                        { "type": "segmentation", "mask_url": "...", "affected_area_percent": 32 }
      model_version   — which model version produced this prediction (e.g. 'v1')
    """
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(PK_INT, primary_key=True, autoincrement=True)
    disease: Mapped[str] = mapped_column(String(128), nullable=False)
    disease_label: Mapped[str | None] = mapped_column(String(256), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    # Nullable — basic classification models won't provide this.
    # Segmentation models can populate it later without schema changes.
    affected_region: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, default="v1")

    observation: Mapped["Observation"] = relationship("Observation", back_populates="prediction", uselist=False)


# ── Severity Record ───────────────────────────────────────────────────────────
# Calculated independently from the raw prediction.
# The SeverityEngine owns this calculation.

class SeverityRecord(TimestampMixin, Base):
    """
    Stores computed severity score (0–100).
    Calculated by the SeverityEngine independently of the AI prediction.
    Calculation method is recorded so it can be audited / replaced.
    """
    __tablename__ = "severity_records"

    id: Mapped[int] = mapped_column(PK_INT, primary_key=True, autoincrement=True)
    severity: Mapped[float] = mapped_column(Float, nullable=False)
    # How was severity calculated? Records method for auditability.
    # e.g. 'mock', 'confidence_proxy', 'segmentation_area', 'validated_threshold'
    calculation_method: Mapped[str] = mapped_column(String(64), nullable=False, default="mock")

    observation: Mapped["Observation"] = relationship(
        "Observation", back_populates="severity_record", uselist=False
    )


# ── Advisory ──────────────────────────────────────────────────────────────────

class Advisory(TimestampMixin, Base):
    """
    Advisory generated by the AdvisoryEngine.
    Kept separate from prediction and severity.
    """
    __tablename__ = "advisories"

    id: Mapped[int] = mapped_column(PK_INT, primary_key=True, autoincrement=True)
    recommended_action: Mapped[str | None] = mapped_column(Text, nullable=True)
    monitoring_recommendation: Mapped[str | None] = mapped_column(Text, nullable=True)
    next_observation_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Free-form advisory text (used by mobile app)
    advisory_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Source: 'mock', 'rule_based', 'expert_system' etc.
    advisory_source: Mapped[str] = mapped_column(String(64), nullable=False, default="mock")

    observation: Mapped["Observation"] = relationship(
        "Observation", back_populates="advisory", uselist=False
    )


# ── Observation ───────────────────────────────────────────────────────────────
# Ties everything together: image + prediction + severity + risk + advisory.

class Observation(TimestampMixin, Base):
    """
    Central record for one crop health check event.
    References all component tables: Prediction, SeverityRecord, Advisory.

    Flow:
      Image → AIInferenceService → Prediction
           → SeverityEngine → SeverityRecord
           → DecisionEngine → risk, decision_reason
           → AdvisoryEngine → Advisory
           → Observation (assembled here)
    """
    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(PK_INT, primary_key=True, autoincrement=True)
    farm_id: Mapped[int] = mapped_column(ForeignKey("farms.id", ondelete="CASCADE"), nullable=False)
    field_id: Mapped[int] = mapped_column(ForeignKey("fields.id", ondelete="CASCADE"), nullable=False)
    crop_id: Mapped[str] = mapped_column(ForeignKey("crops.id"), nullable=False)
    crop_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    observed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.utcnow, nullable=False
    )

    # FK to Prediction (raw AI output)
    prediction_id: Mapped[int | None] = mapped_column(
        ForeignKey("predictions.id"), nullable=True
    )
    # FK to SeverityRecord
    severity_record_id: Mapped[int | None] = mapped_column(
        ForeignKey("severity_records.id"), nullable=True
    )
    # FK to Advisory
    advisory_id: Mapped[int | None] = mapped_column(
        ForeignKey("advisories.id"), nullable=True
    )

    # Denormalized summary fields (copied from child records for fast reads)
    disease: Mapped[str | None] = mapped_column(String(128), nullable=True)
    disease_label: Mapped[str | None] = mapped_column(String(256), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    severity: Mapped[float | None] = mapped_column(Float, nullable=True)

    # DecisionEngine output
    risk: Mapped[str | None] = mapped_column(String(32), nullable=True)  # LOW | MEDIUM | HIGH
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Trend (computed by TrendEngine from history)
    trend: Mapped[str | None] = mapped_column(String(32), nullable=True)  # IMPROVING | STABLE | INCREASING

    # Model version (copied from Prediction for fast access)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Relationships
    farm: Mapped["Farm"] = relationship("Farm")
    field: Mapped["Field"] = relationship("Field", back_populates="observations")
    crop: Mapped["Crop"] = relationship("Crop", back_populates="observations")
    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="observation")
    severity_record: Mapped["SeverityRecord"] = relationship("SeverityRecord", back_populates="observation")
    advisory: Mapped["Advisory"] = relationship("Advisory", back_populates="observation")


# ── Alert ──────────────────────────────────────────────────────────────────────

class Alert(TimestampMixin, Base):
    """
    Alert generated when DecisionEngine flags HIGH or MEDIUM risk.
    """
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(PK_INT, primary_key=True, autoincrement=True)
    observation_id: Mapped[int] = mapped_column(ForeignKey("observations.id", ondelete="CASCADE"), nullable=False)
    field_id: Mapped[int] = mapped_column(ForeignKey("fields.id", ondelete="CASCADE"), nullable=False)
    farm_id: Mapped[int] = mapped_column(ForeignKey("farms.id", ondelete="CASCADE"), nullable=False)
    disease: Mapped[str] = mapped_column(String(128), nullable=False)
    risk: Mapped[str] = mapped_column(String(32), nullable=False)
    severity: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")  # active | resolved

    observation: Mapped["Observation"] = relationship("Observation")
    field: Mapped["Field"] = relationship("Field")
    farm: Mapped["Farm"] = relationship("Farm")
