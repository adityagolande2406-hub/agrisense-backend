"""Observations router — save and retrieve observations."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Observation
from app.schemas import ObservationCreate, ObservationResponse

router = APIRouter(prefix="/observations", tags=["Observations"])


@router.post("", response_model=ObservationResponse, status_code=201)
async def create_observation(body: ObservationCreate, db: AsyncSession = Depends(get_db)):
    obs = Observation(
        farm_id=body.farm_id,
        field_id=body.field_id,
        crop_id=body.crop_id,
        crop_stage=body.crop_stage,
        disease=body.disease,
        disease_label=body.disease_label,
        confidence=body.confidence,
        severity=body.severity,
        risk=body.risk,
        trend=body.trend,
        model_version=body.model_version,
    )
    db.add(obs)
    await db.flush()
    # Update advisory text on observation
    if body.advisory:
        obs.decision_reason = body.advisory  # store in advisory field for now
    return ObservationResponse.model_validate(obs)


@router.get("/{observation_id}", response_model=ObservationResponse)
async def get_observation(observation_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Observation).where(Observation.id == observation_id))
    obs = result.scalar_one_or_none()
    if not obs:
        raise HTTPException(404, "Observation not found")
    return ObservationResponse.model_validate(obs)


@router.get("", response_model=list[ObservationResponse])
async def list_observations(
    field_id: int | None = None,
    farm_id: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    q = select(Observation).order_by(desc(Observation.observed_at))
    if field_id:
        q = q.where(Observation.field_id == field_id)
    if farm_id:
        q = q.where(Observation.farm_id == farm_id)
    result = await db.execute(q)
    return [ObservationResponse.model_validate(o) for o in result.scalars().all()]
