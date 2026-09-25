"""Farms, Fields, Crops routers."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import Farm, Field, Crop
from app.schemas import (
    FarmCreate, FarmResponse,
    FieldCreate, FieldResponse,
    CropResponse,
)

farms_router = APIRouter(prefix="/farms", tags=["Farms"])
fields_router = APIRouter(prefix="/fields", tags=["Fields"])
crops_router = APIRouter(prefix="/crops", tags=["Crops"])


# ── Farms ──────────────────────────────────────────────────────────────────────

@farms_router.get("", response_model=list[FarmResponse])
async def list_farms(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Farm).options(selectinload(Farm.fields)))
    farms = result.scalars().all()
    return [
        FarmResponse(
            id=f.id, name=f.name, location=f.location,
            latitude=f.latitude, longitude=f.longitude,
            field_count=f.field_count, main_crop=f.main_crop,
            current_health_status=None, owner_id=f.owner_id,
        )
        for f in farms
    ]


@farms_router.post("", response_model=FarmResponse, status_code=201)
async def create_farm(body: FarmCreate, db: AsyncSession = Depends(get_db)):
    farm = Farm(owner_id=1, **body.model_dump())  # TODO: get owner from JWT
    db.add(farm)
    await db.flush()
    return FarmResponse(
        id=farm.id, name=farm.name, location=farm.location,
        latitude=farm.latitude, longitude=farm.longitude,
        field_count=0, main_crop=None, current_health_status=None,
        owner_id=farm.owner_id,
    )


@farms_router.get("/{farm_id}", response_model=FarmResponse)
async def get_farm(farm_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Farm).where(Farm.id == farm_id).options(selectinload(Farm.fields))
    )
    farm = result.scalar_one_or_none()
    if not farm:
        raise HTTPException(404, "Farm not found")
    return FarmResponse(
        id=farm.id, name=farm.name, location=farm.location,
        latitude=farm.latitude, longitude=farm.longitude,
        field_count=farm.field_count, main_crop=farm.main_crop,
        current_health_status=None, owner_id=farm.owner_id,
    )


# ── Fields ──────────────────────────────────────────────────────────────────────

@fields_router.get("", response_model=list[FieldResponse])
async def list_fields(farm_id: int | None = None, db: AsyncSession = Depends(get_db)):
    q = select(Field).options(selectinload(Field.observations))
    if farm_id:
        q = q.where(Field.farm_id == farm_id)
    result = await db.execute(q)
    fields = result.scalars().all()
    return [
        FieldResponse(
            id=f.id, name=f.name, farm_id=f.farm_id, crop_id=f.crop_id,
            crop_variety=f.crop_variety, crop_stage=f.crop_stage,
            current_risk=f.current_risk, latest_severity=f.latest_severity,
        )
        for f in fields
    ]


@fields_router.post("", response_model=FieldResponse, status_code=201)
async def create_field(body: FieldCreate, db: AsyncSession = Depends(get_db)):
    field = Field(**body.model_dump())
    db.add(field)
    await db.flush()
    return FieldResponse(
        id=field.id, name=field.name, farm_id=field.farm_id,
        crop_id=field.crop_id, crop_variety=field.crop_variety,
        crop_stage=field.crop_stage, current_risk=None, latest_severity=None,
    )


@fields_router.get("/{field_id}", response_model=FieldResponse)
async def get_field(field_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Field).where(Field.id == field_id).options(selectinload(Field.observations))
    )
    field = result.scalar_one_or_none()
    if not field:
        raise HTTPException(404, "Field not found")
    return FieldResponse(
        id=field.id, name=field.name, farm_id=field.farm_id,
        crop_id=field.crop_id, crop_variety=field.crop_variety,
        crop_stage=field.crop_stage,
        current_risk=field.current_risk, latest_severity=field.latest_severity,
    )


@fields_router.get("/{field_id}/observations")
async def get_field_observations(field_id: int, db: AsyncSession = Depends(get_db)):
    from app.models import Observation
    from app.schemas import ObservationResponse
    result = await db.execute(
        select(Observation).where(Observation.field_id == field_id)
    )
    obs = result.scalars().all()
    return [ObservationResponse.model_validate(o) for o in obs]


# ── Crops ──────────────────────────────────────────────────────────────────────

@crops_router.get("", response_model=list[CropResponse])
async def list_crops(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Crop).where(Crop.is_active == True))  # noqa: E712
    return [CropResponse.model_validate(c) for c in result.scalars().all()]
