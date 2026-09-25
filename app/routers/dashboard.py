"""Dashboard router — consolidated farm health summary."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import Farm, Observation, Alert
from app.schemas import DashboardSummary

router = APIRouter(prefix="/farm-health", tags=["Dashboard"])


@router.get("", response_model=DashboardSummary)
async def get_farm_health(db: AsyncSession = Depends(get_db)):
    """Returns the dashboard summary for the primary farm."""
    # Get first farm (MVP: single farm view)
    farm_res = await db.execute(
        select(Farm).options(selectinload(Farm.fields)).limit(1)
    )
    farm = farm_res.scalar_one_or_none()
    if not farm:
        return DashboardSummary(farm_name="No farms yet", location="")

    # Get latest observation
    obs_res = await db.execute(
        select(Observation)
        .where(Observation.farm_id == farm.id)
        .order_by(desc(Observation.observed_at))
        .limit(1)
    )
    latest_obs = obs_res.scalar_one_or_none()

    # Active alert count
    alert_res = await db.execute(
        select(Alert)
        .where(Alert.farm_id == farm.id)
        .where(Alert.status == "active")
    )
    alert_count = len(alert_res.scalars().all())

    return DashboardSummary(
        farm_name=farm.name,
        location=farm.location,
        main_crop=farm.main_crop,
        field_name=farm.fields[0].name if farm.fields else None,
        latest_disease=latest_obs.disease if latest_obs else None,
        confidence=latest_obs.confidence if latest_obs else None,
        severity=latest_obs.severity if latest_obs else None,
        risk=latest_obs.risk if latest_obs else None,
        trend=latest_obs.trend if latest_obs else None,
        active_alert_count=alert_count,
    )
