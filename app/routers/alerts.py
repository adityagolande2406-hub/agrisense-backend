"""Alerts router."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Alert, Field, Farm
from app.schemas import AlertResponse

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("", response_model=list[AlertResponse])
async def list_alerts(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Alert).order_by(desc(Alert.created_at)).limit(50)
    )
    alerts = result.scalars().all()
    response = []
    for a in alerts:
        field_res = await db.execute(select(Field).where(Field.id == a.field_id))
        field = field_res.scalar_one_or_none()
        farm_res = await db.execute(select(Farm).where(Farm.id == a.farm_id))
        farm = farm_res.scalar_one_or_none()
        response.append(AlertResponse(
            id=a.id,
            observation_id=a.observation_id,
            field_id=a.field_id,
            farm_id=a.farm_id,
            disease=a.disease,
            risk=a.risk,
            severity=a.severity,
            status=a.status,
            created_at=a.created_at,
            field_name=field.name if field else None,
            farm_name=farm.name if farm else None,
        ))
    return response
