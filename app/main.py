"""
AgriSense FastAPI Backend — main application entry point.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine, Base
from app.routers import (
    auth_router,
    farms_router,
    fields_router,
    crops_router,
    observations_router,
    predict_router,
    alerts_router,
    dashboard_router,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agrisense")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AgriSense backend starting...")
    logger.info("Mode: %s | Mock inference: %s", settings.APP_ENV, settings.USE_MOCK_INFERENCE)
    if settings.USE_MOCK_INFERENCE:
        logger.warning(
            "[MOCK MODE ACTIVE] AI inference is mocked. "
            "Set USE_MOCK_INFERENCE=false and configure INFERENCE_SERVICE_URL for production."
        )
    # Create tables and seed data (dev only — production uses Alembic migrations)
    if settings.is_development:
        async with engine.begin() as conn:
            # Import all models so SQLAlchemy sees them
            from app.models import (  # noqa: F401
                User, Farm, Field, Crop, Observation,
                Prediction, SeverityRecord, Advisory, Alert
            )
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema verified.")

        from app.database import AsyncSessionLocal
        from sqlalchemy import select
        async with AsyncSessionLocal() as session:
            crops_exist = (await session.execute(select(Crop))).first()
            if not crops_exist:
                dev_crops = [
                    Crop(id="grape", name="Grape", scientific_name="Vitis vinifera"),
                    Crop(id="grapes", name="Grapes", scientific_name="Vitis vinifera"),
                    Crop(id="tomato", name="Tomato", scientific_name="Solanum lycopersicum"),
                    Crop(id="wheat", name="Wheat", scientific_name="Triticum aestivum"),
                ]
                session.add_all(dev_crops)
                dev_user = User(
                    full_name="Demo Farmer",
                    email="farmer@agrisense.io",
                    hashed_password="mock_password_hash",
                    preferred_language="en",
                )
                session.add(dev_user)
                await session.flush()
                dev_farm = Farm(
                    owner_id=dev_user.id,
                    name="Green Valley Farm",
                    location="Nashik, Maharashtra",
                    latitude=19.9975,
                    longitude=73.7898,
                )
                session.add(dev_farm)
                await session.flush()
                dev_field = Field(
                    farm_id=dev_farm.id,
                    crop_id="grape",
                    name="Vineyard North",
                    crop_stage="flowering",
                )
                session.add(dev_field)
                await session.commit()
                logger.info("Development seed data created.")
    yield
    await engine.dispose()
    logger.info("AgriSense backend stopped.")


app = FastAPI(
    title="AgriSense API",
    description=(
        "AgriSense crop health monitoring backend. "
        "Provides AI-powered disease detection, severity assessment, "
        "risk decision, and advisory generation for farmers."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router)
app.include_router(farms_router)
app.include_router(fields_router)
app.include_router(crops_router)
app.include_router(observations_router)
app.include_router(predict_router)
app.include_router(alerts_router)
app.include_router(dashboard_router)


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "service": "agrisense-backend",
        "version": "1.0.0",
        "mode": settings.APP_ENV,
        "mock_inference": settings.USE_MOCK_INFERENCE,
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "AgriSense API",
        "docs": "/docs",
        "health": "/health",
        "mock_mode": settings.USE_MOCK_INFERENCE,
    }
