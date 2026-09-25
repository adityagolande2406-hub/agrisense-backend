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
    logger.info(
        "Mode: %s | Mock inference: %s | Embedded: %s",
        settings.APP_ENV, settings.USE_MOCK_INFERENCE, settings.USE_EMBEDDED_INFERENCE,
    )

    # ── Always create DB tables (create_all is idempotent) ───────────────────
    # This runs in both development AND production so Render's ephemeral
    # SQLite DB always has the correct schema on every cold start.
    async with engine.begin() as conn:
        from app.models import (  # noqa: F401
            User, Farm, Field, Crop, Observation,
            Prediction, SeverityRecord, Advisory, Alert
        )
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema verified (create_all complete).")

    # ── Seed essential crops (idempotent) ────────────────────────────────────
    from app.database import AsyncSessionLocal
    from sqlalchemy import select
    async with AsyncSessionLocal() as session:
        crops_exist = (await session.execute(select(Crop))).first()
        if not crops_exist:
            seed_crops = [
                Crop(id="grape",      name="Grape",      scientific_name="Vitis vinifera"),
                Crop(id="grapes",     name="Grapes",     scientific_name="Vitis vinifera"),
                Crop(id="tomato",     name="Tomato",     scientific_name="Solanum lycopersicum"),
                Crop(id="wheat",      name="Wheat",      scientific_name="Triticum aestivum"),
                Crop(id="rice",       name="Rice",       scientific_name="Oryza sativa"),
                Crop(id="cotton",     name="Cotton",     scientific_name="Gossypium hirsutum"),
                Crop(id="maize",      name="Maize",      scientific_name="Zea mays"),
                Crop(id="potato",     name="Potato",     scientific_name="Solanum tuberosum"),
                Crop(id="sugarcane",  name="Sugarcane",  scientific_name="Saccharum officinarum"),
            ]
            session.add_all(seed_crops)
            await session.commit()
            logger.info("Crop seed data inserted (%d crops).", len(seed_crops))

    # ── Development-only seed (demo user/farm/field) ─────────────────────────
    if settings.is_development:
        async with AsyncSessionLocal() as session:
            from app.models import User, Farm, Field
            demo_exists = (await session.execute(
                select(User).where(User.email == "farmer@agrisense.io")
            )).first()
            if not demo_exists:
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

    # ── Load embedded AI model (production cloud deployment) ─────────────────
    if settings.USE_EMBEDDED_INFERENCE:
        from app.services.embedded_inference import EmbeddedInferenceService
        svc = EmbeddedInferenceService()
        svc.load_model()
        if not EmbeddedInferenceService._is_ready:
            logger.error("Embedded model failed to load: %s", EmbeddedInferenceService._load_error)
        else:
            logger.info("Embedded AI model loaded and ready.")

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
