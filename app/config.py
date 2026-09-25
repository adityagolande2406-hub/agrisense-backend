"""
AgriSense Backend — Configuration
All settings are loaded from environment / .env file.
No values are hardcoded here.
"""
import json
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ───────────────────────────────────────────────────────────────
    # Default: SQLite (no server needed for local dev/testing)
    # Production: set DATABASE_URL=postgresql+asyncpg://user:pass@host/db
    DATABASE_URL: str = "sqlite+aiosqlite:///./agrisense_dev.db"

    # ── Security ───────────────────────────────────────────────────────────────
    SECRET_KEY: str = "dev-secret-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days

    # ── Feature Flags ─────────────────────────────────────────────────────────
    # When True: mock AI inference is used (development / demo mode).
    # When False: real inference service is called (production mode).
    USE_MOCK_INFERENCE: bool = True

    # When True: the PyTorch model runs embedded in the FastAPI process.
    # Use this for cloud deployments (Render, Railway, Fly.io) where running
    # a separate microservice is not feasible.
    # When False and USE_MOCK_INFERENCE=False: calls INFERENCE_SERVICE_URL (local dev).
    USE_EMBEDDED_INFERENCE: bool = False

    # ── Inference Service ─────────────────────────────────────────────────────
    # URL of the separate AI inference microservice.
    # Only used when USE_MOCK_INFERENCE=False AND USE_EMBEDDED_INFERENCE=False.
    INFERENCE_SERVICE_URL: str = "http://localhost:8001/infer"
    INFERENCE_TIMEOUT_SECONDS: int = 60

    # ── Severity Thresholds ───────────────────────────────────────────────────
    # Externally configurable — do NOT hardcode agricultural thresholds.
    # These will be tuned after the real model + crop scope is selected.
    SEVERITY_HIGH_THRESHOLD: float = 60.0
    SEVERITY_MEDIUM_THRESHOLD: float = 30.0

    # ── App ───────────────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    APP_NAME: str = "AgriSense"
    CORS_ORIGINS: List[str] = ["*"]

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def using_mock_inference(self) -> bool:
        """True when running with mock AI (not real model)."""
        return self.USE_MOCK_INFERENCE


settings = Settings()
