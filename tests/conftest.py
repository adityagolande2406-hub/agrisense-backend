"""
Test configuration — overrides database to use SQLite for all tests.
No PostgreSQL server needed.
"""
import os
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Force SQLite before any app imports read the config
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_agrisense.db")
os.environ.setdefault("USE_MOCK_INFERENCE", "true")
os.environ.setdefault("APP_ENV", "development")

TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_agrisense.db"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

TestSessionLocal = async_sessionmaker(
    bind=test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def create_tables():
    """Create all tables in the test SQLite DB before any tests run."""
    from app.database import Base
    from app.models import (  # noqa: F401
        User, Farm, Field, Crop, Observation,
        Prediction, SeverityRecord, Advisory, Alert,
    )
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Teardown — drop all tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()
    # Remove SQLite file
    try:
        os.remove("./test_agrisense.db")
    except OSError:
        pass


@pytest_asyncio.fixture
async def db_session():
    """Provides a test DB session."""
    async with TestSessionLocal() as session:
        yield session


@pytest.fixture(autouse=True)
def override_db(db_session):
    """Overrides the FastAPI get_db dependency with the test SQLite session."""
    from app.main import app
    from app.database import get_db

    async def _get_test_db():
        yield db_session

    app.dependency_overrides[get_db] = _get_test_db
    yield
    app.dependency_overrides.clear()
