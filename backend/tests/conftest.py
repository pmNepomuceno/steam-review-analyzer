import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
import respx
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

import src.models  # noqa: F401  (registers tables on Base.metadata)
from src.config import settings
from src.database import Base, get_session
from src.main import app
from src.reviews import constants
from src.reviews.dependencies import get_http_client

TEST_DB = "steam_reviews_test"
TEST_URL = make_url(settings.database_url).set(database=TEST_DB)


@pytest.fixture(scope="session", autouse=True)
def _create_test_database() -> None:
    async def create() -> None:
        admin = create_async_engine(
            TEST_URL.set(database="postgres"), isolation_level="AUTOCOMMIT", poolclass=NullPool
        )
        async with admin.connect() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": TEST_DB}
            )
            if not exists:
                await conn.execute(text(f'CREATE DATABASE "{TEST_DB}"'))
        await admin.dispose()

    asyncio.run(create())


@pytest.fixture(autouse=True)
def _fast_steam(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "steam_request_delay_s", 0)
    monkeypatch.setattr(constants, "RETRY_BACKOFF_S", 0)


@pytest_asyncio.fixture
async def engine() -> AsyncIterator:
    engine = create_async_engine(TEST_URL, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def steam() -> respx.MockRouter:
    """Intercepts outgoing Steam calls; the ASGI test client is unaffected."""
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest_asyncio.fixture
async def client(engine, steam) -> AsyncIterator[httpx.AsyncClient]:
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def override_session():
        async with sessions() as session:
            yield session

    async with httpx.AsyncClient() as steam_http:
        app.dependency_overrides[get_session] = override_session
        app.dependency_overrides[get_http_client] = lambda: steam_http
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    app.dependency_overrides.clear()
