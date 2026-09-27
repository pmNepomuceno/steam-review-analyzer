import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.database import SessionLocal
from src.exceptions import AppError
from src.games.router import router as games_router
from src.reviews.router import router as reviews_router
from src.reviews.service import fail_interrupted_runs

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Best effort: Neon may still be waking up when Render starts the container, and
    # requests recover a dead run on their own (`unprocessed_response`).
    try:
        async with SessionLocal() as session:
            await fail_interrupted_runs(session)
    except Exception:
        logger.exception("Could not check for interrupted runs at startup")
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(20.0), headers={"User-Agent": "steam-review-analyzer/0.1"}
    ) as http:
        app.state.http = http
        yield


app = FastAPI(title="Steam Review Analyzer", lifespan=lifespan)
app.include_router(games_router)
app.include_router(reviews_router)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
