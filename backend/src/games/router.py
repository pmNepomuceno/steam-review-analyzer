from typing import Annotated

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.games import service
from src.games.constants import AspectStatus
from src.games.dependencies import valid_appid
from src.games.schemas import UNPROCESSED_RESPONSES, AspectSummary
from src.reviews import service as reviews_service
from src.reviews.dependencies import get_http_client

router = APIRouter(prefix="/games/{appid}", tags=["games"])


@router.get("/aspects", response_model=AspectSummary, responses=UNPROCESSED_RESPONSES)
async def get_aspects(
    appid: Annotated[int, Depends(valid_appid)],
    # scope="function" closes the session when this returns, before the response and any
    # background processing run, so the request's connection isn't held for that run.
    session: Annotated[AsyncSession, Depends(get_session, scope="function")],
    http: Annotated[httpx.AsyncClient, Depends(get_http_client)],
) -> AspectSummary | JSONResponse:
    await reviews_service.ensure_ingested(session, http, appid)
    game = await service.get_game(session, appid)
    if game.aspects_status != AspectStatus.DONE:
        return reviews_service.unprocessed_response(session.bind, game)
    return AspectSummary(**await reviews_service.aspect_summary(session, appid))
