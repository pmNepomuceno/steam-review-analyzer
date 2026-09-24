from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.games import service as games_service
from src.games.constants import AspectStatus
from src.games.dependencies import valid_appid
from src.games.schemas import UNPROCESSED_RESPONSES
from src.ml.anchors import ASPECTS
from src.ml.constants import NEGATIVE, POSITIVE
from src.reviews import service
from src.reviews.dependencies import get_http_client
from src.reviews.schemas import ReviewPage

router = APIRouter(prefix="/games/{appid}/reviews", tags=["reviews"])


@router.get("", response_model=ReviewPage, responses=UNPROCESSED_RESPONSES)
async def get_reviews(
    appid: Annotated[int, Depends(valid_appid)],
    # Closed when this returns, before any background processing run (see games/router.py).
    session: Annotated[AsyncSession, Depends(get_session, scope="function")],
    http: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    aspect: Annotated[Literal[ASPECTS] | None, Query()] = None,
    sentiment: Annotated[Literal[POSITIVE, NEGATIVE] | None, Query()] = None,
) -> ReviewPage | JSONResponse:
    await service.ensure_ingested(session, http, appid)
    if aspect is not None or sentiment is not None:
        game = await games_service.get_game(session, appid)
        if game.aspects_status != AspectStatus.DONE:
            return service.unprocessed_response(session.bind, game)
    total, items = await service.list_reviews(session, appid, limit, offset, aspect, sentiment)
    return ReviewPage(appid=appid, total=total, limit=limit, offset=offset, items=items)
