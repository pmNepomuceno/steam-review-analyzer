from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.games.dependencies import valid_appid
from src.reviews import service
from src.reviews.dependencies import get_http_client
from src.reviews.schemas import ReviewPage

router = APIRouter(prefix="/games/{appid}/reviews", tags=["reviews"])


@router.get("", response_model=ReviewPage)
async def get_reviews(
    appid: Annotated[int, Depends(valid_appid)],
    session: Annotated[AsyncSession, Depends(get_session)],
    http: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ReviewPage:
    await service.ensure_ingested(session, http, appid)
    total, items = await service.list_reviews(session, appid, limit, offset)
    return ReviewPage(appid=appid, total=total, limit=limit, offset=offset, items=items)
