"""Steam HTTP client: fetches an app's name and its English reviews."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from src.games.exceptions import GameNotFound
from src.reviews import constants
from src.reviews.exceptions import SteamUnavailable

logger = logging.getLogger(__name__)


async def _get_json(http: httpx.AsyncClient, url: str, params: dict[str, Any]) -> Any:
    """GET with retries on transport errors, 429 and 5xx. Other failures raise immediately."""
    backoff = constants.RETRY_BACKOFF_S
    for attempt in range(1, constants.RETRY_ATTEMPTS + 1):
        try:
            resp = await http.get(url, params=params)
        except httpx.TransportError as exc:
            reason = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 429 or resp.status_code >= 500:
                reason = f"HTTP {resp.status_code}"
            elif resp.is_success:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise SteamUnavailable("Steam returned invalid JSON") from exc
            else:
                raise SteamUnavailable(f"Steam returned HTTP {resp.status_code}")

        logger.warning("Steam request failed (%s), attempt %d/%d", reason, attempt,
                       constants.RETRY_ATTEMPTS)
        if attempt < constants.RETRY_ATTEMPTS:
            await asyncio.sleep(backoff)
            backoff *= 2
    raise SteamUnavailable("Steam is unavailable or rate-limiting requests")


async def fetch_app_name(http: httpx.AsyncClient, appid: int) -> str:
    payload = await _get_json(
        http, constants.APPDETAILS_URL, {"appids": appid, "filters": "basic"}
    )
    entry = payload.get(str(appid)) if isinstance(payload, dict) else None
    if not entry or not entry.get("success"):
        raise GameNotFound(appid)
    name = (entry.get("data") or {}).get("name")
    if not name:
        raise GameNotFound(appid)
    return name


def parse_review(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Map one Steam review to a `reviews` row, or None if it is empty or malformed."""
    try:
        text = (raw.get("review") or "").strip()
        if not text:
            return None
        return {
            "id": int(raw["recommendationid"]),
            "review_text": text,
            "voted_up": bool(raw["voted_up"]),
            "votes_up": int(raw.get("votes_up", 0)),
            "playtime_forever": int((raw.get("author") or {}).get("playtime_forever", 0)),
            "language": raw.get("language") or constants.STEAM_LANGUAGE,
            "created_at": datetime.fromtimestamp(int(raw["timestamp_created"]), tz=UTC),
        }
    except (KeyError, TypeError, ValueError):
        return None


async def fetch_reviews(
    http: httpx.AsyncClient, appid: int, max_reviews: int, delay_s: float
) -> list[dict[str, Any]]:
    """Page through Steam's cursor-based review API until exhausted or `max_reviews` is hit."""
    url = constants.APPREVIEWS_URL.format(appid=appid)
    reviews: dict[int, dict[str, Any]] = {}
    seen_cursors = {"*"}
    cursor = "*"

    while len(reviews) < max_reviews:
        payload = await _get_json(
            http,
            url,
            {
                "json": 1,
                "filter": "recent",
                "language": constants.STEAM_LANGUAGE,
                "purchase_type": "all",
                "num_per_page": constants.STEAM_PAGE_SIZE,
                "cursor": cursor,
            },
        )
        batch = payload.get("reviews") if isinstance(payload, dict) else None
        if not batch:
            break
        for raw in batch:
            parsed = parse_review(raw)
            if parsed is not None:
                reviews.setdefault(parsed["id"], parsed)

        cursor = payload.get("cursor")
        if not cursor or cursor in seen_cursors:
            break
        seen_cursors.add(cursor)
        await asyncio.sleep(delay_s)

    return list(reviews.values())[:max_reviews]
