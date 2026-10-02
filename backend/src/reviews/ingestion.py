"""Steam HTTP client: fetches an app's name and its English reviews."""

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from src.config import settings
from src.games.exceptions import GameNotFound
from src.reviews import constants
from src.reviews.exceptions import SteamUnavailable

logger = logging.getLogger(__name__)

# One pace for every Steam call in this process (ingestion pages, store search, the rating
# fetch, retries included): Steam's informal rate limit is per client, not per endpoint.
# Calls start one at a time, at least `steam_request_delay_s` apart, and may overlap.
_pace_lock = asyncio.Lock()
_last_call = 0.0  # time.monotonic() when the last Steam call started


async def _pace() -> None:
    global _last_call
    async with _pace_lock:
        wait = _last_call + settings.steam_request_delay_s - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call = time.monotonic()


def new_client() -> httpx.AsyncClient:
    """The Steam client settings: the API's shared client, and a processing run's own."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(20.0), headers={"User-Agent": "steam-review-analyzer/0.1"}
    )


async def _get_json(http: httpx.AsyncClient, url: str, params: dict[str, Any]) -> Any:
    """GET with retries on transport errors, 429 and 5xx. Other failures raise immediately."""
    backoff = constants.RETRY_BACKOFF_S
    for attempt in range(1, constants.RETRY_ATTEMPTS + 1):
        await _pace()
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
    # One appid is requested, so there is one entry. Steam no longer always keys it by that
    # appid (Hades, 1145360, comes back under 1206340), so fall back to the only entry, but
    # only if its data names the requested app.
    entry = payload.get(str(appid)) if isinstance(payload, dict) else None
    if entry is None and isinstance(payload, dict) and len(payload) == 1:
        [entry] = payload.values()
        if not isinstance(entry, dict) or (entry.get("data") or {}).get("steam_appid") != appid:
            entry = None
    if not isinstance(entry, dict) or not entry.get("success"):
        raise GameNotFound(appid)
    name = (entry.get("data") or {}).get("name")
    if not name:
        raise GameNotFound(appid)
    return name


async def fetch_review_summary(http: httpx.AsyncClient, appid: int) -> dict[str, Any] | None:
    """Steam's store rating for the app, as `games` columns: English reviews by Steam purchasers,
    all time, which is what a logged-out English visitor sees on the store page.

    None when Steam answers without a usable `query_summary`.
    """
    payload = await _get_json(
        http,
        constants.APPREVIEWS_URL.format(appid=appid),
        {"json": 1, "language": constants.STEAM_LANGUAGE, "purchase_type": "steam",
         "num_per_page": 0, "cursor": "*"},
    )
    summary = payload.get("query_summary") if isinstance(payload, dict) else None
    try:
        return {
            "steam_score_desc": str(summary["review_score_desc"]),
            "steam_positive": int(summary["total_positive"]),
            "steam_total": int(summary["total_reviews"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


async def search_store(http: httpx.AsyncClient, term: str) -> list[dict[str, Any]]:
    """Apps matching `term` (a name or an appid) in Steam's store search, best match first."""
    payload = await _get_json(
        http, constants.STORESEARCH_URL, {"term": term, "l": constants.STEAM_LANGUAGE, "cc": "us"}
    )
    items = payload.get("items") if isinstance(payload, dict) else None
    results = []
    for item in items or []:
        try:
            if item.get("type") == "app":
                results.append(
                    {"appid": int(item["id"]), "name": str(item["name"]),
                     "image": item.get("tiny_image")}
                )
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return results


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
    http: httpx.AsyncClient, appid: int, max_reviews: int
) -> list[dict[str, Any]]:
    """Page through Steam's cursor-based review API until exhausted or `max_reviews` is hit.

    Pages are paced by `_get_json`, like every other Steam call.
    """
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

    return list(reviews.values())[:max_reviews]
