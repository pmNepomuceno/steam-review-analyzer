"""Builders for fake Steam API responses."""

import httpx
import respx

from src.reviews import constants

APPREVIEWS_RE = r"https://store\.steampowered\.com/appreviews/\d+.*"


def make_reviews(n: int) -> list[dict]:
    return [
        {
            "recommendationid": str(1000 + i),
            "review": f"Review number {i}. Runs fine, price is ok.",
            "voted_up": i % 3 != 0,
            "votes_up": i,
            "language": "english",
            "timestamp_created": 1_700_000_000 + i,
            "author": {"playtime_forever": 60 * i},
        }
        for i in range(n)
    ]


def mock_steam(
    router: respx.MockRouter, appid: int, n_reviews: int, name: str = "Test Game"
) -> tuple[respx.Route, respx.Route]:
    """Mock appdetails plus cursor-paginated appreviews; returns (appdetails, appreviews)."""
    details = router.get(constants.APPDETAILS_URL).mock(
        return_value=httpx.Response(
            200, json={str(appid): {"success": True, "data": {"name": name}}}
        )
    )
    reviews = make_reviews(n_reviews)

    def page(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params["cursor"]
        start = 0 if cursor == "*" else int(cursor)
        chunk = reviews[start : start + constants.STEAM_PAGE_SIZE]
        return httpx.Response(
            200, json={"success": 1, "reviews": chunk, "cursor": str(start + len(chunk))}
        )

    return details, router.get(url__regex=APPREVIEWS_RE).mock(side_effect=page)
