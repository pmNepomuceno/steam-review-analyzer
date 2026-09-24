from enum import StrEnum

APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"
APPREVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"

STEAM_PAGE_SIZE = 100  # max num_per_page Steam accepts
STEAM_LANGUAGE = "english"

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = 1.0  # doubled after each failed attempt

INGEST_LOCK_NAMESPACE = 1  # first key of the two-int Postgres advisory lock
INSERT_CHUNK_SIZE = 500


class Skipped(StrEnum):
    """Why `service.process_reviews` did nothing for an app."""

    NOT_INGESTED = "not ingested"
    ALREADY_DONE = "already processed; --force redoes it"
    IN_PROGRESS = "another worker is processing it; --force takes it over if that worker died"
    FAILED = "last run failed (see games.aspects_error); --force retries it"
