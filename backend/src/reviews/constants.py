APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"
APPREVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"

STEAM_PAGE_SIZE = 100  # max num_per_page Steam accepts
STEAM_LANGUAGE = "english"

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = 1.0  # doubled after each failed attempt

INGEST_LOCK_NAMESPACE = 1  # first key of the two-int Postgres advisory lock
INSERT_CHUNK_SIZE = 500
