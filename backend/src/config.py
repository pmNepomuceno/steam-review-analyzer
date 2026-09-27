from pathlib import Path

from pydantic import PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env", extra="ignore", env_ignore_empty=True
    )

    database_url: str
    # Migrations only. Neon's pooled endpoint (DATABASE_URL) runs PgBouncer, which is not
    # meant for DDL; the direct endpoint is. Unset locally, where there is no pooler.
    database_url_direct: str | None = None
    min_review_count: int = 200
    max_reviews: int = 1000
    steam_request_delay_s: float = 0.5
    # Off in production (render.yaml): at Render's 0.1 CPU one game takes 14-19 min, so the
    # demo serves only games processed ahead of time and answers 403 for any other appid.
    allow_on_demand_processing: bool = True
    # Aspect encoder threads; unset follows the container's CPU quota (ml/aspects.py). 0 is
    # rejected rather than read as "unset" or passed on (onnxruntime would take all cores).
    encoder_threads: PositiveInt | None = None


settings = Settings()
