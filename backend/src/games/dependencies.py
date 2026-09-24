from fastapi import Path

from src.games.constants import MAX_APPID


def valid_appid(appid: int = Path(gt=0, le=MAX_APPID)) -> int:
    return appid
