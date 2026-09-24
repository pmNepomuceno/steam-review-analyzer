from src.exceptions import AppError


class GameNotFound(AppError):
    status_code = 404

    def __init__(self, appid: int):
        super().__init__(f"Steam has no app with appid {appid}")
