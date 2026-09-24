from src.exceptions import AppError


class InsufficientReviews(AppError):
    status_code = 422

    def __init__(self, appid: int, found: int, required: int):
        super().__init__(
            f"App {appid} has only {found} usable English reviews; at least {required} are required"
        )


class SteamUnavailable(AppError):
    status_code = 502

    def __init__(self, detail: str = "Steam did not return a usable response"):
        super().__init__(detail)
