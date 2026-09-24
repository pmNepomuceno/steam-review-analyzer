class AppError(Exception):
    """Base class for domain errors; `main.py` turns these into JSON responses."""

    status_code = 500

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail
