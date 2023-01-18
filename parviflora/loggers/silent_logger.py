from typing import Optional

from .base_logger import BaseLogger


class SilentLogger(BaseLogger):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def log_msg(self, msg: str) -> None:
        pass

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        pass
