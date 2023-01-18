from abc import ABC, abstractmethod
from typing import Optional


class BaseLogger(ABC):
    @abstractmethod
    def __enter__(self):
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_traceback):
        ...

    @abstractmethod
    def log_msg(self, msg: str) -> None:
        ...

    @abstractmethod
    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        ...
