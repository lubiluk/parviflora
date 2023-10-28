from typing import Optional

import pandas as pd

from .base_logger import BaseLogger


class DataframeLogger(BaseLogger):
    def __init__(self) -> None:
        super().__init__()
        self._scalars = {}
        self._msgs = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def log_msg(self, msg: str) -> None:
        self._msgs.append(msg)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        if name not in self._scalars:
            self._scalars[name] = []

        self._scalars[name].append(value)

    @property
    def scalar_df(self):
        return pd.DataFrame(self._scalars)
