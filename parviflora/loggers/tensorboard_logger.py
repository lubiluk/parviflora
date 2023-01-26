from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from .base_logger import BaseLogger


class TensorboardLogger(BaseLogger):
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def log_msg(self, msg: str) -> None:
        print(msg)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        self.writer.add_scalar(name, value, step)

    def open(self) -> None:
        self.writer = SummaryWriter()

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
