from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from .base_logger import BaseLogger


class TensorboardLogger(BaseLogger):
    def __enter__(self):
        self.writer = SummaryWriter()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.writer.flush()
        self.writer.close()

    def log_msg(self, msg: str) -> None:
        print(msg)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        self.writer.add_scalar(name, value, step)
