from pathlib import Path
from typing import Optional

import wandb

from .base_logger import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        save_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.project = project
        self.name = name
        self.config = config
        self.save_dir = str(save_dir) if save_dir is not None else None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def log_msg(self, msg: str) -> None:
        print(msg)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        self.run.log({name: value}, step=step)

    def open(self) -> None:
        self.run = wandb.init(
            project=self.project,
            name=self.name,
            config=self.config,
            dir=self.save_dir,
        )

    def close(self) -> None:
        self.run.finish()
