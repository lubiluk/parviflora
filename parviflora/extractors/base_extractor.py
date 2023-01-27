from abc import ABC, abstractmethod
from typing import Any, Optional
import torch
import torch.nn as nn
import gymnasium.spaces as spaces


class BaseExtractor(ABC, nn.Module):
    @abstractmethod
    def __init__(self, observation_space: spaces.Dict) -> None:
        super().__init__()
        self.n_features: int = 0

    @abstractmethod
    def forward(self, observation, device: Optional[torch.device] = None):
        ...
