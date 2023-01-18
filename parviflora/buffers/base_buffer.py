from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import gymnasium as gym
import torch
from torch import Tensor
from numpy.typing import NDArray
import numpy as np


class BaseBuffer(ABC):
    @abstractmethod
    def __init__(
        self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None
    ) -> None:
        ...

    @abstractmethod
    def store(
        self,
        observation: Union[NDArray, dict[str, NDArray]],
        action: NDArray,
        reward: float,
        next_observation: Union[NDArray, dict[str, NDArray]],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        ...

    def sample_batch(
        self, batch_size: int = 32
    ) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        # idxs = torch.randint(0, self.size, size=(batch_size,))
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self._get_batch(idxs)

    @abstractmethod
    def _get_batch(self, idxs: Tensor) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        ...

    @abstractmethod
    def start_episode(self):
        ...

    @abstractmethod
    def end_episode(self):
        ...
