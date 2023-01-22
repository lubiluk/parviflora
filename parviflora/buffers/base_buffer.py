from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Self, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pandas import DataFrame
from torch import Tensor

from ..utils.shape import combined_shape


class BaseBuffer(ABC):
    @abstractmethod
    def __init__(
        self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None
    ) -> None:
        self.device = device

        self.actions = torch.zeros(
            combined_shape(size, env.action_space.shape),
            dtype=torch.float32,
            device=device,
        )
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.terminations = torch.zeros(size, dtype=torch.float32, device=device)
        self.truncations = torch.zeros(size, dtype=torch.float32, device=device)
        self.infos = np.empty((size, 1), dtype=object)
        self._ptr, self.size, self.max_size = 0, 0, size

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
        self._store_observations(observation, next_observation)
        self.actions[self._ptr] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self._ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.terminations[self._ptr] = torch.as_tensor(terminated, dtype=torch.float32)
        self.truncations[self._ptr] = torch.as_tensor(truncated, dtype=torch.float32)
        self.infos[self._ptr] = info
        self._ptr = (self._ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    @abstractmethod
    def _store_observations(
        self,
        observation: Union[NDArray, dict[str, NDArray]],
        next_observation: Union[NDArray, dict[str, NDArray]],
    ) -> None:
        ...

    def sample_batch(
        self, batch_size: int = 32
    ) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        # idxs = torch.randint(0, self.size, size=(batch_size,))
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self._batch(idxs)

    def _batch(self, idxs: Tensor) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        data = dict(
            action=self.actions[idxs],
            reward=self.rewards[idxs],
            terminated=self.terminations[idxs],
            truncated=self.truncations[idxs],
            info=self.infos[idxs],
        )
        observations = self._observations_batch(idxs)
        data.update(observations)

        return data

    @abstractmethod
    def _observations_batch(
        self, idxs: Tensor
    ) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        ...

    def start_episode(self):
        pass

    def end_episode(self):
        pass

    def save(self, filepath: Union[str, Path]) -> None:
        names, data = self._observations_for_saving()

        names.append("action")
        data.append(self.actions.numpy())

        names.append("reward")
        data.append(self.rewards.numpy())

        names.append("termination")
        data.append(self.terminations.numpy())

        names.append("truncation")
        data.append(self.truncations.numpy())

        names.append("info")
        data.append(self.infos)

        df = pd.DataFrame(data, columns=names)
        df.to_csv(filepath, index=False)

    @abstractmethod
    def _observations_for_saving(self) -> Tuple[list[str], list[NDArray]]:
        ...

    def load(self, filepath: Union[str, Path]) -> None:
        df = pd.read_csv(filepath)
        self.actions = torch.from_numpy(df["action"], dtype=torch.float32)
        self.rewards = torch.from_numpy(df["reward"], dtype=torch.float32)
        self.terminations = torch.from_numpy(df["termination"], dtype=torch.float32)
        self.truncations = torch.from_numpy(df["truncation"], dtype=torch.float32)
        self.infos = torch.from_numpy(df["info"], dtype=torch.float32)

    @abstractmethod
    def _load_observations(self, df: DataFrame) -> None:
        ...
