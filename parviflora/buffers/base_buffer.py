from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
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
        self.infos = np.empty((size, ), dtype=object)
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
        idxs = torch.randint(0, self.size, size=(batch_size,))
        # idxs = np.random.randint(0, self.size, size=batch_size)
        return self.batch(idxs)

    def batch(self, idxs: Tensor) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
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

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        data_dict = self._observations_for_saving()
        data_dict.update(
            {
                "action": self.actions.cpu().numpy(),
                "reward": self.rewards.cpu().numpy(),
                "termination": self.terminations.cpu().numpy(),
                "truncation": self.truncations.cpu().numpy(),
                "info": self.infos,
                "size": self.size,
                "max_size": self.max_size,
                "_ptr": self._ptr,
            }
        )
        data_dict.update(self._additional_saving_data())

        path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(path, **data_dict)

    @abstractmethod
    def _observations_for_saving(self) -> dict[str, NDArray]:
        ...

    def _additional_saving_data(self) -> dict[str, Any]:
        return {}

    def load(self, filepath: str) -> None:
        filepath = Path(filepath)
        data_dict: dict[str, NDArray] = np.load(filepath, allow_pickle=True)

        self._load_observations(data_dict)
        self.actions = torch.as_tensor(
            data_dict["action"], dtype=torch.float32, device=self.device
        )
        self.rewards = torch.as_tensor(
            data_dict["reward"], dtype=torch.float32, device=self.device
        )
        self.terminations = torch.as_tensor(
            data_dict["termination"], dtype=torch.float32, device=self.device
        )
        self.truncations = torch.as_tensor(
            data_dict["truncation"], dtype=torch.float32, device=self.device
        )
        self.infos = data_dict["info"]
        self.size = data_dict["size"].item()
        self.max_size = data_dict["max_size"].item()
        self._ptr = data_dict["_ptr"].item()
        self._load_additional_data(data_dict)


    @abstractmethod
    def _load_observations(self, data_dict: dict[str, NDArray]) -> None:
        ...

    def _load_additional_data(self, data_dict: dict[str, Any]) -> None:
        ...

    def clear(self):
        self.actions.zero_()
        self.rewards.zero_()
        self.terminations.zero_()
        self.truncations.zero_()
        self.infos.fill(None)
        self._ptr, self.size = 0, 0
