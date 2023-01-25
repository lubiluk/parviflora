from typing import Any, Optional, Tuple

import gymnasium as gym
import torch
from numpy.typing import NDArray
from torch import Tensor

from ..buffers.base_buffer import BaseBuffer
from ..utils.shape import combined_shape


class DictReplayBuffer(BaseBuffer):
    """
    A dictionary experience replay buffer for off-policy agents.
    """

    def __init__(
        self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None
    ):
        assert isinstance(env.observation_space, gym.spaces.Dict)
        super().__init__(env=env, size=size, device=device)

        obs_space = {
            k: combined_shape(size, v.shape) for k, v in env.observation_space.items()
        }

        self.observations: dict[str, Tensor] = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device)
            for k, v in env.observation_space.items()
        }
        self.next_observations: dict[str, Tensor] = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device)
            for k, v in env.observation_space.items()
        }

    def _store_observations(
        self,
        observation: dict[str, NDArray],
        next_observation: dict[str, NDArray],
    ) -> None:
        for k in observation.keys():
            self.observations[k][self._ptr] = torch.as_tensor(
                observation[k], dtype=torch.float32
            )
        for k in next_observation.keys():
            self.next_observations[k][self._ptr] = torch.as_tensor(
                next_observation[k], dtype=torch.float32
            )

    def _observations_batch(self, idxs: Tensor) -> dict[str, dict[str, Tensor]]:
        return dict(
            observation={k: v[idxs] for k, v in self.observations.items()},
            next_observation={k: v[idxs] for k, v in self.next_observations.items()},
        )

    def _observations_for_saving(self) -> Tuple[list[str], list[NDArray]]:
        data_dict = {
            f"observation[{k}]": v.numpy() for k, v in self.observations.items()
        }
        data_dict.update(
            {f"observation[{k}]": v.numpy() for k, v in self.observations.items()}
        )

        return data_dict

    def _load_observations(self, data_dict: dict[str, NDArray]) -> None:
        observation_columns = [
            c for c in data_dict.keys() if c.startswith("observation[")
        ]

        for c in observation_columns:
            k = c[len("observation[") : -1]
            self.observations[k] = torch.as_tensor(data_dict[c], dtype=torch.float32)

        next_observation_columns = [
            c for c in data_dict.columns if c.startswith("next_observation[")
        ]

        for c in next_observation_columns:
            k = c[len("next_observation[") : -1]
            self.next_observations[k] = torch.as_tensor(
                data_dict[c], dtype=torch.float32
            )
