from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ..buffers.base_buffer import BaseBuffer
from ..utils.shape import combined_shape


class ReplayBuffer(BaseBuffer):
    """
    A basic experience replay buffer for off-policy agents.
    """

    def __init__(
        self,
        env: gym.Env,
        size: int = 100000,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(env=env, size=size, device=device)

        obs_space = combined_shape(size, env.observation_space.shape)

        self.observations = torch.zeros(
            obs_space,
            dtype=torch.float32,
            device=device,
        )
        self.next_observations = torch.zeros(
            obs_space,
            dtype=torch.float32,
            device=device,
        )

    def _store_observations(
        self,
        observation: NDArray,
        next_observation: NDArray,
    ) -> None:
        self.observations[self._ptr] = torch.as_tensor(observation, dtype=torch.float32)
        self.next_observations[self._ptr] = torch.as_tensor(
            next_observation, dtype=torch.float32
        )

    def _observations_batch(self, idxs: Tensor) -> dict[str, Tensor]:
        return dict(
            observation=self.observations[idxs],
            next_observation=self.next_observations[idxs],
        )

    def _observations_for_saving(self) -> Tuple[list[str], list[NDArray]]:
        return {
            "observation": self.observations[:self.size].cpu().numpy(),
            "next_observation": self.next_observations[:self.size].cpu().numpy(),
        }

    def _load_observations(self, data_dict: dict[str, NDArray]) -> None:
        self.observations = torch.as_tensor(
            data_dict["observation"], dtype=torch.float32
        )
        self.next_observations = torch.as_tensor(
            data_dict["next_observation"], dtype=torch.float32
        )
