from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from pandas import DataFrame

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
        names = []
        data = []

        names.append(f"observation")
        data.append(self.observations.numpy())

        names.append(f"next_observations")
        data.append(self.next_observations.numpy())

        return names, data

    def _load_observations(self, df: DataFrame) -> None:
        self.observations = torch.from_numpy(df["observations"], dtype=torch.float32)
        self.next_observations = torch.from_numpy(
            df["next_observations"], dtype=torch.float32
        )
