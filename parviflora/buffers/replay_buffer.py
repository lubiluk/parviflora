from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor
from numpy.typing import NDArray

from ..buffers.base_buffer import BaseBuffer
from ..utils.shape import combined_shape


class ReplayBuffer(BaseBuffer):
    """
    A basic experience replay buffer for off-policy agents.
    """

    def __init__(
        self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None
    ):
        self.device = device

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
        self.actions = torch.zeros(
            combined_shape(size, env.action_space.shape),
            dtype=torch.float32,
            device=device,
        )
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.terminations = torch.zeros(size, dtype=torch.float32, device=device)
        self.truncations = torch.zeros(size, dtype=torch.float32, device=device)
        self.infos = np.empty((size, 1), dtype=object)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(
        self,
        observation: NDArray,
        action: NDArray,
        reward: float,
        next_observation: NDArray,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self.observations[self.ptr] = torch.as_tensor(observation, dtype=torch.float32)
        self.next_observations[self.ptr] = torch.as_tensor(
            next_observation, dtype=torch.float32
        )
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.terminations[self.ptr] = torch.as_tensor(terminated, dtype=torch.float32)
        self.truncations[self.ptr] = torch.as_tensor(truncated, dtype=torch.float32)
        self.infos[self.ptr] = info
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _get_batch(self, idxs: Tensor) -> dict[str, Tensor]:
        return dict(
            observation=self.observations[idxs],
            next_observation=self.next_observations[idxs],
            action=self.actions[idxs],
            reward=self.rewards[idxs],
            terminated=self.terminations[idxs],
            truncated=self.truncations[idxs],
            info=self.infos[idxs],
        )

    def start_episode(self):
        pass

    def end_episode(self):
        pass
