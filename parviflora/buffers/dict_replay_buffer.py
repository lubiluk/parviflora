from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
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

        self.device = device

        obs_space = {
            k: combined_shape(size, v.shape) for k, v in env.observation_space.items()
        }

        self.observations = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device)
            for k, v in env.observation_space.items()
        }
        self.next_observations = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device)
            for k, v in env.observation_space.items()
        }
        self.actions = torch.zeros(
            combined_shape(size, env.action_space.shape),
            dtype=torch.float32,
            device=device,
        )
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.terminations = torch.zeros(size, dtype=torch.float32, device=device)
        self.truncations = torch.zeros(size, dtype=torch.float32, device=device)
        self.infos = np.empty((size, ), dtype=object)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(
        self,
        observation: dict[str, NDArray],
        action: NDArray,
        reward: float,
        next_observation: dict[str, NDArray],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        for k in observation.keys():
            self.observations[k][self.ptr] = torch.as_tensor(
                observation[k], dtype=torch.float32
            )
        for k in next_observation.keys():
            self.next_observations[k][self.ptr] = torch.as_tensor(
                next_observation[k], dtype=torch.float32
            )
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.terminations[self.ptr] = torch.as_tensor(terminated, dtype=torch.float32)
        self.truncations[self.ptr] = torch.as_tensor(truncated, dtype=torch.float32)
        self.infos[self.ptr] = info
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _get_batch(self, idxs: Tensor) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        return dict(
            observation={k: v[idxs] for k, v in self.observations.items()},
            next_observation={k: v[idxs] for k, v in self.next_observations.items()},
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
