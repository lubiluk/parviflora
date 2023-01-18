from typing import Optional

import gymnasium.spaces as spaces
import torch
from numpy.typing import NDArray

from .base_extractor import BaseExtractor


class DictExtractor(BaseExtractor):
    def __init__(
        self, observation_space: spaces.Dict, device: Optional[torch.device] = None
    ) -> None:
        super().__init__(observation_space=observation_space, device=device)
        self.n_features = spaces.utils.flatdim(observation_space)
        self._keys = observation_space.keys()

    def forward(self, observation: dict[str, NDArray]):
        # obs_lin = torch.as_tensor(observation["observation"], dtype=torch.float32).to(self.device)
        # dgoal = torch.as_tensor(observation["desired_goal"], dtype=torch.float32).to(self.device)
        # return torch.cat((obs_lin, dgoal), dim=-1)

        tensors = [
            torch.as_tensor(observation[k], dtype=torch.float32, device=self.device)
            for k in self._keys
        ]

        return torch.cat(tensors, dim=-1)
