from typing import Optional

import gymnasium.spaces as spaces
import torch
from numpy.typing import NDArray

from .base_extractor import BaseExtractor


class ArrayExtractor(BaseExtractor):
    def __init__(
        self, observation_space: spaces.Dict, device: Optional[torch.device] = None
    ) -> None:
        super().__init__(observation_space=observation_space, device=device)
        self.n_features = observation_space.shape[0]

    def forward(self, observation: NDArray):
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)

        return tensor
