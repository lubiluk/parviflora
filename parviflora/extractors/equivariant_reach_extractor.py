from typing import Optional

import gymnasium.spaces as spaces
import torch
from e3nn import o3

from .base_extractor import BaseExtractor


class EquivariantReachExtractor(BaseExtractor):
    """
    Converts a FetchReach / PandaReach GoalEnv dict observation into
    equivariant irrep features, encoding translation invariance by
    construction instead of learning it from data.

    Raw dict keys used:
        observation  (10,): [ee_pos(3), gripper_state(2), ee_vel(3), gripper_vel(2)]
        achieved_goal (3,): current end-effector position
        desired_goal  (3,): target end-effector position

    Output tensor layout  →  irreps "2x1o + 4x0e"  (dim = 10):
        dims  0-2  : error_vec  = desired_goal − achieved_goal   [1x1o  polar vector]
        dims  3-5  : ee_vel     = observation[5:8]               [1x1o  polar vector]
        dims  6-7  : gripper finger positions = observation[3:5] [2x0e  scalars]
        dims  8-9  : gripper finger velocities = observation[8:10][2x0e scalars]

    Translation invariance: absolute positions are never used; only the
    goal-relative displacement appears in the features.
    """

    irreps_out: o3.Irreps = o3.Irreps("2x1o + 4x0e")  # type: ignore[assignment]

    def __init__(self, observation_space: spaces.Dict) -> None:
        super().__init__(observation_space=observation_space)
        self.n_features = self.irreps_out.dim  # 10

    def forward(
        self,
        observation: dict,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        achieved = torch.as_tensor(
            observation["achieved_goal"], dtype=torch.float32, device=device
        )
        desired = torch.as_tensor(
            observation["desired_goal"], dtype=torch.float32, device=device
        )
        raw = torch.as_tensor(
            observation["observation"], dtype=torch.float32, device=device
        )

        # Translation-invariant goal error  (1x1o)
        error_vec = desired - achieved  # (..., 3)
        # End-effector velocity               (1x1o)
        ee_vel = raw[..., 5:8]  # (..., 3)
        # Gripper scalars                     (4x0e)
        gripper_sc = torch.cat([raw[..., 3:5], raw[..., 8:10]], dim=-1)  # (..., 4)

        # Layout must match irreps "2x1o + 4x0e":
        #   vectors (1o) before scalars (0e)
        return torch.cat([error_vec, ee_vel, gripper_sc], dim=-1)  # (..., 10)
