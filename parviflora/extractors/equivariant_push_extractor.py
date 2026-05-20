"""
Equivariant feature extractor for FetchPush-v4.

Raw observation dict
────────────────────
  observation  (25,):
    [0:3]   grip_pos       — absolute gripper position  (not translation-invariant)
    [3:6]   object_pos     — absolute object position   (not translation-invariant)
    [6:9]   object_rel_pos = object_pos − grip_pos      (translation-invariant 1o)
    [9:11]  gripper_state  — finger joint positions     (0e scalars)
    [11:14] object_rot     — Euler angles               (not equivariant → skipped)
    [14:17] object_velp    — object linear vel, relative to gripper  (1o)
    [17:20] object_velr    — object angular vel (axial/1e → skipped for now)
    [20:23] grip_velp      — gripper linear velocity    (1o)
    [23:25] gripper_vel    — finger joint velocities    (0e scalars)
  achieved_goal (3,): object_pos
  desired_goal  (3,): target position for the object

Output tensor layout  →  irreps "4x1o + 4x0e"  (dim = 16):
  dims  0-2  : error_vec   = desired_goal − achieved_goal  [1x1o]
  dims  3-5  : grip_to_obj = obs[6:9] = object_pos − grip_pos  [1x1o]
  dims  6-8  : object_velp = obs[14:17]  [1x1o]
  dims  9-11 : grip_velp   = obs[20:23]  [1x1o]
  dims 12-13 : gripper_state = obs[9:11]  [2x0e]
  dims 14-15 : gripper_vel   = obs[23:25]  [2x0e]

Design notes
────────────
• Translation invariance: error_vec and grip_to_obj are differences of
  absolute positions, so they're invariant to global translation.
• Object rotation (obs[11:14]) is represented as Euler angles, which are
  not equivariant under SO(3) rotations of the scene. Angular velocity
  (obs[17:20]) is an axial vector (1e irrep); we omit it for simplicity.
  Both can be added later if needed.
• FetchPush sets block_gripper=True, so the gripper is always closed.
  Gripper scalars carry little information but are included for consistency.
"""

from typing import Optional

import gymnasium.spaces as spaces
import torch
from e3nn import o3

from .base_extractor import BaseExtractor


class EquivariantPushExtractor(BaseExtractor):
    irreps_out: o3.Irreps = o3.Irreps("4x1o + 4x0e")  # type: ignore[assignment]

    def __init__(self, observation_space: spaces.Dict) -> None:
        super().__init__(observation_space=observation_space)
        self.n_features = self.irreps_out.dim  # 16

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

        # Vector from gripper to object            (1x1o)
        # obs[6:9] = object_rel_pos = object_pos − grip_pos
        grip_to_obj = raw[..., 6:9]  # (..., 3)

        # Object linear velocity (relative to gripper)  (1x1o)
        object_velp = raw[..., 14:17]  # (..., 3)

        # Gripper linear velocity                  (1x1o)
        grip_velp = raw[..., 20:23]  # (..., 3)

        # Gripper scalars                          (4x0e)
        gripper_state = raw[..., 9:11]  # (..., 2)
        gripper_vel = raw[..., 23:25]  # (..., 2)

        # Layout: vectors (1o) first, then scalars (0e)
        return torch.cat(
            [
                error_vec,
                grip_to_obj,
                object_velp,
                grip_velp,
                gripper_state,
                gripper_vel,
            ],
            dim=-1,
        )  # (..., 16)
