from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray

from ..utils.shape import combined_shape
from .dict_replay_buffer import DictReplayBuffer


class HerReplayBuffer(DictReplayBuffer):
    """
    A Hindsight Experience Replay buffer for off-policy agents.
    """

    def __init__(
        self,
        env: gym.Env,
        size: int = 100000,
        device: Optional[torch.device] = None,
        n_sampled_goal: int = 1,
        goal_selection_strategy: str = "final",
    ):
        super().__init__(env=env, size=size, device=device)
        self.env = env
        self.n_sampled_goal = n_sampled_goal
        self.selection_strategy = goal_selection_strategy

    def store(
        self,
        observation: dict[str, NDArray],
        action: NDArray,
        reward: float,
        next_observation: dict[str, NDArray],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ):
        super().store(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        if self._ptr == self.ep_start_ptr:
            raise "Episode longer than buffer size"

    def start_episode(self):
        self.ep_start_ptr = self._ptr

    def end_episode(self):
        self._synthesize_experience()

    def _get_current_episode(self):
        if self.ep_start_ptr == self._ptr:
            idxs = [self._ptr]

        if self.ep_start_ptr <= self._ptr:
            idxs = np.arange(self.ep_start_ptr, self._ptr)
        else:
            idxs = np.concatenate(
                [np.arange(self.ep_start_ptr, self.size), np.arange(self._ptr)]
            )

        return self.batch(idxs)

    def _synthesize_experience(self):
        ep = self._get_current_episode()
        ep_len = len(ep["reward"])

        for idx in range(ep_len):
            observation = {k: v[idx] for k, v in ep["observation"].items()} 
            action = ep["action"][idx]
            next_observation = {k: v[idx] for k, v in ep["next_observation"].items()} 
            info = ep["info"][idx]
            np_agoal = next_observation["achieved_goal"].cpu().numpy()

            for _ in range(self.n_sampled_goal):
                if self.selection_strategy == "final":
                    sel_idx = -1
                elif self.selection_strategy == "future":
                    # We cannot sample a goal from the future in the last step of an episode
                    if idx == ep_len - 1:
                        break
                    sel_idx = np.random.choice(np.arange(idx + 1, ep_len))
                elif self.selection_strategy == "episode":
                    sel_idx = np.random.choice(np.arange(ep_len))
                else:
                    raise ValueError(
                        "Unsupported selection_strategy: {}".format(
                            self.selection_strategy
                        )
                    )

                sel_agoal = ep["next_observation"]["achieved_goal"][sel_idx]
                info = ep["info"][sel_idx][0]
                terminated = ep["terminated"][sel_idx]
                truncated = ep["truncated"][sel_idx]
                np_sel_agoal = sel_agoal.cpu().numpy()

                reward = self.env.compute_reward(np_agoal, np_sel_agoal, info)

                observation["desired_goal"] = sel_agoal
                next_observation["desired_goal"] = sel_agoal

                self.store(
                    observation,
                    action,
                    reward,
                    next_observation,
                    terminated,
                    truncated,
                    info,
                )
