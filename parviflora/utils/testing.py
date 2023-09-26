import time
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


@dataclass
class TestResults:
    mean_ep_ret: float
    mean_ep_len: float
    success_rate: Optional[float] = None
    ep_frames: Optional[list[list[NDArray]]] = None


# def test(
#     algo: SAC,
#     env: gym.Env,
#     n_episodes: int,
#     sleep: float = 0,
#     store_experience: bool = False,
#     render: bool = False,
# ) -> TestResutls:
#     ep_returns = []
#     ep_lengths = []
#     ep_successes = []
#     algo.policy.eval()

#     if render:
#         frames = []

#     for _ in range(n_episodes):
#         (o, i), d, ep_ret, ep_len = env.reset(), False, 0, 0

#         if store_experience:
#             algo.buffer.start_episode()

#         if render:
#             frames.append([])

#         while not (d or (ep_len == algo.max_episode_len)):
#             # Take deterministic actions at test time
#             a = algo.policy.act(o, True)

#             o2, r, ter, tru, i = env.step(a)

#             if store_experience:
#                 # Store experience to replay buffer
#                 algo.buffer.store(o, a, r, o2, ter, tru, i)

#             o = o2

#             if render:
#                 frame = env.render()
#                 frames[-1].append(frame)

#             if sleep > 0:
#                 time.sleep(sleep)

#             d = ter or tru
#             ep_ret += r
#             ep_len += 1

#         if store_experience:
#             algo.buffer.end_episode()

#         ep_returns.append(ep_ret)
#         ep_lengths.append(ep_len)

#         if "is_success" in i:
#             ep_successes.append(i)

#     results = TestResutls(
#         mean_ep_ret=np.array(ep_returns).mean(), mean_ep_len=np.array(ep_lengths).mean()
#     )

#     if len(ep_successes) > 0:
#         results.success_rate = np.array(ep_successes).mean()

#     if render:
#         results.ep_frames = frames

#     return results
