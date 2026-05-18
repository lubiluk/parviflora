"""
FetchReach-v4  —  MLP + HER baseline (SAC)

Identical hyperparameters to train_fetch_reach_equivariant.py so the two
runs can be compared directly.

Policy:  MlpPolicy  [64 × 64]  with DictExtractor
         Flattens the full goal-conditioned observation dict and processes
         it through a standard MLP. Translation/rotation invariance must
         be learned from data.
"""

from pathlib import Path

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

import torch

from parviflora.algos.sac import SAC
from parviflora.buffers.her_replay_buffer import HerReplayBuffer
from parviflora.extractors.dict_extractor import DictExtractor
from parviflora.loggers.tensorboard_logger import TensorboardLogger
from parviflora.policies.mlp_policy import MlpPolicy

ENV_ID = "FetchReach-v4"
N_STEPS = 30_000
LOG_INTERVAL = 1_000
SAVE_PATH = Path("data/checkpoint_fetch_reach_mlp.pt")


def main():
    device = torch.device("cpu")

    env = gym.make(ENV_ID)

    policy = MlpPolicy(
        env.observation_space,
        env.action_space,
        hidden_sizes=[64, 64],
        extractor_type=DictExtractor,
    )
    policy.to(device)

    buffer = HerReplayBuffer(
        env=env,
        size=N_STEPS * 5,
        n_sampled_goal=4,
        goal_selection_strategy="future",
        device=device,
    )

    logger = TensorboardLogger()
    logger.open()

    algo = SAC(
        env,
        policy=policy,
        buffer=buffer,
        update_every=1,
        update_after=1_000,
        batch_size=256,
        alpha="auto",
        gamma=0.99,
        lr=7e-4,
        logger=logger,
        max_episode_len=100,
        start_steps=1_000,
    )

    algo.train(n_steps=N_STEPS, log_interval=LOG_INTERVAL)

    env.close()
    logger.close()

    SAVE_PATH.parent.mkdir(exist_ok=True)
    algo.save(str(SAVE_PATH))
    print(f"Checkpoint saved → {SAVE_PATH}")

    env = gym.make(ENV_ID, render_mode="human")
    results = algo.test(env, n_episodes=50, sleep=1 / 30)
    env.close()
    msg = (
        f"Test reward: {results['mean_ep_ret']:.3f}  "
        f"Episode length: {results['mean_ep_len']:.1f}"
    )
    if "success_rate" in results:
        msg += f"  Success rate: {results['success_rate']:.1%}"
    if "mean_final_goal_dist" in results:
        msg += f"  Mean final goal dist: {results['mean_final_goal_dist']:.4f}"
    if "mean_min_goal_dist" in results:
        msg += f"  Mean min goal dist: {results['mean_min_goal_dist']:.4f}"
    print(msg)


if __name__ == "__main__":
    main()
