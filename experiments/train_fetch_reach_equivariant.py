"""
FetchReach-v4  —  Equivariant policy + HER (SAC)

Identical hyperparameters to train_fetch_reach_mlp.py so the two
runs can be compared directly.

Policy:  EquivariantPolicy  [16 scalars × 8 vectors per hidden layer]
         Encodes SO(3) equivariance and translation invariance by
         construction.  Key differences vs the MLP baseline:

           • EquivariantReachExtractor computes error_vec = goal − ee_pos
             (translation invariant by construction, never learned).

           • Actor uses LinearBlock layers (o3.Linear + gated nonlinearity)
             so the mean displacement output is a proper equivariant vector.

           • Q-function is an MLP over hand-crafted SO(3)-invariant scalars
             (norms and inner products of the geometric vectors).  This gives
             stable early-training convergence while keeping the actor
             fully equivariant.

           • Spatial action is squashed with norm_squash (scale magnitude,
             preserve direction) to keep the squashing equivariant.

           • The Gaussian noise on the spatial action is isotropic (one
             shared σ for all three xyz dimensions) so the distribution
             itself is equivariant: N(Rμ, σ²I) = R·N(μ, σ²I).
"""

from pathlib import Path

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

import torch

from parviflora.algos.sac import SAC
from parviflora.buffers.her_replay_buffer import HerReplayBuffer
from parviflora.loggers.wandb_logger import WandbLogger
from parviflora.policies.equivariant_policy import EquivariantPolicy

ENV_ID = "FetchReach-v4"
N_STEPS = 30_000
LOG_INTERVAL = 1_000
SAVE_PATH = Path("data/checkpoint_fetch_reach_equivariant.pt")


def main():
    device = torch.device("cpu")

    env = gym.make(ENV_ID)

    policy = EquivariantPolicy(
        env.observation_space,
        env.action_space,
        n_scalars=16,
        n_vectors=8,
    )
    policy.to(device)

    buffer = HerReplayBuffer(
        env=env,
        size=N_STEPS * 5,
        n_sampled_goal=4,
        goal_selection_strategy="future",
        device=device,
    )

    logger = WandbLogger(name="reach-equivariant")
    logger.open()

    algo = SAC(
        env,
        policy=policy,
        buffer=buffer,
        update_every=1,
        update_after=1_000,
        batch_size=256,
        alpha="auto",
        # Lower target entropy to account for the isotropic Gaussian constraint.
        # The equivariant actor uses ONE sigma for all 3 spatial dims, giving
        # fewer entropy degrees of freedom than the 4-independent-Gaussian MLP.
        # With target_entropy="auto" (= -4), the equilibrium sigma_xyz ≈ 0.09,
        # producing expected noise magnitude ≈ 0.14 >> success_threshold=0.05.
        # With target_entropy=-1.5, sigma_xyz ≈ 0.024 → noise ≈ 0.04 < 0.05.
        target_entropy=-1.5,
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
