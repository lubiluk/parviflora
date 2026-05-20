"""
FetchPush-v4  —  Equivariant policy + HER (SAC)

Identical hyperparameters to train_fetch_push_mlp.py for a direct comparison.

Policy:  EquivariantPushPolicy  [16 scalars × 8 vectors per hidden layer]

  What's encoded by construction
  ────────────────────────────────
  • Translation invariance: error_vec = goal − object_pos, and
    grip_to_obj = object_pos − grip_pos are differences of positions.
    No absolute coordinates appear in the feature vector.

  • SO(3) equivariance: the actor maps equivariant features to an
    equivariant action. Rotating the scene rotates the output action
    by the same amount. The Q-function maps to an invariant scalar.

  • Scalar gate conditioning fix: the 4 vector channels produce
    10 invariant scalars (4 norms + 6 pairwise dot products) that are
    prepended as 0e features so the LinearBlock gates can depend on
    distance-to-goal, gripper-object distance, and velocity alignments.
    Without this, the actor collapses to a fixed-gain controller.

  Key geometric features for Push
  ─────────────────────────────────
    dot(error_vec, grip_to_obj) < 0  ←→  gripper is behind the object
                                          relative to the goal (push side)
    dot(error_vec, object_velp) > 0  ←→  object is moving toward the goal
    dot(grip_to_obj, act_xyz)   > 0  ←→  action moves gripper toward object
"""

from pathlib import Path

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

import torch

from parviflora.algos.sac import SAC
from parviflora.buffers.her_replay_buffer import HerReplayBuffer
from parviflora.extractors.equivariant_push_extractor import EquivariantPushExtractor
from parviflora.loggers.tensorboard_logger import TensorboardLogger
from parviflora.policies.equivariant_policy import EquivariantPolicy

ENV_ID = "FetchPush-v4"
N_STEPS = 500_000
LOG_INTERVAL = 1_000
SAVE_PATH = Path("data/checkpoint_fetch_push_equivariant.pt")


def main():
    device = torch.device("cpu")

    env = gym.make(ENV_ID)

    policy = EquivariantPolicy(
        env.observation_space,
        env.action_space,
        extractor_type=EquivariantPushExtractor,
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
        # Lower target entropy to account for the isotropic Gaussian constraint:
        # one sigma_xyz for all 3 spatial dims gives fewer entropy degrees of
        # freedom than the MLP's 4 independent Gaussians.
        target_entropy=-1.5,
        gamma=0.99,
        lr=7e-4,
        logger=logger,
        max_episode_len=50,
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
