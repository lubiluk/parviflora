"""
FetchPickAndPlace-v4  —  Equivariant policy + HER (SAC)

Identical hyperparameters to train_fetch_pick_and_place_mlp.py for a direct
comparison.

Policy:  EquivariantPolicy with EquivariantPushExtractor
         The observation layout for PickAndPlace is identical to Push (same
         25-dim vector), so the extractor is reused without modification.

  What changes vs Push
  ─────────────────────
  • block_gripper=False: the gripper action (dim 3, scalar) is now critical.
    The actor already treats it as an independent scalar; no architecture
    change is needed.
  • target_in_the_air=True: goals can have arbitrary z.  The error_vec
    (desired_goal − object_pos) is still a polar vector — it just now also
    has a non-zero z component.  Equivariance handles this automatically.
  • Task is harder: more capacity is appropriate.
    n_scalars=32, n_vectors=16 (vs 16/8 for Push),
    critic_hidden_sizes=(256, 256) (vs 64/64 for Push).

  What's encoded by construction
  ────────────────────────────────
  • Translation invariance: error_vec = goal − object_pos,
    grip_to_obj = object_pos − grip_pos — no absolute positions.
  • SO(3) equivariance: rotating the scene rotates the action vector
    by the same amount; Q-values are rotation-invariant.
  • Scalar gate conditioning: 10 invariant scalars (4 norms + 6 pairwise
    dots of the 4 vector channels) let the LinearBlock gates depend on
    distance-to-goal, gripper-object distance, and velocity alignments.
"""

from pathlib import Path

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

import torch

from parviflora.algos.sac import SAC
from parviflora.buffers.her_replay_buffer import HerReplayBuffer
from parviflora.extractors.equivariant_push_extractor import EquivariantPushExtractor
from parviflora.loggers.wandb_logger import WandbLogger
from parviflora.policies.equivariant_policy import EquivariantPolicy

ENV_ID = "FetchPickAndPlace-v4"
N_STEPS = 1_000_000
LOG_INTERVAL = 1_000
SAVE_PATH = Path("data/checkpoint_fetch_pick_and_place_equivariant.pt")


def main():
    device = torch.device("cpu")

    env = gym.make(ENV_ID)

    policy = EquivariantPolicy(
        env.observation_space,
        env.action_space,
        extractor_type=EquivariantPushExtractor,
        # Larger capacity than Push: grasping + 3-D placement is harder
        n_scalars=32,
        n_vectors=16,
        critic_hidden_sizes=(256, 256),
    )
    policy.to(device)

    buffer = HerReplayBuffer(
        env=env,
        size=N_STEPS,
        n_sampled_goal=4,
        goal_selection_strategy="future",
        device=device,
    )

    logger = WandbLogger(name="pick-and-place-equivariant")
    logger.open()

    algo = SAC(
        env,
        policy=policy,
        buffer=buffer,
        update_every=1,
        update_after=1_000,
        batch_size=1048,
        alpha="auto",
        # Lower target entropy for the isotropic Gaussian constraint
        # (one sigma_xyz for all 3 spatial dims).
        target_entropy=-1.5,
        gamma=0.95,
        lr=1e-3,
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
