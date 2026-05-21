import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)
import torch

from parviflora.algos.sac import SAC
from parviflora.buffers.her_replay_buffer import HerReplayBuffer
from parviflora.extractors.dict_extractor import DictExtractor
from parviflora.loggers.wandb_logger import WandbLogger
from parviflora.policies.mlp_policy import MlpPolicy


def main():
    device = torch.device("mps")

    env = gym.make("FetchReach-v4")

    policy = MlpPolicy(
        env.observation_space,
        env.action_space,
        hidden_sizes=[64, 64],
        extractor_type=DictExtractor,
    )
    policy.to(device)

    buffer = HerReplayBuffer(
        env=env,
        size=300_000 * 5,
        n_sampled_goal=4,
        goal_selection_strategy="future",
        device=device,
    )
    logger = WandbLogger(name="panda-reg")
    logger.open()

    algo = SAC(
        env,
        policy=policy,
        buffer=buffer,
        update_every=1,
        update_after=1000,
        batch_size=256,
        alpha="auto",
        # alpha=0.05,
        gamma=0.95,
        # polyak=0.95,
        lr=7e-4,
        logger=logger,
        max_episode_len=100,
        start_steps=1_000,
    )

    algo.train(n_steps=300_000, log_interval=1000)

    env.close()
    logger.close()

    policy.cpu()

    env = gym.make("PandaPush-v3", render_mode="human")
    test_rew, test_ep_len = algo.test(env, n_episodes=50, sleep=1 / 30)
    env.close()
    print(f"Test reward {test_rew}, Test episode length: {test_ep_len}")


if __name__ == "__main__":
    main()
