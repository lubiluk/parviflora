from pathlib import Path

import gymnasium as gym
import torch
import panda_gym

from parviflora.policies.mlp_policy import MlpPolicy

from .algos.sac import SAC
from .buffers.dict_replay_buffer import DictReplayBuffer
from .buffers.her_replay_buffer import HerReplayBuffer
from .buffers.replay_buffer import ReplayBuffer
from .envs.bit_flipping_env import BitFlippingEnv
from .extractors.array_extractor import ArrayExtractor
from .extractors.dict_extractor import DictExtractor
from .loggers.tensorboard_logger import TensorboardLogger


def main():
    logger = TensorboardLogger()
    logger.open()

    env = gym.make("Pendulum-v1")
    policy = MlpPolicy(env.observation_space, env.action_space, hidden_sizes=(256,256), extractor_type=ArrayExtractor)
    buffer = ReplayBuffer(env=env, size=20000)

    algo = SAC(
        env,
        policy=policy,
        buffer=buffer,
        update_every=1,
        update_after=100,
        batch_size=256,
        alpha="auto",
        logger=logger,
    )

    algo.train(n_steps=20000, log_interval=1000)
    env.close()

    env = gym.make("Pendulum-v1", render_mode="human")
    test_rew, test_ep_len = algo.test(env, n_episodes=5)
    print(f"Test reward {test_rew}, Test episode length: {test_ep_len}")
    env.close()

    env = gym.make("Pendulum-v1")
    policy = MlpPolicy(env.observation_space, env.action_space, hidden_sizes=(256,256), extractor_type=ArrayExtractor)
    buffer = ReplayBuffer(env=env, size=20000)

    algo = SAC(
        env,
        policy=policy,
        buffer=buffer,
        update_every=1,
        update_after=100,
        batch_size=256,
        alpha="auto",
        logger=logger,
    )

    algo.train_passively(n_steps=20000, log_interval=1000)
    env.close()

    env = gym.make("Pendulum-v1", render_mode="human")
    test_rew, test_ep_len = algo.test_passively(env, n_episodes=5)
    print(f"Test reward {test_rew}, Test episode length: {test_ep_len}")
    env.close()


    logger.close()

    return

    

    # env = BitFlippingEnv(n_bits=15, continuous=True, max_steps=15)
    # ac_kwargs = dict(hidden_sizes=[64, 64], extractor_type=DictExtractor)
    # rb_kwargs = dict(size=1000000, n_sampled_goal=4, goal_selection_strategy="future")
    # algo = SAC(
    #     env,
    #     ac_kwargs=ac_kwargs,
    #     replay_buffer_type=HerReplayBuffer,
    #     rb_kwargs=rb_kwargs,
    #     update_every=1,
    #     update_after=1000,
    #     batch_size=256,
    #     ent_coef="auto",
    #     gamma=0.95,
    #     lr=0.0003,
    #     # logger=logger,
    #     max_ep_len=100,
    # )
    # algo.train(n_steps=10000, log_interval=100)
    # env.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("PandaPush-v3")

    policy = MlpPolicy(
        env.observation_space,
        env.action_space,
        hidden_sizes=[512, 512, 512],
        extractor_type=DictExtractor,
    )
    # policy.load_state_dict(torch.load("data/model_300k_ref.pt", map_location=torch.device('cpu')))
    policy.to(device)

    buffer = HerReplayBuffer(
        env=env,
        size=300_000*5,
        n_sampled_goal=4,
        goal_selection_strategy="future",
        device=device,
    )
    # buffer.load("data/her_buffer_300k_all.npz")
    logger = TensorboardLogger()
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
        start_steps=300_000
    )
    algo.train(n_steps=300_000, log_interval=1000)
    # algo.train_offline(100)
    # algo.train_offline_exact(300_000)
    env.close()
    logger.close()

    policy.cpu()
    torch.save(policy.state_dict(), "data/model_300k_rand.pt")
    buffer.save(Path("data/her_buffer_300k_rand.npz"))

    env = gym.make("PandaPush-v3", render_mode="human")
    test_rew, test_ep_len = algo.test(env, n_episodes=50, sleep=1 / 30)
    env.close()
    print(f"Test reward {test_rew}, Test episode length: {test_ep_len}")


if __name__ == "__main__":
    main()
