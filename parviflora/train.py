import gymnasium as gym
import torch

from parviflora.policies.mlp_policy import MlpPolicy

from .algos.sac import SAC
from .buffers.dict_replay_buffer import DictReplayBuffer
from .buffers.her_replay_buffer import HerReplayBuffer
from .buffers.replay_buffer import ReplayBuffer
from .envs.bit_flipping_env import BitFlippingEnv
from .extractors.array_extractor import ArrayExtractor
from .extractors.dict_extractor import DictExtractor
from .loggers.tensorboard_logger import TensorboardLogger

import gym_process


def main():
    # if torch.cuda.is_available():
    #     if gpu_buffer:
    #         buff_device = torch.device("cuda")
    #         self.logger.log_msg("\nUsing GPU replay buffer\n")

    #     if gpu_computation:
    #         comp_device = torch.device("cuda")
    #         self.logger.log_msg("\nUsing GPU computaion\n")
    # else:
    #     self.logger.log_msg("\nGPU unavailable\n")


    # env = gym.make("PaperSteam-v0")
    # ac_kwargs = dict(hidden_sizes=(32, 32), extractor_type=ArrayExtractor)
    # rb_kwargs = dict(size=100000)

    # with TensorboardLogger() as logger:
    #     algo = SAC(
    #         env,
    #         ac_kwargs=ac_kwargs,
    #         replay_buffer_type=ReplayBuffer,
    #         rb_kwargs=rb_kwargs,
    #         update_every=1,
    #         update_after=100,
    #         batch_size=256,
    #         alpha="auto",
    #         logger=logger,
    #     )
    #     algo.train(n_steps=20000, log_interval=100)

    env = gym.make("Pendulum-v1")
    policy = MlpPolicy(env.observation_space, env.action_space, hidden_sizes=(256,256), extractor_type=ArrayExtractor)
    buffer = ReplayBuffer(env=env, size=2000)

    with TensorboardLogger() as logger:
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
        algo.train(n_steps=2000, log_interval=1000)

    # env = gym.make("Pendulum-v1", render_mode="human")
    # test_rew, test_ep_len = algo.test(env, n_episodes=5)
    # print(f"Test reward {test_rew}, Test episode length: {test_ep_len}")

    buffer.save("data/buffer.npz")
    new_buffer = ReplayBuffer(env=env, size=2000)
    new_buffer.load("data/buffer.npz")

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


    import panda_gym

    env = gym.make("PandaReach-v3")
    ac_kwargs = dict(hidden_sizes=[64, 64], extractor_type=DictExtractor)
    rb_kwargs = dict(size=1000000, n_sampled_goal=4, goal_selection_strategy="future")

    with TensorboardLogger() as logger:
        algo = SAC(
            env,
            policy_kwargs=ac_kwargs,
            buffer_type=HerReplayBuffer,
            buffer_kwargs=rb_kwargs,
            update_every=1,
            update_after=1000,
            batch_size=256,
            alpha="auto",
            gamma=0.95,
            lr=0.001,
            logger=logger,
            max_episode_len=100,
        )
        algo.train(n_steps=3000, log_interval=1000)
        env.close()

    env = gym.make("PandaReach-v3", render_mode="human")
    test_rew, test_ep_len = algo.test(env, n_episodes=100, sleep=1/30)
    env.close()
    print(f"Test reward {test_rew}, Test episode length: {test_ep_len}")


if __name__ == "__main__":
    main()
