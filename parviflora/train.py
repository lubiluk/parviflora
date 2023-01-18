import gymnasium as gym
import torch

from .algos.sac import SAC
from .buffers.dict_replay_buffer import DictReplayBuffer
from .buffers.her_replay_buffer import HerReplayBuffer
from .buffers.replay_buffer import ReplayBuffer
from .envs.bit_flipping_env import BitFlippingEnv
from .extractors.array_extractor import ArrayExtractor
from .extractors.dict_extractor import DictExtractor
from .loggers.tensorboard_logger import TensorboardLogger


def main():
    # env = gym.make("Pendulum-v1")
    # ac_kwargs = dict(hidden_sizes=(256, 256), extractor_type=ArrayExtractor)
    # rb_kwargs = dict(size=1000000)

    # with TensorboardLogger() as logger:
    #     algo = SAC(
    #         env,
    #         ac_kwargs=ac_kwargs,
    #         replay_buffer_type=ReplayBuffer,
    #         rb_kwargs=rb_kwargs,
    #         update_every=1,
    #         update_after=100,
    #         batch_size=256,
    #         ent_coef="auto",
    #         logger=logger,
    #     )
    #     algo.train(n_steps=20000, log_interval=1000)

    # env = gym.make("Pendulum-v1", render_mode="human")
    # test_rew, test_ep_len = algo.test(env, n_episodes=10)
    # print(f"Test reward {test_rew}, Test episode length: {test_ep_len}")

    import panda_gym

    env = gym.make("PandaReach-v3")
    ac_kwargs = dict(hidden_sizes=[64, 64], extractor_type=DictExtractor)
    rb_kwargs = dict(size=1000000, n_sampled_goal=4, goal_selection_strategy="future")

    with TensorboardLogger() as logger:
        algo = SAC(
            env,
            ac_kwargs=ac_kwargs,
            replay_buffer_type=HerReplayBuffer,
            rb_kwargs=rb_kwargs,
            update_every=1,
            update_after=1000,
            batch_size=256,
            ent_coef="auto",
            gamma=0.95,
            lr=0.001,
            logger=logger,
            max_ep_len=100,
        )
        algo.train(n_steps=30000, log_interval=1000)
        env.close()

    env = gym.make("PandaReach-v3", render_mode="human")
    test_rew, test_ep_len = algo.test(env, n_episodes=100, sleep=1/30)
    env.close()
    print(f"Test reward {test_rew}, Test episode length: {test_ep_len}")


if __name__ == "__main__":
    main()
