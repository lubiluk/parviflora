import gymnasium as gym
import torch

from .envs.bit_flipping_env import BitFlippingEnv

from .algos.sac import SAC
from .loggers.tensorboard_logger import TensorboardLogger


def main():
    env = BitFlippingEnv(n_bits=15, continuous=True, max_steps=15)
    ac_kwargs = dict(hidden_sizes=(256, 256))
    rb_kwargs = dict(size=1000000)

    with TensorboardLogger() as logger:
        algo = SAC(
            env,
            ac_kwargs=ac_kwargs,
            rb_kwargs=rb_kwargs,
            update_every=1,
            update_after=100,
            batch_size=256,
            ent_coef="auto",
            logger=logger,
        )
        algo.train(n_steps=20000)

    env = gym.make("Pendulum-v1", render_mode="human")
    algo.test(env, n_episodes=10)


if __name__ == "__main__":
    main()
