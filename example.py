import sys

import numpy as np
from gymnasium import ObservationWrapper, Env
from gymnasium.spaces import MultiDiscrete
import gymnasium as gym
sys.modules["gym"] = gym
from stable_baselines3.common import logger
from stable_baselines3 import DQN
from farl.farl import FARL


class CustomCartPoleEnv(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self._bins = 10
        n = self.observation_space.shape[0]
        self.observation_space = MultiDiscrete([self._bins] * n)

    @staticmethod
    def _rescale_observation(obs: np.ndarray) -> np.ndarray:
        obs[0] = obs[0] / (4.8 * 2) + 0.5
        obs[1] = 1 / (1 + np.exp(-obs[1]))
        obs[2] = obs[2] / (0.418 * 2) + 0.5
        obs[3] = 1 / (1 + np.exp(-obs[3]))
        return obs

    def _discretize(self, obs: np.ndarray) -> np.ndarray:
        return np.array(list(map(lambda t: int(t[1] * self.observation_space.nvec[t[0]]), enumerate(obs))))

    def observation(self, observation: np.ndarray) -> np.ndarray:
        rescaled = self._rescale_observation(observation)
        discretized = self._discretize(rescaled)
        return discretized

    def step(self, action):
        obs, reward, terminated, truncated, info = super(CustomCartPoleEnv, self).step(action)
        if terminated:
            reward = -100
        return obs, reward, terminated, truncated, info


class CustomCliffWalkingEnv(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = MultiDiscrete([self.observation_space.n])
        self.steps_performed = 0

    def observation(self, observation: int) -> np.ndarray:
        return np.array([observation])

    def reset(self, **kwargs):
        self.steps_performed = 0
        return super(CustomCliffWalkingEnv, self).reset(**kwargs)

    def step(self, action):
        self.steps_performed += 1

        obs, rew, terminated, truncated, info = super(CustomCliffWalkingEnv, self).step(action)
        if self.steps_performed >= 100_000:
            truncated = True
            rew = -100
        return obs, rew, terminated, truncated, info


def main():
    # env = CustomCartPoleEnv(
    #     gym.make('CartPole-v1')
    # )
    env = CustomCliffWalkingEnv(
        gym.make("CliffWalking-v0")
    )

    model = DQN(
        policy='MlpPolicy',
        env=env,
        buffer_size=1,
        learning_starts=1,
        batch_size=1,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        verbose=1,
    )
    dqn_logger = logger.configure('.', ['stdout', 'json'])
    model.set_logger(dqn_logger)
    # model = FARL(env, exploration_initial_eps=1, exploration_fraction=0.1, exploration_final_eps=0.05,
    #              alpha=1e-5, gamma=0.99, verbose=True)
    # model.learn(total_timesteps=10_000_000, log_interval=1000, log_path='farl_progress.txt')
    model.learn(total_timesteps=10_000_000, log_interval=100)

    eps = 10
    ret_sum = 0
    for _ in range(eps):
        obs, _ = env.reset()
        done = False
        ret = 0
        while not done:
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ret += reward
        ret_sum += ret
    print(f'Return avg: {ret_sum / eps}')


if __name__ == '__main__':
    main()
