import sys
from gymnasium import ObservationWrapper, Env
from gymnasium.spaces import Box, MultiDiscrete
import gymnasium as gym
sys.modules["gym"] = gym
from stable_baselines3 import DQN
from farl import FARL
import numpy as np


class CustomCartPoleEnv(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self._bins = 8
        self.observation_space = Box(low=0, high=1, shape=(self._bins ** 4,))

        # n = self.observation_space.shape[0]
        # self.observation_space = MultiDiscrete([self._bins] * n)

    @staticmethod
    def _rescale_observation(obs: np.ndarray) -> np.ndarray:
        obs[0] /= 4.8
        obs[1] = 1 / (1 + np.exp(-obs[1]))
        obs[3] = 1 / (1 + np.exp(-obs[3]))
        return obs

    def _discretize(self, obs: np.ndarray) -> np.ndarray:
        obs_bins = np.array(list(map(lambda x: int(x * self._bins), obs)))
        idx = sum(e * self._bins ** i for i, e in enumerate(obs_bins))
        new_obs = np.zeros(self._bins ** len(obs))
        new_obs[idx] = 1
        return new_obs

    def observation(self, observation: np.ndarray) -> np.ndarray:
        observation = self._rescale_observation(observation)
        observation = self._discretize(observation)
        return observation

    def step(self, action):
        obs, reward, terminated, truncated, info = super(CustomCartPoleEnv, self).step(action)
        if terminated:
            reward = -100
        return obs, reward, terminated, truncated, info


def main():
    env = CustomCartPoleEnv(
        gym.make('CartPole-v1')
    )

    # dqn = DQN.load('dqn_model', env=env)
    # model = DQN('MlpPolicy', env, verbose=1, exploration_initial_eps=1, exploration_fraction=0.1)
    # model.learn(total_timesteps=1_000_000, log_interval=1000)
    # dqn.save('dqn_model')
    model = FARL(env, exploration_initial_eps=1, exploration_fraction=0.1, exploration_final_eps=0.05,
                alpha=1e-3, verbose=True)
    model.learn(num_episodes=200_000, log_interval=1000)

    obs, _ = env.reset()
    done = False
    ret = 0
    while not done:
        action, _ = model.predict(observation=obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ret += reward
        print(done, action)
        env.render()
    print(f'Return: {ret}')


if __name__ == '__main__':
    main()
