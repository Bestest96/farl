# Copied from
#    https://github.com/rldotai/rl-algorithms/blob/master/py3/gtd.py
#    https://github.com/dalmia/David-Silver-Reinforcement-learning/blob/master/Week 6 - Value Function Approximations/Q-Learning with Value Function Approximation.py
#    https://github.com/DLR-RM/stable-baselines3/blob/abffa161982b92e7f4b0186611cb73653bc2553b/stable_baselines3/common/utils.py#L95
# Modified with accordance to
#    https://www.sciencedirect.com/science/article/pii/S0378779621002108


import itertools
from typing import Callable, Optional

import gymnasium as gym
import numpy as np


Schedule = Callable[[float], float]


def get_linear_fn(start: float, end: float, end_fraction: float) -> Schedule:
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.
    :return: Linear schedule function.
    """

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


class FARL:
    def __init__(
            self,
            env: gym.Env,
            exploration_initial_eps: float = 1.0,
            exploration_final_eps: float = 0.05,
            exploration_fraction: float = 0.1,
            gamma: float = 0.9,
            alpha: float = 1e-7,
            beta: Optional[float] = None,
            verbose: bool = False,
    ):
        self.env = env

        self.exploration_schedule = get_linear_fn(
            exploration_initial_eps,
            exploration_final_eps,
            exploration_fraction,
        )
        self.exploration_rate = self.exploration_schedule(1.0)

        self.gamma = gamma
        self.alpha = alpha
        if beta is None:
            beta = alpha / 100
        self.beta = beta
        self.verbose = verbose

        if not isinstance(env.observation_space, gym.spaces.MultiDiscrete):
            raise Exception('This implementation of FARL only supports MultiDiscrete actions')

        self.obs_nvec = env.observation_space.nvec
        self.n_obs = np.prod(self.obs_nvec)
        self.n_act = env.action_space.n

        n = self.n_obs * self.n_act

        # self.w = np.zeros(n)
        self.w = np.random.uniform(low=-1/np.sqrt(n), high=1/np.sqrt(n), size=(n,))
        # self.h = np.zeros(n)
        self.h = np.random.uniform(low=-1/np.sqrt(n), high=1/np.sqrt(n), size=(n,))

    def learn(self, total_timesteps: int, log_interval: int = 100):
        stats = dict(
            episode_rewards=[],
            episode_lengths=[],
        )
        num_timesteps = 0
        i_episode = 0
        while num_timesteps < total_timesteps:
            self.exploration_rate = self.exploration_schedule(1 - num_timesteps / total_timesteps)
            stats['episode_rewards'].append(0)
            stats['episode_lengths'].append(0)

            state, _ = self.env.reset()
            state = self._multi_discrete_to_onehot(state)

            for t in itertools.count():

                action = np.random.choice(self.n_act, p=self._action_proba_distribution(state))

                new_state, reward, terminated, truncated, _ = self.env.step(action)
                new_state = self._multi_discrete_to_onehot(new_state)
                done = terminated or truncated

                stats['episode_rewards'][i_episode] += reward
                stats['episode_lengths'][i_episode] = t + 1

                self._update(state, action, reward, new_state, done)
                state = new_state

                if done:
                    break

            num_timesteps += t + 1
            i_episode += 1

            if self.verbose and i_episode % log_interval == 0 and i_episode > 0:
                log_range = slice(i_episode - log_interval, i_episode)
                avg_length = np.average(stats["episode_lengths"][log_range])
                print(f'Episode {i_episode}: '
                      f'avg reward = {np.average(stats["episode_rewards"][log_range])}, '
                      f'avg length = {avg_length}, '
                      f'eps = {round(self.exploration_rate, 4)}')

    def _action_proba_distribution(self, obs: np.ndarray) -> np.ndarray:
        p = np.ones(self.n_act, dtype=float) * self.exploration_rate / self.n_act
        q_values = self._get_q_values(obs)
        best_action = np.argmax(q_values)
        p[best_action] += (1.0 - self.exploration_rate)
        return p

    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        feature_matrix = np.zeros((self.n_act, self.n_obs * self.n_act), float)
        for a in range(self.n_act):
            feature_matrix[a, a * self.n_obs: (a + 1) * self.n_obs] = state
        q_values_for_state = np.dot(feature_matrix, self.w)
        return q_values_for_state

    def _multi_discrete_to_onehot(self, s) -> np.ndarray:
        onehot = np.zeros(self.n_obs)
        idx = sum(e * self.obs_nvec[i] ** i for i, e in enumerate(s))
        onehot[idx] = 1
        return onehot

    def _update(
            self,
            s: np.ndarray,
            a: int,
            r: float,
            sp: np.ndarray,
            done: bool,
    ):
        x, max_xp = np.zeros(self.n_obs * self.n_act), np.zeros(self.n_obs * self.n_act)
        x[a * self.n_obs:(a + 1) * self.n_obs] = s
        a_max = np.argmax(self._get_q_values(sp))
        max_xp[a_max * self.n_obs:(a_max + 1) * self.n_obs] = sp

        if not done:
            delta = r + self.gamma * np.dot(max_xp, self.w) - np.dot(x, self.w)
            self.w += self.alpha * (delta * x - self.gamma * np.dot(x, self.h) * max_xp)
        else:
            delta = r - np.dot(x, self.w)
            self.w += self.alpha * delta * x
        self.h += self.beta * (delta - np.dot(x, self.h)) * x

    def predict(self, observation: np.ndarray, deterministic: bool = False) -> (int, None):
        observation = self._multi_discrete_to_onehot(observation)
        if deterministic:
            return np.argmax(self._get_q_values(observation)), None
        return np.random.choice(self.n_act, p=self._action_proba_distribution(observation)), None
