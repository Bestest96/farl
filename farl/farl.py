# Copied from
#    https://github.com/rldotai/rl-algorithms/blob/master/py3/gtd.py
#    https://github.com/dalmia/David-Silver-Reinforcement-learning/blob/master/Week 6 - Value Function Approximations/Q-Learning with Value Function Approximation.py
#    https://github.com/DLR-RM/stable-baselines3/blob/abffa161982b92e7f4b0186611cb73653bc2553b/stable_baselines3/common/utils.py#L95
# Modified with accordance to
#    https://www.sciencedirect.com/science/article/pii/S0378779621002108


import itertools
from functools import reduce
from typing import Callable, Optional
import pickle

import gymnasium as gym
import numpy as np
import torch

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


def get_eps_schedule(start: float, mid: float, mid_fraction: float) -> Schedule:
    explore_schedule = get_linear_fn(start, 2 * mid, mid_fraction)
    decay_schedule = get_linear_fn(2 * mid, 1e-6, 1 - mid_fraction)

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > mid_fraction:
            return decay_schedule(progress_remaining + mid_fraction)
        else:
            return explore_schedule(progress_remaining)

    return func


class FARL:
    def __init__(
            self,
            env: gym.Env,
            eval_env: gym.Env,
            exploration_initial_eps: float = 1.0,
            exploration_final_eps: float = 0.05,
            exploration_fraction: float = 0.1,
            gamma: float = 0.9,
            alpha: float = 1e-5,
            beta: Optional[float] = None,
            verbose: bool = False,
            feature_representation: str = 'fsr',
    ):
        if not isinstance(env.observation_space, gym.spaces.MultiDiscrete):
            raise Exception('This implementation of FARL only supports MultiDiscrete actions')

        self.env = env
        self.eval_env = eval_env
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # self.exploration_schedule = get_eps_schedule(
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
        if feature_representation not in ['fsr', 'tabular']:
            raise RuntimeError(f'Unknown feature representation {feature_representation}.'
                               f'Only "fsr" (fixed sparse representation) and "tabular" are supported.')
        self.feature_representation = feature_representation

        self.obs_nvec = env.observation_space.nvec
        if feature_representation == 'fsr':
            self.n_obs = sum(self.obs_nvec)
        else:  # tabular
            self.n_obs = np.prod(self.obs_nvec)
        self.n_act = env.action_space.n

        n = self.n_obs * self.n_act + 1
        # self.w = np.random.uniform(low=-1/np.sqrt(n), high=1/np.sqrt(n), size=(n,))
        # self.h = np.random.uniform(low=-1/np.sqrt(n), high=1/np.sqrt(n), size=(n,))
        high = 1/np.sqrt(n)
        low = -1/np.sqrt(n)
        self.w = (high - low) * torch.rand(size=(n,), device=self.device) + low
        self.h = (high - low) * torch.rand(size=(n,), device=self.device) + low

        # bias
        self.w[-1] = 0
        self.h[-1] = 0

    def _extract_features(self, s: torch.Tensor) -> torch.Tensor:
        if self.feature_representation == 'fsr':
            return self._multi_discrete_to_binary(s)
        # tabular
        return self._multi_discrete_to_onehot(s)

    def learn(self, total_timesteps: int, log_interval: int = 100, save_path: Optional[str] = None, log_path: Optional[str] = None, seed: Optional[int] = None):
        stats = dict(
            episode_rewards=[],
            episode_lengths=[],
            eval_rewards=[]
        )
        num_timesteps = 0
        i_episode = 0
        best_eval_reward = -np.inf
        while num_timesteps < total_timesteps:
            self.exploration_rate = self.exploration_schedule(1 - num_timesteps / total_timesteps)
            stats['episode_rewards'].append(0)
            stats['episode_lengths'].append(0)

            state, _ = self.env.reset()
            state = self._extract_features(torch.from_numpy(state).to(self.device))

            for t in itertools.count():

                action = torch.multinomial(self._action_proba_distribution(state), 1)

                new_state, reward, terminated, truncated, _ = self.env.step(action.item())
                new_state = self._extract_features(torch.from_numpy(new_state).to(self.device))
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
                state, _ = self.eval_env.reset(seed=seed)
                state = self._extract_features(torch.from_numpy(state).to(self.device))
                stats["eval_rewards"].append(0.0)

                for t in itertools.count():

                    action = torch.argmax(self._get_q_values(state))

                    new_state, reward, terminated, truncated, _ = self.eval_env.step(action.item())
                    new_state = self._extract_features(torch.from_numpy(new_state).to(self.device))
                    done = terminated or truncated

                    stats["eval_rewards"][-1] += reward

                    state = new_state

                    if done:
                        break

                if stats["eval_rewards"][-1] > best_eval_reward:
                    best_eval_reward = stats["eval_rewards"][-1]
                    self.save(f"{save_path}/farl_best.zip")

                log_range = slice(i_episode - log_interval, i_episode)
                avg_length = np.average(stats["episode_lengths"][log_range])
                logs = f'Episode {i_episode}, ' \
                       f'total_timesteps {num_timesteps}, ' \
                       f'avg_train_reward {np.average(stats["episode_rewards"][log_range])}, ' \
                       f'avg_length {avg_length}, ' \
                       f'eps {round(self.exploration_rate, 4)} ' \
                       f'best_eval_reward {best_eval_reward}'
                print(logs)
                if log_path:
                    with open(log_path, 'a') as f:
                        f.write(logs + '\n')

    def _action_proba_distribution(self, obs: torch.Tensor) -> torch.Tensor:
        p = torch.ones(self.n_act, device=self.device) * self.exploration_rate / self.n_act
        q_values = self._get_q_values(obs)
        best_action = torch.argmax(q_values)
        p[best_action] += (1.0 - self.exploration_rate)
        return p

    def _get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        feature_matrix = torch.zeros((self.n_act, self.n_obs * self.n_act + 1), device=self.device)
        for a in range(self.n_act):
            feature_matrix[a, a * self.n_obs: (a + 1) * self.n_obs] = state
        feature_matrix[:, -1] = 1  # bias
        q_values_for_state = torch.matmul(feature_matrix, self.w)
        return q_values_for_state

    def _multi_discrete_to_onehot(self, s: torch.Tensor) -> torch.Tensor:
        onehot = torch.zeros(self.n_obs, device=self.device)
        idx = reduce(lambda acc, t: acc * self.obs_nvec[t[0]] + t[1], list(enumerate(s))[1:], s[0])
        onehot[idx] = 1
        return onehot

    def _multi_discrete_to_binary(self, s) -> torch.Tensor:
        binary = torch.zeros(self.n_obs, device=self.device)
        idx = 0
        for n_i, s_i in zip(self.obs_nvec, s):
            binary[idx + s_i] = 1
            idx += n_i
        return binary

    def _update(
            self,
            s: torch.Tensor,
            a: int,
            r: float,
            sp: torch.Tensor,
            done: bool,
    ):
        x, max_xp = torch.zeros(self.n_obs * self.n_act + 1, device=self.device), torch.zeros(self.n_obs * self.n_act + 1, device=self.device)
        x[a * self.n_obs:(a + 1) * self.n_obs] = s
        a_max = torch.argmax(self._get_q_values(sp))
        max_xp[a_max * self.n_obs:(a_max + 1) * self.n_obs] = sp

        if not done:
            delta = r + self.gamma * torch.dot(max_xp, self.w) - torch.dot(x, self.w)
            self.w += self.alpha * (delta * x - self.gamma * torch.dot(x, self.h) * max_xp)
        else:
            delta = r - torch.dot(x, self.w)
            self.w += self.alpha * delta * x
        self.h += self.beta * (delta - torch.dot(x, self.h)) * x

    def predict(self, observation: torch.Tensor, deterministic: bool = False) -> (int, None):
        observation = self._extract_features(observation)
        if deterministic:
            return torch.argmax(self._get_q_values(observation)), None
        return torch.multinomial(self._action_proba_distribution(observation), 1), None

    def save(self, path: str):
        dct = self.__dict__.copy()
        # sometimes can't pickle env
        del dct['env']
        del dct['eval_env']
        # can't pickle local function
        del dct['exploration_schedule']
        with open(path, 'wb') as f:
            pickle.dump(dct, f)

    @classmethod
    def load(cls, path: str, env, eval_env = None) -> 'FARL':
        with open(path, 'rb') as f:
            dct = pickle.load(f)
        obj = FARL(env, eval_env)
        dct.update(dict(
            env=env,
            eval_env=eval_env,
            exploration_schedule=obj.__dict__['exploration_schedule'],
        ))
        obj.__dict__ = dct
        return obj
