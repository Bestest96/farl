import gym
from stable_baselines3 import DQN
from farl import FARL
import numpy as np


from gym import ObservationWrapper


class CustomCartPoleEnv(ObservationWrapper):
    def observation(self, observation):
        observation[0] /= 4.8
        observation[1] = 1 / (1 + np.exp(-observation[1])) - 0.5
        observation[3] = 1 / (1 + np.exp(-observation[3])) - 0.5
        return observation


class CustomMountainCarEnv(ObservationWrapper):
    def observation(self, observation):
        observation[0] = 1 / (1 + np.exp(-observation[0])) - 0.5
        observation[1] = 1 / (1 + np.exp(-observation[1])) - 0.5
        return observation


def main():
    # env = gym.make('MountainCar-v0')
    env = gym.make('CartPole-v1')
    # env = CustomMountainCarEnv(env)
    env = CustomCartPoleEnv(env)

    # dqn = DQN.load('dqn_model', env=env)
    # dqn = DQN('MlpPolicy', env, verbose=1, exploration_initial_eps=0.05)
    # dqn.learn(total_timesteps=1_500_000, log_interval=100)
    # dqn.save('dqn_model')
    farl = FARL(env, exploration_initial_eps=1, exploration_fraction=0.1, verbose=True)
    farl.learn(num_episodes=1_000_000, log_interval=2500)

    obs = env.reset()
    done = False
    while not done:
        action, _ = farl.predict(observation=obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        print(done, action)
        env.render()


if __name__ == '__main__':
    main()
