import gym
from gym import spaces
import numpy as np
from .pybullet_api.gym_locomotion_envs.gym_locomotion_envs import HalfCheetahUrdfBulletEnv, AntBulletEnv
import copy


class BasicEvoEnv(gym.Env):
    def __init__(self):
        pass

    def render(self, mode='human'):
        return self._env.render(mode)

    def step(self, a):
        pass

    def reset(self):
        pass

    def set_new_design(self, vec):
        pass

    def get_current_design(self):
        pass


class HalfCheetahUrdfEnv(BasicEvoEnv):
    def __init__(self, config, seed = None):
        self._config = config
        self._render = self._config['env']['render']
        self._current_design = [0.145 / 0.12, 0.15 / 0.12, 0.094 / 0.12, 0.133 / 0.12, 0.106 / 0.12, 0.07 / 0.12, 1,1,1,1,1,1,1]
        self._config_numpy = np.array(self._current_design)
        self._env = HalfCheetahUrdfBulletEnv(0.005, self._render, self._current_design, seed, design_mode=self._config['design_mode'])
        if self._config['rl_algorithm_config']['concate_design_2_state']:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=[self._env.observation_space.shape[0] + self._config['design_dim']], dtype=np.float32)#env.observation_space
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=[self._env.observation_space.shape[0]], dtype=np.float32)
        self.action_space = self._env.action_space
        self._initial_state = self._env.reset()

    def step(self, a):
        #info = {}
        state, reward, done, info = self._env.step(a)
        if self._config['rl_algorithm_config']['concate_design_2_state']:
            state = np.append(state, self._config_numpy)
        return state, reward, done, info

    def reset(self):
        state, info = self._env.reset()
        self._initial_state = state
        if self._config['rl_algorithm_config']['concate_design_2_state']:
            state = np.append(state, self._config_numpy)
        return state, info

    def set_new_design(self, vec):
        #assert type(vec) == int,'env can only receive one design per once'
        self._env.reset_design(vec)
        self._current_design = vec
        self._config_numpy = np.array(vec)

    def get_random_design(self):
        optimized_params = np.random.uniform(low=0.8, high=2.0, size=6)#todo, another hand-engineered constrain
        return optimized_params

    def get_current_design(self):
        return copy.copy(self._current_design)


class AntEnv(BasicEvoEnv):
    def __init__(self, config={}, seed=None):
        self._config = config
        self._render = self._config['env']['render']
        self._current_design = [0.25, 0.28284271247461906, 0.08, 0.28284271247461906, 0.08, 0.5656854249492381, 0.08, 0.28284271247461906, 0.08, 0.28284271247461906, 0.08, 0.5656854249492381, 0.08, 0.28284271247461906, 0.08, \
                                0.28284271247461906, 0.08, 0.5656854249492381, 0.08, 0.28284271247461906, 0.08, 0.28284271247461906, 0.08, 0.5656854249492381, 0.08]
        self._config_numpy = np.array(self._current_design)
        self._env = AntBulletEnv(render=self._render, design=self._current_design, seed=seed, design_mode=self._config['design_mode'])
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=[self._env.observation_space.shape[0]],
                                            dtype=np.float32)
        self.action_space = self._env.action_space
        self._initial_state = self._env.reset()

    def step(self, a):
        state, reward, done, info = self._env.step(a)
        return state, reward, done, info

    def reset(self):
        state, info = self._env.reset()
        self.initial_state = state
        return state, info

    def set_new_design(self, vec):
        self._env.reset_design(vec)
        self._current_design = vec
        self._config_numpy = np.array(vec)

    def get_current_design(self):
        return copy.copy(self._current_design)