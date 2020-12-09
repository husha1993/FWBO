from gym import spaces
import numpy as np
import copy


class CLEnv(object):
    '''
    controllable task-length environment for locomotion tasks
    '''
    def __init__(self, env):
        self._env = env
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._initial_state = self._env.reset()
        self.current_step = 0
        self._task_t = 1000 #standard settings for gym based tasks

    def render(self, mode="human"):
        return self._env.render(mode)

    def step(self, a):
        state, reward, done, info = self._env.step(a)
        self.current_step +=1
        info['fail'] = done
        if(self.current_step > self._task_t):
            done = True
        return state, reward, done, info

    def reset(self, return_info=False):
        state, info = self._env.reset()
        self._initial_state = state
        self.current_step = 0
        if return_info:
            return state, info
        return state

    def set_task_t(self, t):
        """ Set the max t an episode can have under training mode for curriculum learning
        """
        self._task_t = min(t, 1000)

    def set_new_design(self, design):
        self._env.set_new_design(design)

    def get_current_design(self):
        return self._env.get_current_design()


