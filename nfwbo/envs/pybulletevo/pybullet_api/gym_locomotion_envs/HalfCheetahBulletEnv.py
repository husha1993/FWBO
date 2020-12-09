import os, inspect

from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
import numpy as np
import pybullet
from gym import spaces
import gym.utils, gym.utils.seeding

from .WalkerBaseBulletEnv import WalkerBaseUrdfBulletEnv
from envs.pybulletevo.pybullet_api.robot_locomotors_evo.robot_locomotorsevo import HalfCheetahUrdf


class HalfCheetahUrdfBulletEnv(WalkerBaseUrdfBulletEnv):
  def __init__(self, timeStep, render=False, design=None, seed=0, design_mode='8lr'):
      self.robot = HalfCheetahUrdf(timeStep, design, design_mode)
      WalkerBaseUrdfBulletEnv.__init__(self, self.robot, timeStep, render, seed)
      self.observation_space = spaces.Box(-np.inf, np.inf, shape=[18], dtype=np.float32)

  def _isDone(self, state):
      if (self.robot.controlMethod == 'spd' or self.robot.controlMethod == 'torque'):
          if (self.robot.kneecollision(self.stadium_scene.ground_plane_mjcf[0])):
              return True
          else:
              return False
      else:
          return False
      # return False

  def disconnect(self):
      self._p.disconnect()

  def step(self, a):
      energy = 0
      if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
          for i in range(self.robot.subStep):
              force = self.robot.apply_action(a)
              energy += np.abs(force * self.robot.joint_speeds).sum()
              self.scene.global_step()

      state = self.robot.calc_state()  # also calculates self.joints_at_limit

      done = self._isDone(state)
      if not np.isfinite(state).all():
          print("~INF~", state)
          done = 0
      self.check_contact()

      reward = 1 * np.max(
          [state[-5] / 10.0, 0]) - 0.000 * energy / self.robot.subStep + 1 * 0.05 - 0.0 * np.linalg.norm(
          a) ** 2 + 0 * np.abs(state[-6] / 10)

      return state, reward, bool(done), {"energy": energy / self.robot.subStep, "vel": state[-5] / 10, "done": done,
                                         "pos": self.robot.robot_body.pose().xyz(), "torso_contact": self.robot.torso_contact, 'feet_contact': self.robot.feet_contact}
  def reset_design(self, design):
      self.stateId = -1
      self.scene = None
      self.robot.reset_design(self._p, design)

  def check_alive(self):
      self.robot.check_alive()
