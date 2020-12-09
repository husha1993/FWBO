import os, inspect
import time
import logging
_log = logging.getLogger(__name__)

from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
import numpy as np
import pybullet
from gym import spaces
import gym.utils, gym.utils.seeding

from .env_bases import MJCFBaseBulletEnv
from envs.pybulletevo.pybullet_api.robot_locomotors_evo.robot_locomotorsevo import HalfCheetahUrdf

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class WalkerBaseUrdfBulletEnv(MJCFBaseBulletEnv):
      def __init__(self, robot, timeStep, render=False, seed = None):
        self.timeStep = timeStep
        MJCFBaseBulletEnv.__init__(self, robot, render, seed)
        self.stateId=-1
      def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

      def create_single_player_scene(self, bullet_client):
        self.stadium_scene = SinglePlayerStadiumScene(bullet_client, gravity=9.8, timestep=self.timeStep, frame_skip=1)
        return self.stadium_scene

      def check_contact(self):
        self.robot.kneecollision(self.stadium_scene.ground_plane_mjcf[0])

      def reset(self):
        if (self.stateId>=0):
          self._p.restoreState(self.stateId)

        r = MJCFBaseBulletEnv.reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self._p,
          self.stadium_scene.ground_plane_mjcf)

        # self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in
        #              self.foot_ground_object_names])
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,1)
        if (self.stateId<0):
          self.stateId=self._p.saveState()

        self._p.setGravity(0,0,-9.8)
        self.check_contact()
        count = 0
        # _log.debug('count={} within while loop'.format(count))
        while (not (sum(self.robot.feet_contact[:2]) & sum(self.robot.feet_contact[3:5]))) :
            # a stable initial state is defined as 1) all feets contact with grounds or 2)torso contact with grounds
            for i in range(5):
                self._p.stepSimulation()
            self.check_contact()
            count += 1
            if count >= 100:
                break

        for _ in range(200):
            self.robot.reset_position()
            self.scene.global_step()
        self.robot.reset_position_final()
        self._p.setGravity(0,0,-9.8)
        r = self.robot.calc_state()
        self.robot._initial_z = r[-1]
        self.robot.initial_z = None
        body_pose = self.robot.robot_body.pose()
        x, y, z = body_pose.xyz()
        self.robot.posPre = np.array([x,y,z])
        r = self.robot.calc_state()
        self.check_contact()
        self.check_alive()
        return r, {'vel':self.robot_body.speed(), 'pos': self.robot.robot_body.pose().xyz(), 'torso_contact': self.robot.torso_contact, \
           'feet_contact': self.robot.feet_contact, 'alive': self.robot._alive}

      def _isDone(self):
        return self._alive < 0

      def step(self, a):
        pass



