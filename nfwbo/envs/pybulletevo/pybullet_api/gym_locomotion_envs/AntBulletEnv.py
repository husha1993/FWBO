import os, inspect

from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
import numpy as np
import pybullet
from gym import spaces
import gym.utils, gym.utils.seeding

from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv

from envs.pybulletevo.pybullet_api.robot_locomotors_evo.Ant import Ant


class AntBulletEnv(WalkerBaseBulletEnv):
    def __init__(self, render=False, design=None, seed=None, design_mode='24lr'):
        self.robot = Ant(design, design_mode)
        WalkerBaseBulletEnv.__init__(self, self.robot, render)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=[28], dtype=np.float32)
        self.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def disconnect(self):
        self._p.disconnect()

    def step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        self._alive = float(
            self.robot.alive_bonus(
                state[0] + self.robot.initial_z,
                self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        done = self._isDone()
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
                self.robot.feet
        ):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean(
        ))  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("alive=")
            print(self._alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        self.rewards = [
            0.1*self._alive, progress
        ]
        if (debugmode):
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {'v(m/s)':self.robot_body.speed(), 'pose(m)': self.robot.robot_body.pose().xyz(), 'rpy': self.robot.body_rpy,\
                'done': done, 'ori': self.robot.robot_body.pose().orientation(), 'feet_contact': self.robot.feet_contact}

    def reset(self):
        if (self.stateId >= 0):
            # print("restoreState self.stateId:",self.stateId)
            self._p.restoreState(self.stateId)

        r = MJCFBaseBulletEnv.reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
            self._p, self.stadium_scene.ground_plane_mjcf)
        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex],
                                self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        if (self.stateId < 0):
            self.stateId = self._p.saveState()
            # print("saving state self.stateId:",self.stateId)

        return r, {'v(m/s)':self.robot_body.speed(), 'pose(m)': self.robot.robot_body.pose().xyz(), 'rpy': self.robot.body_rpy, \
           'ori': self.robot.robot_body.pose().orientation(), 'feet_contact': self.robot.feet_contact}

    def reset_design(self, design):
        self.stateId = -1
        self.scene = None
        self.robot.reset_design(self._p, design)

    def check_alive(self):
        self.robot.check_alive()

    def set_new_design(self, design):
        self.stateId = -1
        self.scene = None
        self.robot.reset_design(self._p, design)
