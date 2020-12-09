import os, inspect
import tempfile
import atexit
import xmltodict

import numpy as np

from .robot_bases import BodyPart
from .robot_bases import XmlBasedRobot, MJCFBasedRobot, URDFBasedRobot, StablePDController


def cleanup_func_for_tmp(filepath):
    os.remove(filepath)


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class WalkerBaseUrdf(URDFBasedRobot):
    '''
    urdf based robot for stable simulation
    '''

    def __init__(self, model_urdf, robot_name, action_dim, obs_dim, kps, kds, forces, timeStep, basePosition=[0, 0, 0],
                 baseOrientation=[0, 0, 0, 1], fixed_base=True, self_collision=False):
        URDFBasedRobot.__init__(self, model_urdf, robot_name, action_dim, obs_dim, basePosition=basePosition,
                                baseOrientation=baseOrientation, fixed_base=fixed_base, self_collision=self_collision)
        self.spd = None
        self.a = np.zeros((action_dim,))
        self.b = np.zeros((action_dim,))
        self.set_joint = False
        self.joints_lower = []
        self.joints_upper = []
        self.kps = kps
        self.kds = kds
        self.timeStep = timeStep
        self.forces = forces

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self._reset_position = []
        if (self.set_joint == False):
            # don't need to waste much time on useless resetting
            self.spd = StablePDController(self.objects, self._p, self.kps, self.kds, self.forces, self.timeStep)
            for i in range(len(self.ordered_joints)):
                self.a[i] = np.max([self.ordered_joints[i].upperLimit, -self.ordered_joints[i].lowerLimit]) * 2
                self.joints_lower.append(self.ordered_joints[i].lowerLimit)
                self.joints_upper.append(self.ordered_joints[i].upperLimit)
                self.b[i] = 0

            self.joints_lower = np.array(self.joints_lower)
            self.joints_upper = np.array(self.joints_upper)
            self.set_joint = True
        for j in range(len(self.ordered_joints)):
            #pos = self.np_random.uniform(low=0.0 * self.ordered_joints[j].lowerLimit, high=0.0 * self.ordered_joints[j].upperLimit)
            pos = self.np_random.uniform(low=-0.01, high=0.01)
            self.ordered_joints[j].reset_current_position(pos, 0)
            self._reset_position.append(pos)
        # foor contact information, may be not needed
        # self.feet = [self.parts[f] for f in self.foot_list]
        # self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
        self.scene.actor_introduce(self)
        self.initial_z = None

    def reset_position(self):
        # set velocities of joints to zero
        for j, pos in zip(self.ordered_joints, self._reset_position):
            j.set_velocity(0)
            #j.reset_current_position(0, 0)

    def reset_position_final(self):
        # clean force caches of joint motors
        for j, pos in zip(self.ordered_joints, self._reset_position):
            j.disable_motor()

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        if (self.controlMethod == 'spd'):
            a = a * self.a + self.b
            # a = np.array([0,0,0,0,0,0])
            a = np.clip(a, self.joints_lower, self.joints_upper)
            forces = self.spd.getForce(a)
            forces_list = forces.tolist()
            # print(np.linalg.norm(forces[3:]))
        else:
            a = np.clip(a, -1, 1)
            forces = a * self.coff
            forces_list = forces.tolist()
            forces_list = [0, 0, 0] + forces_list
        self._p.setJointMotorControlArray(self.objects, self.spd.joint_indices, self._p.TORQUE_CONTROL,
                                          forces=forces_list)
        return np.array(forces_list[3:])



