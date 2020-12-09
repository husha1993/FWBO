import os, inspect
import tempfile
import atexit
import xmltodict

import numpy as np
from pybullet_envs.robot_locomotors import WalkerBase
from pybullet_envs.robot_locomotors import Ant as Ant0


from .data.xml_parser import MuJoCoXmlRobot


def cleanup_func_for_tmp(filepath):
    os.remove(filepath)


currentdir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'data')


class Ant(Ant0):
  foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

  def __init__(self, design=None, design_mode='24lr'):
    self.design_mode = design_mode
    self.design = design
    self._adapted_xml_file = tempfile.NamedTemporaryFile(delete=False, prefix='ant_', suffix='.xml')
    self._adapted_xml_filepath = self._adapted_xml_file.name
    file = self._adapted_xml_filepath
    self._adapted_xml_file.close()
    atexit.register(cleanup_func_for_tmp, self._adapted_xml_filepath)
    self.adapt_xml(self._adapted_xml_filepath, design)
    WalkerBase.__init__(self, file, "torso", action_dim=8, obs_dim=28, power=2.5)

  def adapt_xml(self, file, design=None):
      if self.design_mode == '25lr':
          self._adatpt_xml_wo_human25lr(file, design)
      else:
          raise NotImplementedError

  def _adatpt_xml_wo_human25lr(self, file, design):

      robot = MuJoCoXmlRobot(os.path.join(currentdir, 'ant.xml'))
      robot.update(design, file)
      self.hlb = robot.get_height() #lower bound of body height
      self.hub = robot.get_height() + max(robot.get_params()[1] + robot.get_params()[3] + robot.get_params()[5],\
                                          robot.get_params()[7] + robot.get_params()[9] + robot.get_params()[11],\
                                          robot.get_params()[13] + robot.get_params()[15] + robot.get_params()[17],\
                                          robot.get_params()[19] + robot.get_params()[21] + robot.get_params()[23])

  def alive_bonus(self, z, pitch):
    return +1 if z > (self.hlb + 0.01) and z < self.hub else -1  # self.hlb is central sphere rad, die if it scrapes the ground
    #return +1 if z > (0.25 + 0.01) else -1  # 0.25 is central sphere rad, die if it scrapes the ground
    #return 1 #if self._alive == 1 else 0

  def calc_state(self):
    j = np.array([j.current_relative_position() for j in self.ordered_joints],
                 dtype=np.float32).flatten()
    # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
    self.joint_speeds = j[1::2]
    self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

    body_pose = self.robot_body.pose()
    parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
    self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2]
                    )  # torso z is more informative than mean z
    self.body_rpy = body_pose.rpy()
    z = self.body_xyz[2]
    if self.initial_z == None:
      self.initial_z = z
    r, p, yaw = self.body_rpy
    self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                        self.walk_target_x - self.body_xyz[0])
    #self.walk_target_dist = np.linalg.norm(
        #[self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
    self.walk_target_dist = np.linalg.norm([0, self.walk_target_x - self.body_xyz[0]])
    angle_to_target = self.walk_target_theta - yaw

    rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw),
                                                             np.cos(-yaw), 0], [0, 0, 1]])
    vx, vy, vz = np.dot(rot_speed,
                        self.robot_body.speed())  # rotate speed back to body point of view

    more = np.array(
        [
            z - self.initial_z,
            np.sin(angle_to_target),
            np.cos(angle_to_target),
            0.3 * vx,
            0.3 * vy,
            0.3 * vz,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r,
            p
        ],
        dtype=np.float32)
    return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

  def reset_design(self, bullet_client, design):
      self._adapted_xml_file = tempfile.NamedTemporaryFile(delete=False, prefix='ant_', suffix='.xml')
      self._adapted_xml_filepath = self._adapted_xml_file.name
      file = self._adapted_xml_filepath
      self._adapted_xml_file.close()
      atexit.register(cleanup_func_for_tmp, self._adapted_xml_filepath)
      self.adapt_xml(file, design)
      self.model_xml = file

      self.doneLoading = 0
      self.parts = None
      self.objects = []
      self.jdict = None
      self.ordered_joints = None
      self.robot_body = None
      self.robto_name = 'torso'
      bullet_client.resetSimulation()

      self.reset(bullet_client)

  def robot_specific_reset(self, bullet_client):
      WalkerBase.robot_specific_reset(self, bullet_client)
      body_pose = self.robot_body.pose()
      x, y, z = body_pose.xyz()
      self._initial_z = z

  def check_alive(self):
      self._alive = 1 if self.torso_contact != 1 else 0
