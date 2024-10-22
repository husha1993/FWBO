import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet
from pkg_resources import parse_version
import logging

from envs.pybulletevo.pybullet_api.pybullet_helpers import bullet_client

class MJCFBaseBulletEnv(gym.Env):
  """
	Base class for Bullet physics simulation loading MJCF (MuJoCo .xml) environments in a Scene.
	These environments create single-player scenes and behave like normal Gym environments, if
	you don't use multiplayer.
	"""

  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

  def __init__(self, robot, render=False, seed = None):
    self.scene = None
    self.physicsClientId = -1
    self.ownsPhysicsClient = 0
    self.camera = Camera()
    self.isRender = render
    self.robot = robot
    self.seed(seed)
    self._cam_dist = 3
    self._cam_yaw = 0
    self._cam_pitch = -30
    self._render_width = 320
    self._render_height = 240

    self.action_space = robot.action_space
    self.observation_space = robot.observation_space

  def configure(self, args):
    self.robot.args = args

  def seed(self, seed=None):
    self.np_random, seed = gym.utils.seeding.np_random(seed)
    self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
    return [seed]

  def reset(self):
    if (self.physicsClientId < 0):
      logging.debug('going to build a new client')
      self.ownsPhysicsClient = True

      if self.isRender:
        self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        #self._p.startStateLogging(self._p.STATE_LOGGING_VIDEO_MP4, "/cs/vml4/shah/mrd-baselines/Coadaptation/test.mp4")
      else:
        self._p = bullet_client.BulletClient()

      self.physicsClientId = self._p._client
      self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

    if self.scene is None:
      self.scene = self.create_single_player_scene(self._p)
    if not self.scene.multiplayer and self.ownsPhysicsClient:
      self.scene.episode_restart(self._p)

    self.robot.scene = self.scene

    self.frame = 0
    self.done = 0
    self.reward = 0
    dump = 0
    s = self.robot.reset(self._p)
    self.potential = self.robot.calc_potential()
    return s

  def render(self, mode='human', close=False):
    if mode == "human":
      self.isRender = True
    if mode != "rgb_array":
      return np.array([])

    lookat = [0, 0, 0]
    if (hasattr(self, 'robot')):
      if (hasattr(self.robot, 'body_xyz') or hasattr(self.robot, 'posCurrent')):
        #lookat = self.robot.body_xyz
        x, y, z = self.robot.robot_body.pose().xyz()
        lookat = [x+ 0.2, y, z-0.3]
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=lookat,
                                                            distance=self._cam_dist/1.5,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
    asp = float(self._render_width/2) / self._render_height
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=1,
                                                     nearVal=0.05,
                                                     farVal=100)
    self._render_width = 160
    self._render_height = 240
    (_, _, px, _, _) = self._p.getCameraImage(width=int(self._render_width/1),
                                              height=int(self._render_height/1),
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = np.reshape(np.array(px), (int(self._render_height/1), int(self._render_width/1), -1))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def close(self):
    if (self.ownsPhysicsClient):
      if (self.physicsClientId >= 0):
        self._p.disconnect()
    self.physicsClientId = -1

  def HUD(self, state, a, done):
    pass

  # def step(self, *args, **kwargs):
  # 	if self.isRender:
  # 		base_pos=[0,0,0]
  # 		if (hasattr(self,'robot')):
  # 			if (hasattr(self.robot,'body_xyz')):
  # 				base_pos = self.robot.body_xyz
  # 				# Keep the previous orientation of the camera set by the user.
  # 				#[yaw, pitch, dist] = self._p.getDebugVisualizerCamera()[8:11]
  # 				self._p.resetDebugVisualizerCamera(3,0,0, base_pos)
  #
  #
  # 	return self.step(*args, **kwargs)
  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed


class Camera:

  def __init__(self):
    pass

  def move_and_look_at(self, i, j, k, x, y, z):
    lookat = [x, y, z]
    distance = 10
    yaw = 10
    self._p.resetDebugVisualizerCamera(distance, yaw, -20, lookat)
