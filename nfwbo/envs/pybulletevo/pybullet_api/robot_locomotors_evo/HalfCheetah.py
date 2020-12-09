import os, inspect
import tempfile
import atexit
import xmltodict

import numpy as np

from .robot_bases import BodyPart
from .robot_bases import XmlBasedRobot, MJCFBasedRobot, URDFBasedRobot, StablePDController
from .WalkerBase import WalkerBaseUrdf


def cleanup_func_for_tmp(filepath):
    os.remove(filepath)


currentdir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'data')


class HalfCheetahUrdf(WalkerBaseUrdf):
    foot_list = ["ffoot", "fshin", "fthigh", "bfoot", "bshin", "bthigh"]
    def __init__(self, timeStep, design=None, design_mode='8lr'):
        self.design_mode = design_mode
        kps = [0, 0, 0, 500, 500, 500, 500, 500, 500]
        kds = [0, 0, 0, 50, 50, 50, 50, 50, 50]

        def actuate(x):
            x = (x - 1) * (8 / 0.5)
            return 1 + (1 / (1 + np.exp(-x)) - 0.5) * 1.8

        #self.coff = np.array([120, 90, 60, 120, 90, 60])  # mimic setting in roboschool
        self._adapted_xml_file = tempfile.NamedTemporaryFile(delete=False, prefix='halfcheetah_', suffix='.urdf')
        self._adapted_xml_filepath = self._adapted_xml_file.name
        file = self._adapted_xml_filepath
        self.controlMethod = "spd"
        self._adapted_xml_file.close()
        atexit.register(cleanup_func_for_tmp, self._adapted_xml_filepath)
        self.adapt_xml(self._adapted_xml_filepath, design)

        fbase = 120
        self.forces = [0, 0, 0]
        self.forces.extend([fbase * i for i in self.forces_r])
        self.subStep = 3
        self.frameSkip = 1
        WalkerBaseUrdf.__init__(self, file, 'torso', 6, 26, kps, kds, self.forces, timeStep, basePosition=[0, 0, 0],
                                baseOrientation=[0, 0, 0, 1], fixed_base=True, self_collision=True)

    def adapt_xml(self, file, design=None):
        '''
        choose to adapt from the file with human knowledge or without knowledge
        '''

        if self.design_mode == '13lr':
            self.forces_r = [1, 0.75, 0.5, 1, 0.75, 0.5]
            self._adapt_xml_wo_human13lr(file, design)
        else:
            raise NotImplementedError

    def robot_specific_reset(self, bullet_client):
        WalkerBaseUrdf.robot_specific_reset(self, bullet_client)
        body_pose = self.robot_body.pose()
        x, y, z = body_pose.xyz()
        self._initial_z = z
        self.posPre = np.array([x, y, z])

    def calc_state(self):
        j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        self.joint_pos = j[0::2]
        self.joint_speeds = j[1::2]
        body_pose = self.robot_body.pose()
        x, y, z = body_pose.xyz()
        self.posCurrent = np.array([x, y, z])
        z = z - self._initial_z
        r, p, yaw = self.body_rpy = body_pose.rpy()
        xv, yv, zv = self.robot_body.speed()
        vr, vp, vy = self.robot_body.speed_angular()
        # state = np.array([xv, vp, zv, p, z])
        v = (self.posCurrent - self.posPre) / (self.timeStep * self.subStep * self.frameSkip)
        state = np.array([v[2], v[0], vp, zv, p, z])
        state = np.append(self.joint_speeds, state)
        state = np.append(self.joint_pos, state)
        self.posPre = self.posCurrent
        return state

    def alive_bonus(self, z, pitch):
        return 0

    def kneecollision(self, groundId):
        # define the state when the bthigh or fthigh or torso contact the grounds as failure
        contact_ids_bthigh = self._p.getContactPoints(self.objects, groundId, 3)
        contact_ids_bthigh_flag = 1 if len(contact_ids_bthigh) > 0 else 0

        contact_ids_bsh = self._p.getContactPoints(self.objects, groundId, 4)
        contact_ids_bsh_flag = 1 if len(contact_ids_bsh) > 0 else 0

        contact_ids_bfoot = self._p.getContactPoints(self.objects, groundId, 5)
        contact_ids_bfoot_flag = 1 if len(contact_ids_bfoot) > 0 else 0

        contact_ids_fthigh = self._p.getContactPoints(self.objects, groundId, 6)
        contact_ids_fthigh_flag = 1 if len(contact_ids_fthigh) > 0 else 0

        contact_ids_fsh = self._p.getContactPoints(self.objects, groundId, 7)
        contact_ids_fsh_flag = 1 if len(contact_ids_fsh) > 0 else 0

        contact_ids_ffoot = self._p.getContactPoints(self.objects, groundId, 8)
        contact_ids_ffoot_flag = 1 if len(contact_ids_ffoot) > 0 else 0

        contact_ids_torso = self._p.getContactPoints(self.objects, groundId, 2)
        contact_ids_torso_flag = 1 if len(contact_ids_torso) > 0 else 0

        #foot_list = ["ffoot", "fshin", "fthigh", "bfoot", "bshin", "bthigh"]
        self.feet_contact = [contact_ids_ffoot_flag, contact_ids_fsh_flag, contact_ids_fthigh_flag,
                             contact_ids_bfoot_flag, contact_ids_bsh_flag, contact_ids_bthigh_flag]
        self.torso_contact = contact_ids_torso_flag

        return len(contact_ids_torso) > 0  #(len(contact_ids_bthigh) > 0 or len(contact_ids_fthigh) > 0 or len(contact_ids_torso) > 0)  # or len(contact_ids_fsh)>0 or len(contact_ids_bsh)>0):

    def _adapt_xml_wo_human13lr(self, file, design=None):
        '''
        adapt from half_cheetah.urdf which **incorporate** human knowledge
        '''
        with open(os.path.join(currentdir, 'half_cheetah.urdf'), 'r') as fd:
            xml_string = fd.read()
        if design is None:
            bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r, radius_r, radius_torso_r = np.random.uniform(low=0.3, high=1.8,
                                                                                                   size=8)
        else:
            bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r, r1, r2, r3, r4, r5, r6, r7 = design

        (bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r) = (2 * bth_r, 2 * bsh_r, 2 * bfo_r, 2 * fth_r, 2 * fsh_r, 2 * ffo_r)

        baseLength = 0.12  # to choose better region of function

        height = max(baseLength / 2 * bth_r + baseLength / 2 * bsh_r + baseLength / 2 * bfo_r,
                     baseLength / 2 * fth_r + baseLength / 2 * fsh_r + baseLength / 2 * ffo_r)
        height *= 2.0 + 0.01
        xml_dict = xmltodict.parse(xml_string)
        xml_dict['robot']['joint'][0]['origin']['@xyz'] = "0 0 {}".format(height)

        radius_torso_r = r7
        xml_dict['robot']['link'][3]['inertial']['mass']['@value'] = '{}'.format(9.45 * radius_torso_r ** 2)
        xml_dict['robot']['link'][3]['collision'][0]['geometry']['capsule']["@radius"] = '{}'.format(
            0.046 * radius_torso_r)
        xml_dict['robot']['link'][3]['collision'][1]['geometry']['capsule']["@radius"] = '{}'.format(
            0.046 * radius_torso_r)

        # Back thigh link
        # inertial frame
        xml_dict['robot']['link'][4]['inertial']['mass']['@value'] = '{}'.format(2.335 / 2 * bth_r * r1 ** 2)
        xml_dict['robot']['link'][4]['inertial']['origin']['@xyz'] = '0.1 0 -0.13'
        xml_dict['robot']['link'][4]['inertial']['origin']['@xyz'] = '{} 0 {}'.format(.1 * bth_r * baseLength / 0.29,
                                                                                      -.13 * bth_r * baseLength / 0.29)
        # collision frame
        xml_dict['robot']['link'][4]['collision']['origin']['@xyz'] = '0.1 0 -0.13'
        xml_dict['robot']['link'][4]['collision']['origin']['@xyz'] = '{} 0 {}'.format(.1 * bth_r * baseLength / 0.29,
                                                                                       -.13 * bth_r * baseLength / 0.29)
        # collision shape
        xml_dict['robot']['link'][4]['collision']['geometry']['capsule']["@length"] = '0.29000'
        xml_dict['robot']['link'][4]['collision']['geometry']['capsule']["@length"] = '{}'.format(
            0.29 * bth_r * baseLength / 0.29)
        xml_dict['robot']['link'][4]['collision']['geometry']['capsule']["@radius"] = '{}'.format(0.046 * r1)
        # bshin joint position
        xml_dict['robot']['joint'][4]['origin']['@xyz'] = '0.16 0 -0.25'
        xml_dict['robot']['joint'][4]['origin']['@xyz'] = '{} 0 {}'.format(0.16 * bth_r * baseLength / 0.29,
                                                                           -0.25 * bth_r * baseLength / 0.29)

        # Back shin link
        # inertial frame
        xml_dict['robot']['link'][5]['inertial']['mass']['@value'] = '{}'.format(2.40 / 2 * bsh_r * r2 ** 2)
        xml_dict['robot']['link'][5]['inertial']['origin']['@xyz'] = '-0.14 0 -0.07'
        xml_dict['robot']['link'][5]['inertial']['origin']['@xyz'] = '{} 0 {}'.format(-0.14 * bsh_r * baseLength / 0.3,
                                                                                      -.07 * bsh_r * baseLength / 0.3)
        # collision frame
        xml_dict['robot']['link'][5]['collision']['origin']['@xyz'] = '-0.14 0 -0.07'
        xml_dict['robot']['link'][5]['collision']['origin']['@xyz'] = '{} 0 {}'.format(-0.14 * bsh_r * baseLength / 0.3,
                                                                                       -.07 * bsh_r * baseLength / 0.3)
        # collision shape
        xml_dict['robot']['link'][5]['collision']['geometry']['capsule']["@length"] = '0.3'
        xml_dict['robot']['link'][5]['collision']['geometry']['capsule']["@length"] = '{}'.format(
            0.3 * bsh_r * baseLength / 0.3)
        xml_dict['robot']['link'][5]['collision']['geometry']['capsule']["@radius"] = '{}'.format(0.046 * r2)

        xml_dict['robot']['joint'][5]['origin']['@xyz'] = '-0.28 0 -0.14'
        xml_dict['robot']['joint'][5]['origin']['@xyz'] = '{} 0 {}'.format(-0.28 * bsh_r * baseLength / 0.3,
                                                                           -0.14 * bsh_r * baseLength / 0.3)

        # Back foot link
        # inertial frame
        xml_dict['robot']['link'][6]['inertial']['mass']['@value'] = '{}'.format(1.5 / 2 * bfo_r * r3 ** 2)
        xml_dict['robot']['link'][6]['inertial']['origin']['@xyz'] = '0.03 0 -0.09'
        xml_dict['robot']['link'][6]['inertial']['origin']['@xyz'] = '{} 0 {}'.format(0.03 * bfo_r * baseLength / 0.188,
                                                                                      -.09 * bfo_r * baseLength / 0.188)
        # collision frame
        xml_dict['robot']['link'][6]['collision']['origin']['@xyz'] = '0.03 0 -0.09'
        xml_dict['robot']['link'][6]['collision']['origin']['@xyz'] = '{} 0 {}'.format(
            0.03 * bfo_r * baseLength / 0.188, -.09 * bfo_r * baseLength / 0.188)
        # collision shape
        xml_dict['robot']['link'][6]['collision']['geometry']['capsule']["@length"] = '0.188'
        xml_dict['robot']['link'][6]['collision']['geometry']['capsule']["@length"] = '{}'.format(
            0.188 * bfo_r * baseLength / 0.188)
        xml_dict['robot']['link'][6]['collision']['geometry']['capsule']["@radius"] = '{}'.format(0.046 * r3)

        # fore thigh link
        # inertial frame
        xml_dict['robot']['link'][7]['inertial']['mass']['@value'] = '{}'.format(2.17598 / 2 * fth_r * r4 ** 2)
        xml_dict['robot']['link'][7]['inertial']['origin']['@xyz'] = '-0.07 0 -0.12'
        xml_dict['robot']['link'][7]['inertial']['origin']['@xyz'] = '{} 0 {}'.format(
            -0.07 * fth_r * baseLength / 0.266, -.12 * fth_r * baseLength / 0.266)
        # collision frame
        xml_dict['robot']['link'][7]['collision']['origin']['@xyz'] = '-0.07 0 -0.12'
        xml_dict['robot']['link'][7]['collision']['origin']['@xyz'] = '{} 0 {}'.format(
            -0.07 * fth_r * baseLength / 0.266, -.12 * fth_r * baseLength / 0.266)
        # collision shape
        xml_dict['robot']['link'][7]['collision']['geometry']['capsule']["@length"] = '0.266'
        xml_dict['robot']['link'][7]['collision']['geometry']['capsule']["@length"] = '{}'.format(
            0.266 * fth_r * baseLength / 0.266)
        xml_dict['robot']['link'][7]['collision']['geometry']['capsule']["@radius"] = '{}'.format(0.046 * r4)

        xml_dict['robot']['joint'][7]['origin']['@xyz'] = '-0.14 0 -0.24'
        xml_dict['robot']['joint'][7]['origin']['@xyz'] = '{} 0 {}'.format(-0.14 * fth_r * baseLength / 0.266,
                                                                           -0.24 * fth_r * baseLength / 0.266)

        # fore sh link
        # inertial frame
        xml_dict['robot']['link'][8]['inertial']['mass']['@value'] = '{}'.format(1.817 / 2 * fsh_r * r5 ** 2)
        xml_dict['robot']['link'][8]['inertial']['origin']['@xyz'] = '0.065 0 -0.09'
        xml_dict['robot']['link'][8]['inertial']['origin']['@xyz'] = '{} 0 {}'.format(
            0.065 * fsh_r * baseLength / 0.212, -.09 * fsh_r * baseLength / 0.212)
        # collision frame
        xml_dict['robot']['link'][8]['collision']['origin']['@xyz'] = '0.065 0 -0.09'
        xml_dict['robot']['link'][8]['collision']['origin']['@xyz'] = '{} 0 {}'.format(
            0.065 * fsh_r * baseLength / 0.212, -.09 * fsh_r * baseLength / 0.212)
        # collision shape
        xml_dict['robot']['link'][8]['collision']['geometry']['capsule']["@length"] = '0.21200'
        xml_dict['robot']['link'][8]['collision']['geometry']['capsule']["@length"] = '{}'.format(
            0.21200 * fsh_r * baseLength / 0.212)
        xml_dict['robot']['link'][8]['collision']['geometry']['capsule']["@radius"] = '{}'.format(0.046 * r5)

        xml_dict['robot']['joint'][8]['origin']['@xyz'] = '0.13 0 -0.18'
        xml_dict['robot']['joint'][8]['origin']['@xyz'] = '{} 0 {}'.format(0.13 * fsh_r * baseLength / 0.212,
                                                                           -0.18 * fsh_r * baseLength / 0.212)

        # fore foot link
        # inertial frame
        xml_dict['robot']['link'][9]['inertial']['mass']['@value'] = '{}'.format(1.2 / 2 * ffo_r * r6 ** 2)
        xml_dict['robot']['link'][9]['inertial']['origin']['@xyz'] = '0.045 0 -0.07'
        xml_dict['robot']['link'][9]['inertial']['origin']['@xyz'] = '{} 0 {}'.format(0.045 * ffo_r * baseLength / 0.14,
                                                                                      -.07 * ffo_r * baseLength / 0.14)
        # collision frame
        xml_dict['robot']['link'][9]['collision']['origin']['@xyz'] = '0.045 0 -0.07'
        xml_dict['robot']['link'][9]['collision']['origin']['@xyz'] = '{} 0 {}'.format(
            0.045 * ffo_r * baseLength / 0.14, -.07 * ffo_r * baseLength / 0.14)
        # collision shape
        xml_dict['robot']['link'][9]['collision']['geometry']['capsule']["@length"] = '0.1400'
        xml_dict['robot']['link'][9]['collision']['geometry']['capsule']["@length"] = '{}'.format(
            0.1400 * ffo_r * baseLength / 0.14)
        xml_dict['robot']['link'][9]['collision']['geometry']['capsule']["@radius"] = '{}'.format(0.046 * r6)

        xml_string = xmltodict.unparse(xml_dict, pretty=True)
        with open(file, 'w') as fd:
            fd.write(xml_string)

    def check_alive(self):
        self._alive = 1

    def reset_design(self, bullet_client, design):
        self._adapted_xml_file = tempfile.NamedTemporaryFile(delete=False, prefix='halfcheetah_', suffix='.urdf')
        self._adapted_xml_filepath = self._adapted_xml_file.name
        file = self._adapted_xml_filepath
        self._adapted_xml_file.close()
        atexit.register(cleanup_func_for_tmp, self._adapted_xml_filepath)
        self.adapt_xml(file, design)
        self.model_urdf = file

        self.doneLoading = 0
        self.done_loading = 0
        self.parts = None
        self.objects = []
        self.jdict = None
        self.ordered_joints = None
        self.robot_body = None
        self.robot_name = "torso"
        bullet_client.resetSimulation()
        self.reset(bullet_client)

        def actuate(x):
            x = (x - 1) * (8 / 0.5)

            return 1 + (1 / (1 + np.exp(-x)) - 0.5) * 1.8

        fbase = 120
        self.forces = [0, 0, 0]
        self.forces.extend([fbase * i for i in self.forces_r])
        self.spd = StablePDController(self.objects, self._p, self.kps, self.kds, self.forces, self.timeStep)