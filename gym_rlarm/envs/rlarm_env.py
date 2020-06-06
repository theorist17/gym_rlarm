import os
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as pb
import pybullet_data
import threading
from gym.spaces import Box, Discrete
import math
import time
JOINT_INFO_KEY = ["index", "name", "type", "qIndex", "uIndex", "flags",
                           "damping", "friction", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                           "linkName", "axis", "parentFramePos", "parentFrameOrn", "parentIndex"]
LINK_STATE_KEY = ["linkWorldPosition", "linkWorldOrientation",
                  "localInertialFramePosition", "localInertialFrameOrientation",
                  "worldLinkFramePosition", "worldLinkFrameOrientation",
                  "worldLinkLinearVelocity", "worldLinkAngularVelocity"]
JOINT_STATE_KEY = ["position", "velocity", "reactionForces", "torque"]

JOINT_NAME = ["joint1", "joint2", "joint3", "joint4", "joint5"]
LINK_NAME = ["link1", "link2", "link3", "link4", "link5"]

JOINT2ID = dict(zip(JOINT_NAME, range(len(JOINT_NAME))))
LINK2ID = dict(zip(LINK_NAME, range(len(LINK_NAME))))
NAME2ID = {**JOINT2ID, **LINK2ID}

def clip(x):
    if isinstance(x, list):
        return [round(x, 4) for x in x]
    elif isinstance(x, tuple):
        return list(map(lambda i: round(i, 4), x))
    elif isinstance(x, float):
        return round(x, 4)
    elif type(x) is np.ndarray:
        return np.round(x, 4)
    else:
        raise Exception("it is not a list, a tuple, or a float (%s)" % type(x))

def degree(x):
    if isinstance(x, list):
        return [math.degrees(x) for x in x]
    elif isinstance(x, tuple):
        return list(map(lambda i: math.degrees(i), x))
    elif isinstance(x, float):
        return math.degrees(x)
    elif type(x) is np.ndarray:
        return np.degrees(x)
    else:
        raise Exception("it is not a list, a tuple, or a float (%s)" % type(x))

class DummyNet():
    def predict(self, state=[0] * 5):
        self.action = np.array([-0.8,-1.3,0.5,0,-1])
        return self.action

class RlarmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mode='GUI'):
        super(RlarmEnv, self).__init__()
        # pybullet
        pb.connect(pb.GUI)
        pb.resetDebugVisualizerCamera(cameraDistance=0.25, cameraYaw=90, cameraPitch=-40,
                                      cameraTargetPosition=[0.0, 0, 0])
        # path
        dir_name = os.path.dirname(__file__)
        self.button_urdf_path = os.path.join(dir_name, '../../assets/button/urdf/simple_button.urdf')
        self.arm_urdf_path = os.path.join(dir_name, '../../assets/Assem1/urdf/Assem1.xml')
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())

        # plane
        self.plane = pb.loadURDF("plane.urdf")

        # arm
        self.arm_base_pos = [0, 0, 0]
        self.arm_base_ori = pb.getQuaternionFromEuler(list(map(math.radians, [-90, 0, 90])))
        self.arm = pb.loadURDF(self.arm_urdf_path, basePosition=self.arm_base_pos, baseOrientation=self.arm_base_ori, useFixedBase=1)
        self.arm_jnum = pb.getNumJoints(self.arm)

        #  button
        self.button_base_pos = [0.1, 0, 0]
        self.button_base_ori = pb.getQuaternionFromEuler(list(map(math.radians, [-90, 0, 0])))
        self.button = pb.loadURDF(self.button_urdf_path, basePosition=[0.1, 0, 0], useFixedBase=1)

        self.print_frame()

        # gym
        minPosition = np.array(self.joint_info_values("lowerLimit")) # 5
        maxPosition = np.array(self.joint_info_values("upperLimit"))
        minVelocity = np.array([0] * self.arm_jnum) # 5
        maxVelocity = np.array(self.joint_info_values("maxVelocity"))
        minGoal = np.array([-1] * 3) # 3
        maxGoal = np.array([1] * 3)
        minEndEffector = np.array([-1] * 3) # 3
        maxEndEffector = np.array([1] * 3)
        lowerLimit = np.concatenate((minPosition, minVelocity, minGoal, minEndEffector))
        upperLimit = np.concatenate((maxPosition, maxVelocity, maxGoal, maxEndEffector))
        self.observation_space = Box(lowerLimit, upperLimit)
        self.action_space = Box(minPosition, maxPosition)

        self.threshold = 0.001
        self.penalty = 0.001
        self.time_step = 1./ 240
        self.max_step = 300
        self.step_id = 0
        self.start_time = None

    def step(self, action):
        self.step_id += 1
        pb.setJointMotorControlArray(self.arm, jointIndices=range(self.arm_jnum),
                                     controlMode=pb.POSITION_CONTROL, targetPositions=action)
        pb.stepSimulation()

        positions, velocities = self.getPositionAndVelocity()
        end_effector = self.getEndEffector()

        observation = np.concatenate((positions, velocities, self.goal, end_effector))
        reward = self.getReward(self.goal, end_effector, action, self.penalty)
        done = self.getDone(self.goal, end_effector, self.threshold)
        info = None

        return observation, reward, done, info

    def reset(self):
        pb.resetSimulation()
        pb.setGravity(0, 0, -9.87)
        self.plane = pb.loadURDF("plane.urdf")
        self.arm = pb.loadURDF(self.arm_urdf_path, basePosition=self.arm_base_pos, baseOrientation=self.arm_base_ori, useFixedBase=1)
        self.button = pb.loadURDF(self.button_urdf_path, basePosition=[0.1, 0, 0], useFixedBase=1)

        self.step_id = 0
        if self.start_time:
            print("Time took",  time.time() - self.start_time, "sec")
        self.start_time = time.time()

        position = np.array([-0.7, 1.1, -0.1, -0.2, 0]) + np.random.uniform(-0.02, 0.02, (self.arm_jnum)) # TODO in paper -0.3, -0.7, 1.1, -0.1, -0.2, 0
        velocity = np.array([0]*self.arm_jnum) + np.random.uniform(-0.1, 0.1, (self.arm_jnum))

        self.goal = np.array([0.07, 0, 0.05]) + np.random.uniform(-0.05, 0.05, (3))
        self.goal[-1] = max(self.goal[-1], 0)
        end_effector = self.getEndEffector()

        observation = np.concatenate((position, velocity, self.goal, end_effector))

        pb.resetBasePositionAndOrientation(self.button, self.goal, self.button_base_ori)

        return observation

    def render(self):
        pass

    def close(self):
        pb.disconnect()

    def getPositionAndVelocity(self):
        '''
        :return: return numpy array of (5) representing position of joints
                 and numpy array of (5) representing velocity of joints
        '''
        joint_states = pb.getJointStates(self.arm, jointIndices=range(self.arm_jnum))
        positions = [joint[0] for joint in joint_states]
        positions = [round(position, 4) for position in positions]
        positions = np.array(positions)

        velocities = [joint[1] for joint in joint_states]
        velocities = [round(velocity) for velocity in velocities]
        velocities = np.array(velocities)
        return positions, velocities

    def getEndEffector(self):
        '''
        :return: return numpy array of 1 X 3 representing position of end effector (tip of hand) in x, y, z
        '''
        end_effector = pb.getLinkState(self.arm, self.arm_jnum-1)
        world_position = np.array(end_effector[0])
        return world_position

    def getReward(self, goal, end_effector, action, penalty): # http://www.dei.unipd.it/~toselloe/pdf/IR0S18_Franceschetti.pdf
        '''
        :param goal:
        :param end_effector:
        :param acceleration:
        :param penalty:
        :return: return a scalar value of reward evaluating a state with an aciton
        '''
        difference = goal - end_effector
        distance = np.linalg.norm(difference)
        regularization = penalty * np.linalg.norm(action)
        reward = - distance - regularization

        return reward

    def getDone(self, goal, end_effector, threshold):
        '''
        :param goal:
        :param end_effector:
        :param threshold:
        :param step:
        :param max_step:
        :return: return boolean done determining whether to proceed or abandon
        '''

        done = False

        difference = goal - end_effector
        distance = np.linalg.norm(difference)

        if self.step_id >= self.max_step:
            done = True
            print('Failed')

        if distance < threshold:
            done = True
            print("Reached goal", goal, "at", end_effector, "distance", distance)

        return done

    def joint_info(self, joint_id):
        joint_info_val = pb.getJointInfo(self.arm, joint_id)
        return dict(zip(JOINT_INFO_KEY, joint_info_val))

    def joint_infos(self):
        joint_infos = list()
        for i in range(pb.getNumJoints(self.arm)):
            joint_infos.append(self.joint_info(i))
        return joint_infos

    def joint_state(self, joint_id):
        joint_state_val = pb.getJointState(self.arm, joint_id)
        return {'index': joint_id, 'name': JOINT_NAME[joint_id], **dict(zip(JOINT_STATE_KEY, joint_state_val))}

    def joint_states(self):
        joint_states = list()
        for i in range(pb.getNumJoints(self.arm)):
            joint_states.append(self.joint_state(i))
        return joint_states

    def link_state(self, link_id):
        link_state_val = pb.getLinkState(self.arm, link_id)
        link_state = dict(zip(LINK_STATE_KEY, link_state_val))
        return {'index': link_id, 'name': LINK_NAME[link_id], **link_state}

    def link_states(self):
        link_states = list()
        for i in range(pb.getNumJoints(self.arm)):
            link_states.append(self.link_state(i))
        return link_states

    def joint_info_values(self, key):
        values = [info[key] for info in self.joint_infos()]
        return values

    def joint_state_values(self, key):
        values = [state[key] for state in self.joint_states()]
        return values

    def link_state_values(self, key):
        values = [state[key] for state in self.link_states()]
        return values

    def print_frame(self):
        print('\nJoint Infos')
        for info in self.joint_infos():
            print(info)

        print('\nJoint States')
        for state in self.joint_states():
            print(state)

        print('\nLink States')
        for link in self.link_states():
            print(link)

if __name__ == '__main__':
    env = RlarmEnv()
    model = DummyNet()

    max_time_step = 300

    while True:
        score = 0
        state = env.reset()

        for step in range(max_time_step):
            action = model.predict(state)
            nextState, reward, done, info = env.step(action)
            env.render()

            state = nextState
            score += reward

            if done:
                print('Score is', score)
                break

    env.close()
