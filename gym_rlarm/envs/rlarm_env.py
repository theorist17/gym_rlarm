import os
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as pb
import pybullet_data

class DummyNet():
    def predict(self):
        action = 1
        state = 0
        return action, state

class RlarmEnv(gym.Env):
    metadata = {'render.modes': ['GUI', 'TUI']}
    # refresh rate
    dt = .1
    # state is comprised of 9 elements
    stateDim = 9
    # we have two joints which we'll put forces on
    actionDim = 2

    def __init__(self, mode='GUI'):
        pb.connect(pb.GUI) # if mode == 'GUI' else pb.connect(pb.DIRECT)

        # world
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane = pb.loadURDF("plane.urdf")
        pb.setTimeStep(1) # default 240hz
        pb.setGravity(0, 0, -9.81)  # everything should fall down
        pb.setTimeStep(0.00001)  # this slows everything down, but let's be accurate...
        pb.setRealTimeSimulation(0)  # we want to be faster than real time :)

        # arm
        dirname = os.path.dirname(__file__)
        armUrdfPath = os.path.join(dirname, '../../assets/kuka_kr210_support/urdf/kr210l150.urdf')
        self.arm = pb.loadURDF(armUrdfPath, basePosition=[0, 0, 0], useFixedBase=1)
        self.armPos, self.armOri = pb.getBasePositionAndOrientation(self.arm)
        self.armNumJoint = pb.getNumJoints(self.arm)

        self.jointInfo = list()
        keys = ["index", "name", "type", "qIndex", "uIndex", "flags",
                    "damping", "friction", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                    "linkName", "axis", "parentFramePos", "parentFrameOrn", "parentIndex"]
        for joint in range(self.armNumJoint):
            values = pb.getJointInfo(self.arm, joint)
            self.jointInfo.append(dict(zip(keys, values)))
        for info in self.jointInfo:
            print(info)

        self.jointState = list()
        keys = ["position", "velocity", "reactionForces", "torque"]
        for joint in range(self.armNumJoint):
            values = pb.getJointState(self.arm, joint)
            self.jointState.append(dict(zip(keys, values)))
        for state in self.jointState:
            print(state)

        #  button
        buttonUrdfPath = os.path.join(dirname, '../../assets/button/urdf/simple_button.urdf')
        self.button = pb.loadURDF(buttonUrdfPath, basePosition=[2, 0, 0], useFixedBase=1)
        self.buttonPos, self.buttonOri = pb.getBasePositionAndOrientation(self.button)

        self.done = False
        self.goal = {'x': float(self.buttonPos[0]), 'y': float(self.buttonPos[1]), 'size': 10}
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=
                    (255, 255, 3), dtype=np.uint8)
        self.reward_range = (0, 100)

    def step(self, action):
        done = False
        pb.setJointMotorControl2(
            bodyIndex=self.arm, jointIndex=0, controlMode=pb.POSITION_CONTROL,
            targetPosition=0.5)

        # pb.setJointMotorControlArray(
        #     bodyUniqueId=self.robot, jointIndices=range(1), controlMode=pb.POSITION_CONTROL,
        #     targetPositions=[-14] * 1)
        reward = 0

        return reward, done

    def reset(self):
        pb.resetSimulation()

    def render(self):
        pb.stepSimulation()

    def close(self):
        pb.disconnect()

if __name__ == '__main__':
    env = RlarmEnv()
    model = DummyNet()
    for _ in range(1000000):
        action, state = model.predict()
        env.step(action)
        env.render()
    env.close()
