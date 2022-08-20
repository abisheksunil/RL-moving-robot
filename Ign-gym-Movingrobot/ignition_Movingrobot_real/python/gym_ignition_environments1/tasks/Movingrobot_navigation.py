import abc
from enum import Flag
from turtle import shape
from typing import Tuple, Union

from numpy.core.umath import sqrt
from scenario.bindings.core import World
from scenario import gazebo as scenario_gazebo


# import roslaunch
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu

import time
import shlex

import gym
import numpy as np
from gym_ignition.base import task
from gym_ignition.utils.typing import (
    Action,
    ActionSpace,
    Observation,
    ObservationSpace,
    Reward,
)

from scenario import core as scenario
#from scenario import scenario_core
import subprocess
from scipy.spatial import distance



def stringToList(string):
    listRes = list(string.split(" "))
    return listRes


class MovingrobotNavigation(task.Task, abc.ABC):
    def __init__(self, agent_rate: float, **kwargs):

        task.Task.__init__(self, agent_rate=agent_rate)

        self.model = None

        self.model_name = None

        self.model_name2 = None

        self.done = None

        self.resetSpace = None

        self.goalPos = 4.0

        self.targetXPos = 4
        self.targetYPos = 4
        self.targetZPos = 0

        self.xPosThreshold = 9
        self.yPosThreshold = 9
        self.zPosThreshold = 9

        self.posx = 0
        self.posy = 0
        self.posz = 0

        self.launch = None

        self.verbose = False

        self.subprocess = False

        self.rospub = True

        self.initNode = True

        self.rosSub = False
        self.scenarioPose = True

        self.imumsg = Imu()

        self.maxAngleSpeed = 1

        self.cost = None

        #self.rosLaunchBridge()
        #self.rosLaunchSub()
        

        if self.initNode:

            rospy.init_node('newPub', anonymous=True)


   

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:

    # Configure reset space limit

        observation_space = self.create_observation_space()
        action_space = self.create_action_space()


        return action_space, observation_space



    def create_observation_space(self) -> ObservationSpace:

          # Configure reset limits
        high = np.array(
            [
                self.xPosThreshold,    # x coordinate max
                self.yPosThreshold,    # y coordinate max
                self.zPosThreshold     # z coordinate max
            ]
        )

          # Configure reset space limit
        self.resetSpace = gym.spaces.Box(
            low=-high, high=high, dtype=np.float32)


        # Configure observation space
        obsHigh = high.copy() * 1.2
        return gym.spaces.Box(
            low=-obsHigh, high=obsHigh, dtype=np.float32)
        


    def create_action_space(self) -> ActionSpace:

        return gym.spaces.Box(
            low=-self.maxAngleSpeed, high=self.maxAngleSpeed, shape=(2,), dtype=np.float32
        )


    def set_action(self, action: Action) -> None:

        linear = action.tolist()[0]
        angular = action.tolist()[1]

        # Denormalize action space

        lineard =  (linear/2+0.5) * (2-(-2)) + (-2)
        angulard =  (angular/2+0.5) * (2-(-2)) + (-2)


        if self.rospub:
            
            pub = rospy.Publisher('/model/Movingrobot/cmd_vel', Twist, queue_size=10)

            if not rospy.is_shutdown():
                twist = Twist()
                twist.linear.x = lineard; twist.linear.y = 0.0; twist.linear.z = 0.0
                twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = angulard
                pub.publish(twist)
                
                            


        if self.subprocess:
            actionTopic = f'ign topic -t "/model/Movingrobot/cmd_vel" -m ignition.msgs.Twist -p "linear: {{x: {linear}}}, angular: {{z: {angular}}}"'
            subprocess.run(actionTopic, shell=True)
    

    def getMsg(self,msg):
            self.imumsg = msg
            
             
    

    def get_observation(self) -> Observation:

 
        if self.rosSub:
            rospy.Subscriber('/imu', Imu, self.getMsg)

            self.x = self.imumsg.orientation.x
            self.y = self.imumsg.orientation.y
            self.z = 0
            observation = Observation(np.array([self.x, self.y, self.z], dtype=np.float32))

            if self.verbose:
                print(f"Pose : {[self.x, self.y, self.z]}")

        
      
        if self.scenarioPose:
            model = self.world.get_model(self.model_name)

            currentPos = model.base_position()

            if self.verbose:

                print(f"Current x, y, z : {currentPos[0], currentPos[1], currentPos[2] }")

            truex = currentPos[0]
            truey = currentPos[1]
            self.z = 0


            # Apply Gaussian Noise 

            self.x, self.y = self.add_gnoise(truex, truey)

            # Publish True and noisy x and y for plotting

            pub2 = rospy.Publisher('/verifynoise', Float64MultiArray, queue_size=10)

            if not rospy.is_shutdown():
                array = Float64MultiArray()
                array.data = [truex, truey, self.x, self.y]
                pub2.publish(array)


            # Create the observation
            observation = Observation(np.array([self.x, self.y, self.z], dtype=np.float32))

        # Return the observation
        return observation


    def add_gnoise(self,odomx, odomy):
        mu = 0.0
        std = 0.00025
        x_noise = np.random.normal(mu, std)
        x_noisy = x_noise + odomx
        y_noise = np.random.normal(mu, std)
        y_noisy = y_noise + odomy
        return x_noisy, y_noisy


    def get_reward(self) -> Reward:

        # Calculate the reward
        # reward = 1.0 if not self.is_done() else 0.0

        if self.verbose:
            print(f"curren Pos :  {self.x, self.y, self.z}")

        currentPos = (self.x, self.y, self.z)
        targetPos = (self.targetXPos, self.targetYPos, self.targetZPos)

        dist = (distance.euclidean(currentPos, targetPos))


        high = np.array(
            [
                self.goalPos,   # 
            ]
        )

          # Configure reset space limit

        spacex = gym.spaces.Box(
            low=high, high=high + 0.5, dtype=np.float32)
        spacey = gym.spaces.Box(
            low=high, high=high + 0.5, dtype=np.float32)    
        spacez = gym.spaces.Box(
            low=0, high=0.5, shape = (1,), dtype=np.float32)

        currentnx = np.array([currentPos[0]],  dtype= np.float32)
        currentny = np.array([currentPos[1]], dtype= np.float32)
        currentnz = np.array([currentPos[2]], dtype= np.float32)


        # self.goalSpace = gym.spaces.Box(
        #     low=high - 4, high=high + 2, dtype=np.float32)

       
        if spacex.contains(currentnx) and spacey.contains(currentny) and spacez.contains(currentnz):
            print("yesssssss")
            self.cost = -0.1
            self.done = True

        elif not self.resetSpace.contains(Observation(np.array([currentPos[0], currentPos[1], currentPos[2]], dtype=np.float32))):  
            self.cost = -0.9 * dist      

        else:
            self.cost = -0.5 * dist

            


        reward = self.cost
        # print(f"reward ; {reward}")

        return reward

    def is_done(self) -> bool:

        # Get the observation
        observation = self.get_observation()

        done =  not self.resetSpace.contains(observation)

        if self.done:
            done = self.done

        self.done = None
   
        return done

    def reset_task(self) -> None:



        if self.model_name not in self.world.model_names():
            raise RuntimeError(
                "The Movingrobot model was not inserted in the world")

        model = self.world.get_model(
            self.model_name)

        # time.sleep(10)

        xPoint = np.random.randint(-7, -3)
        yPoint = np.random.randint(-7, -3)

        if self.verbose:
            print(f"xPoint: {xPoint}")

        ok_reset = model.to_gazebo().reset_base_position([xPoint, yPoint, 1.3])
        

        if not ok_reset:
            raise RuntimeError("Failed to reset the Movingrobot state")

  


