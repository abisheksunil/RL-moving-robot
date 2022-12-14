# Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import functools
import time
import numpy as np
import argparse
from distutils.util import strtobool

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

import gym
from gym_ignition.utils import logger
from gym_ignition_environments1 import randomizers

import rospy 
import roslaunch


# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)

# Available tasks
env_id = "Movingrobot-v0"




def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    import gym_ignition_environments

    return gym.make(env_id, **kwargs)


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # Create a partial function passing the environment id
        m_env = functools.partial(make_env_from_id, env_id=env_id)
    
         # This is a complex example that randomizes both the physics and the model.
        env = randomizers.Movingrobot_rand_updated.MovingrobotEnvPhysicsRandUpdated(
            env=m_env, num_physics_rollouts=2, taskverbo= False)

        # Check the environment
        check_env(env)

        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


# Arg parser 

parser = argparse.ArgumentParser(description='Toggle args')
parser.add_argument('--rl', '-r', type=bool, default=False)
args = parser.parse_args()


####!!!!Roslaunch defenition !!!!!!!!!

def rosLaunchSub():
    rospy.init_node('Movingrobot_odoms', anonymous=True)
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(
        uuid, ["/home/abi/Documents/Ign-gym-Movingrobot/ignition_Movingrobot_real/python/gym_ignition_environments1/src/Movingrobot_odoms/launch/ign_topic.launch"])
    launch.start()
    rospy.loginfo("started")


if __name__ == '__main__':

    # Enable vecEnv 
        
    num_cpu = 4
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])


    # Start the training
    if args.rl == True:
        # env.render()
        # model = PPO("MlpPolicy", env, verbose=1,  tensorboard_log="./har_log")
        # print("done setting the model")
        # # Set new logger
        # # model.set_logger(new_logger)
        # model.learn(total_timesteps=5000)
        # model.save("saves/n3/ppo_Movingrobotr1")
        # model.learn(total_timesteps=5000, reset_num_timesteps=False, tb_log_name="PPO-d")
        # model.save("saves/n3/ppo_Movingrobotr2")
        # model.learn(total_timesteps=5000,  reset_num_timesteps=False, tb_log_name="PPO-d")
        # model.save("saves/n2/ppo_Movingrobotr3")
        # model.learn(total_timesteps=5000, reset_num_timesteps=False)
        # model.save("saves/n2/ppo_Movingrobotr4")
        # model.learn(total_timesteps=5000, reset_num_timesteps=False)
        # model.save("saves/n2/ppo_Movingrobotr5")
        # model.learn(total_timesteps=5000, reset_num_timesteps=False)
        # model.save("saves/n2/ppo_Movingrobotr6")

        # del model # remove to demonstrate saving and loading


        
        
        model = PPO.load("saves/n3/ppo_Movingrobotr2")
        dones = False
        reward = 0
        counter = 0
        # rosLaunchSub()

    # Start the trained model 
        obs = env.reset()
        for _ in range(500):
            # action= env.action_space.sample()
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            print(obs, dones, counter)
            env.render()
            reward += rewards
            counter += 1
            if dones:
                env.reset()
                print(f"reward : {reward}")
                reward = 0
                
        
    
        print(f"rewards : {reward}")


        env.close()
        time.sleep(5)
