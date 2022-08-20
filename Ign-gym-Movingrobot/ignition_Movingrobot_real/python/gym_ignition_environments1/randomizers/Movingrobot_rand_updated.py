# Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import abc
from pickle import FALSE
from typing import Union, Tuple

import gym_ignition_models  # newwwwwwwww
import time  # newwwwwwwwwww

from gym_ignition import randomizers

from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from gym_ignition_environments1 import tasks, models
from gym_ignition_environments1.models.Ground import Ground
from gym_ignition_environments1.models import moving__robot,box
from scenario import core as scenario, gazebo
from scenario import gazebo as scenario_gazebo
from scenario.bindings.core import World

import numpy as np


# Tasks that are supported by this randomizer. Used for type hinting.
SupportedTasks = tasks.Movingrobot_navigation.MovingrobotNavigation


class MovingrobotEnvPhysicsRandUpdated(
    gazebo_env_randomizer.GazeboEnvRandomizer,
    randomizers.abc.TaskRandomizer,
    randomizers.abc.PhysicsRandomizer,
    abc.ABC,
):

    def __init__(
        self,
        env: MakeEnvCallable, 
        num_physics_rollouts: int = 0, 
        verbo : bool = False, 
        taskverbo : bool = False, 
        object_type: str = "box"
        ):

        # Initialize base classes
        randomizers.abc.TaskRandomizer.__init__(self)
        randomizers.abc.PhysicsRandomizer.__init__(
            self, randomize_after_rollouts_num=num_physics_rollouts
        )
        gazebo_env_randomizer.GazeboEnvRandomizer.__init__(
            self, env=env, physics_randomizer=self
        )
        
        self.verbosity = verbo
        self.timesphy = num_physics_rollouts
        self.gravity_z = None
        self.set_grav = None
        self.g = None
        self.set_engine = None
        self.__env_initialised = False

        self.taskverbo = taskverbo

        if self.verbosity:
            scenario_gazebo.set_verbosity(scenario_gazebo.Verbosity_debug)

        # For the goal object 

        self.__object_model_class = models.get_object_model_class(object_type) 

        

        # Initialize the environment randomizer

        #super().__init__(env=env)

        # ===========================
        # PhysicsRandomizer interface
        # ===========================

    def get_engine(self):

        return scenario_gazebo.PhysicsEngine_dart

    def randomize_physics(self, task: SupportedTasks, **kwargs) -> None:

        self.gravity_z = task.np_random.normal(loc=-9.8, scale=0.2)

        #self.set_engine = task.world.to_gazebo().PhysicsEngine_dart = 1

        if not task.world.to_gazebo().set_gravity((0, 0, self.gravity_z)):
            raise RuntimeError("Failed to set the gravity")

        # print(task.world.gravity())

    def physics_expired(self) -> bool:
        return super().physics_expired()

    # ========================
    # TaskRandomizer interface
    # ========================

    def randomize_task(self, task: SupportedTasks, **kwargs) -> None:

        
        task.verbose = self.taskverbo
        # Remove the model from the world
        self._clean_world(task=task)

        if "gazebo" not in kwargs:
            raise ValueError("gazebo kwarg not passed to the task randomizer")

        gazebo = kwargs["gazebo"]

        if not self.__env_initialised:

            print(gazebo.initialized())


            self.__env_initialised = True

        # Select the physics engine
        # self.world.set_physics_engine(scenario_gazebo.PhysicsEngine_dart)

        # Execute a paused run to process model removal
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

        # Insert a new Movingrobot model
        model = moving__robot.Movingrobot(world=task.world)


        # Insert goal object
        
        model2 = box.Box(task=task, gazebo=gazebo, world=task.world)


        # self._populate_world(task=task)

        task.world.set_physics_engine(scenario_gazebo.PhysicsEngine_dart)


        

        # Enable Ground

        # if task._ground_enable:
        #     print("Inserting default ground plane")
        #     self.add_default_ground(task=task,
        #                             gazebo=gazebo)

        if self.timesphy == 0:
            raise RuntimeError("0 ROLLOUTS")
        # print(f"Rollout : {self.timesphy, self.gravity_z}")

        # print(f"Physics expired  : {self.physics_expired()}")


       
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a running Gazebo run")

        # Store the model name in the task
        task.model_name = model.name()

        task.model_name2 = model2.name()

      
        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    @staticmethod
    def _clean_world(task: SupportedTasks) -> None:

        # Remove the model from the simulation
        if task.model_name is not None and task.model_name in task.world.model_names():

            if not task.world.to_gazebo().remove_model(task.model_name):
                raise RuntimeError(
                    "Failed to remove the Movingrobot from the world")

        if task.model_name2 is not None and task.model_name2 in task.world.model_names():

            if not task.world.to_gazebo().remove_model(task.model_name2):
                raise RuntimeError(
                    "Failed to remove the Movingrobot from the world")

    @staticmethod
    def _populate_world(task: SupportedTasks, Movingrobot_model: str = None) -> None:

        # Insert a new cartpole.
        # It will create a unique name if there are clashing.
        model = moving__robot.Movingrobot(world=task.world, model_file=Movingrobot_model)

        # Store the model name in the task
        task.model_name = model.name()

