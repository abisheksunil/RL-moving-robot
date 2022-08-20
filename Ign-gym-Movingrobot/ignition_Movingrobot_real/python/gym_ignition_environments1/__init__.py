# Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import numpy

from gym.envs.registration import register

from . import models, randomizers, tasks

max_float = float(numpy.finfo(numpy.float32).max)


register(
    id="Movingrobot-v0",
    entry_point=f"gym_ignition.runtimes.gazebo_runtime:GazeboRuntime",
    max_episode_steps=50,
    nondeterministic=True,
    kwargs={
        "task_cls": tasks.Movingrobot_navigation.MovingrobotNavigation,
        "agent_rate": 1000,
        "physics_rate": 1000,
        "real_time_factor": max_float ,
    },
)
