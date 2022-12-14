# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

# Workaround for https://github.com/osrf/sdformat/issues/227.
# It has to be done before loading the bindings.
import gym_ignition_models

gym_ignition_models.setup_environment()

# Add IGN_GAZEBO_RESOURCE_PATH to the default search path
import os

from gym_ignition.utils import resource_finder

if "IGN_GAZEBO_RESOURCE_PATH" in os.environ:
    resource_finder.add_path_from_env_var("IGN_GAZEBO_RESOURCE_PATH")


def initialize_verbosity() -> None:

    import gym
    import gym_ignition.utils.logger

    import scenario

    if scenario.detect_install_mode() is scenario.InstallMode.Developer:
        gym_ignition.utils.logger.set_level(
            level=gym.logger.INFO, scenario_level=gym.logger.WARN
        )

    elif scenario.detect_install_mode() is scenario.InstallMode.User:
        gym_ignition.utils.logger.set_level(
            level=gym.logger.WARN, scenario_level=gym.logger.WARN
        )

    else:
        raise ValueError(scenario.detect_install_mode())


# Configure default verbosity
initialize_verbosity()
