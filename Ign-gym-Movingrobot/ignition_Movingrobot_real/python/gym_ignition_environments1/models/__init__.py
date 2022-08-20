# Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

from . import moving__robot, Ground, box
from gym_ignition.scenario.model_wrapper import ModelWrapper

def get_object_model_class(object_type: str) -> ModelWrapper:

    if "box" == object_type:
        return box
        
