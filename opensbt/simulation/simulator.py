
from dataclasses import dataclass
from typing import Dict, List, Any
from pymoo.core.individual import Individual
from abc import ABC, abstractstaticmethod
from opensbt.utils.encoder_utils import NumpyEncoder
from llm.model.models import Utterance

import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


@dataclass
class SimulationOutputBase(ABC):
    def __init__(self, **kwargs):
        for key,value in kwargs.items():
            setattr(self, key, value)

    def to_json(self):
        return json.dumps(self.__dict__,
                        allow_nan=True, 
                        indent=4,
                        cls=NumpyEncoder)

    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)
        return cls(**json_dict)

@dataclass
class DrivingSimulationOutput(SimulationOutputBase):
    
    """
        Class represents data output after execution of a simulation. An example JSON representation of a SimulationOutput instance is:

    {
        "simTime": 10,  // Simulation time
        "times": [0.0, 0.1, 0.2, ... , 10.0 ], // Time steps; delta is not predefined
        "location": {
            "ego": [
                [0.0, 0.0], [0.1, 0.1],  ...  // x,y positions for each time step
            ],
            "adversary": [
                [10.0, 0.0], [9.9, 0.1],  ... 
            ]
        },
        "velocity": {
            "ego": [  
                [1,1,0], [1,1,0], ...   // velocity vector for each time step
            "adversary": [
                [0,1,0], [0, 1, 1], ...
            ]
        },
        "speed": {
            "ego": [1.4, 1.4, ... ], // magnitude of velocity vector (known as "speed") for each time step
            "adversary": [1, 1, ... ]
        },
        "acceleration": {
            "ego": [0.1, 0, ...],
            "adversary": [0.05, 0, ...]
        },
        "yaw": {                     // heading in rad for each time step
            "ego": [0.5, 0.5, ...],
            "adversary": [0.2, 0.2, ...]
        },
        "collisions": [              // actor ids with timesteps if any collisions
        ],
        "actors": {                  // type of actors mapped to ids; the actor types "ego" and "adversary" have to be assigned
            "ego": "ego",
            "adversary": "adversary",
            "vehicles": [],
            "pedestrians": []
        },
        "otherParams": {              // store custom data
            "car_width" : 3,
            "car_length" : 5
        }
    } 
    """
    simTime: float
    times: List
    location: Dict
    velocity: Dict
    speed: Dict
    acceleration: Dict
    yaw: Dict
    collisions: List
    actors: Dict
    otherParams: Dict


class Simulator(ABC):
    """ Base class to be inherited and implemented by a concrete simulator in OpenSBT """

    @abstractstaticmethod
    def simulate(list_individuals: List[Individual], 
                variable_names: List[str], 
                scenario_path: str, 
                sim_time: float = 10, 
                time_step: float = 0.01, 
                do_visualize: bool = True) -> List[SimulationOutputBase]:
        """
         Instantiates a list of scenarios using the scenario_path, the variable_names and variable values in list_individuals, and
         simulates scenarios in defined simulator.
         
        :param list_individuals: List of individuals. On individual corresponds to one scenario.
        :type list_individuals: List[Individual]
        :param variable_names: The scenario variables.
        :type variable_names: List[str]
        :param scenario_path: The path to the abstract/logical scenario.
        :type scenario_path: str
        :param sim_time: The simulation time, defaults to 10
        :type sim_time: float, optional
        :param time_step: The time step, defaults to 0.01
        :type time_step: float, optional
        :param do_visualize: Visualize or not simulation, defaults to True
        :type do_visualize: bool, optional
        :return: Returns a list of Simulation output instances.
        :rtype: List[SimulationOutput]
        """
        pass