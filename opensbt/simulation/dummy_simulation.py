from opensbt.model_ga.individual import Individual
from typing import List
from opensbt.simulation.simulator import Simulator, DrivingSimulationOutput
from math import sin, cos, pi, ceil
import numpy as np
from random import random
from opensbt.utils import geometric
import json
from datetime import datetime
import csv
import os

class DummySimulator(Simulator):
    """ 
        This class implements a simplified driving simulator for two actors, where the ego and adversary move with linear velocity.
        The ego scans for nearby vehicles. If the distance is below some threshold the ego reduces its speed until it is 0.
        The ego continues driving when no adversary is visible within the threshold distance.
        The position of ego deviates based on some random noise.
    """
    
    time_step = 1
    DETECTION_THRESH = 2     # threshold in meters where other actors can be detected
    RANDOMNESS_BIAS = 0.1    # noise to be added to positions
    ## Simulates a set of scenarios and returns the output
    archive = {}
    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    @staticmethod
    def ind_in_archive(ind):
        """ Checks if individual is in the archive of individuals."""
        if tuple(ind) in DummySimulator.archive:
            return True
        else:
            return False
        
    @staticmethod
    def simulate(list_individuals: List[Individual], 
                 variable_names: List[str], 
                 scenario_path: str, 
                 sim_time: float, 
                 time_step: float = time_step,
                 do_visualize: bool = False
        ) -> List[DrivingSimulationOutput]:
        """
        Simulates a set of scenarios based on a list of individuals and returns the simulation outputs.

        Parameters
        ----------
        list_individuals : List[Individual]
            A list of individuals representing different scenarios to be simulated.
        variable_names : List[str]
            A list of variable names used in the simulation.
        scenario_path : str
            The file path to the scenario configuration.
        sim_time : float
            Total simulation time.
        time_step : float, optional
            Time step for the simulation, by default DummySimulator.time_step.
        do_visualize : bool, optional
            Whether to visualize the simulation, by default False.

        Returns
        -------
        List[DrivingSimulationOutput]
            A list of simulation outputs for each individual in the input list.
        """
        results = []

        for ind in list_individuals:      
            ################## duplicate simulation avoidance    
            if not DummySimulator.ind_in_archive(ind): 
                # with open(os.getcwd() + os.sep + f'simulated_inds_{DummySimulator.now}.csv', mode = 'a+') as f:
                #     write_to = csv.writer(f)
                #     write_to.writerow([ind])

                # f.close()
                simout = DummySimulator.simulate_single(ind, 
                                                        variable_names, 
                                                        filepath=scenario_path, 
                                                        sim_time=sim_time,
                                                        time_step=time_step)
            else:
                simout = DummySimulator.archive[tuple(ind)]
            results.append(simout)                
        return results

    @staticmethod
    def simulate_single(vars: List[Individual], 
                        variable_names: List[str], 
                        filepath: str, 
                        sim_time: float, 
                        time_step: float, 
                        detection_dist = DETECTION_THRESH, 
                        randomness_bias = RANDOMNESS_BIAS
    ) -> DrivingSimulationOutput:
        """Executes single scenario in DummySimulator
        """
        egoInitialVelocity = vars[1]
        pedInitialVelocity = vars[3]
        egoOrientation = vars[0]
        pedOrientation = vars[2]
        start_pos_ped = [2, 10] # should be also defined by input parameter
        start_pos_ego = [0, 0] # should be also defined by input parameter

        # first plan the motion of ego, get for each time step predicted position/velocity

        plan_ego_traj= plan_motion(start_pos_ego, egoOrientation, egoInitialVelocity, sim_time, time_step)
        plan_adv_traj = plan_motion(start_pos_ped, pedOrientation, pedInitialVelocity, sim_time, time_step)
        
        n_steps = plan_ego_traj.shape[1]
        
        # iterate through each time step checking other actors nearby to avoid collision
        real_ego_traj = np.asarray([
                                    [plan_ego_traj[0,0]],
                                    [plan_ego_traj[1,0]],
                                    [plan_ego_traj[2,0]],
                                    [plan_ego_traj[3,0]],
                                    [plan_ego_traj[4,0]]
        ])

        # first position is real
        k = 1
        for i in range(1,n_steps):
            current_size = real_ego_traj[1,:]
            if len(current_size) < n_steps:
                pos_ego = [real_ego_traj[1,-1],real_ego_traj[2,-1]]
                pos_others = [[plan_adv_traj[1,i],plan_adv_traj[2,i]]]
                if are_actors_nearby(pos_ego, pos_others, detection_dist=detection_dist):
                    # plan make a step, ego vehicle trajectory in the form (Time, X, Y, V, Yaw)
                    # stop for some time steps
                    # t_stop = 3
                    # for t in range(0, n_steps - current_size - t_stop)):
                    new_t = real_ego_traj[0, i - 1] 
                    new_x =  real_ego_traj[1, i - 1] 
                    new_y =  real_ego_traj[2, i - 1] 
                    new_v = 0
                    new_yaw = real_ego_traj[4, i - 1] 
                    new_state =  np.asarray([[new_t],[new_x],[new_y], [new_v], [new_yaw]])
                    real_ego_traj = np.append(real_ego_traj, new_state, axis = 1)
                    # waiting_steps += t_stop - 1
                else:     
                    new_t =  plan_ego_traj[0, k] 
                    new_x =  plan_ego_traj[1, k] 
                    new_y =  plan_ego_traj[2, k] 
                    new_v = plan_ego_traj[3, k]
                    if real_ego_traj[3, k-1] == 0:
                        new_v = plan_ego_traj[3,k]/2
   
                    new_yaw = plan_ego_traj[4,k]     
                    new_state =  np.asarray([[new_t],[new_x],[new_y], [new_v], [new_yaw]])
                    real_ego_traj = np.append(real_ego_traj, new_state, axis = 1)
                    k += 1

        # add some noise to location vector
        ego_location = [pos for pos in zip(list(real_ego_traj[1, :]), list(real_ego_traj[2, :]))]
        obj_location = [pos for pos in zip(list(plan_adv_traj[1, :]), list(plan_adv_traj[2, :]))]

        ego_location = [(pos[0] +random()*randomness_bias,pos[1] +random()*randomness_bias) for pos in ego_location]

        result = {
            "simTime": 0,
            "times": list(real_ego_traj[0, :]),
            "location": {"ego": ego_location,
                         "adversary": obj_location},

            "velocity": {"ego": list(real_ego_traj[3, :]),
                         "adversary": list(plan_adv_traj[3, :]),
                         },

            "speed": {"ego": list(real_ego_traj[3, :]),
                         "adversary": list(plan_adv_traj[3, :]),
                         },

            "acceleration": {"ego": list(real_ego_traj[3, :]),
                         "adversary": list(plan_adv_traj[3, :]),
                         },

            "yaw": {"ego": list(real_ego_traj[4, :]),
                           "adversary": list(plan_adv_traj[4, :]),
                           },

            "collisions": [],
            "actors" : { 
                "ego": "ego",
                "adversary": "adversary",
                "vehicles" : [],
                "pedestrians" : []
            },
            "otherParams": {
            }
        }

        return DrivingSimulationOutput.from_json(json.dumps(result))
    
def are_actors_nearby(pos_ego, pos_others, detection_dist=3):
    """
    Checks if the ego actor is within a certain distance of any actor in a list of positions
    
    Parameters
    ----------
    pos_ego : tuple or list
        The position of the ego actor
    pos_others : list
        A list of positions of other actors
    detection_dist : float
        The distance within which actors are considered nearby
    
    Returns
    -------
    bool
        True if the ego actor is within the specified distance of any other actor, False otherwise
    """
    for i in range(0, len(pos_others)):
        if geometric.dist(pos_ego,pos_others[i]) < detection_dist:
            return True
    return False   

def plan_motion(starting_position, orientation, velocity, sim_time, sampling_time):
    """
    Plans the motion of an ego vehicle in a linear trajectory based on initial conditions.

    Parameters
    ----------
    starting_position : tuple
        The initial (x, y) position of the ego vehicle.
    orientation : float
        The initial orientation (yaw) of the ego vehicle in degrees.
    velocity : float
        The constant velocity of the ego vehicle.
    sim_time : float
        The total simulation time.
    sampling_time : float
        The time interval between each simulation step.

    Returns
    -------
    np.ndarray
        A 2D array representing the trajectory with rows as [Time, X, Y, V, Yaw].
    """
    theta = orientation
    T = sim_time
    v = velocity
    t = sampling_time

    S = v * T
    dist_x = cos(theta * pi / 180) * S
    dist_y = sin(theta * pi / 180) * S
    n_steps = ceil(T / t)

    asize = n_steps + 1
    array_time = np.linspace(0, T, asize)
    array_x = np.linspace(starting_position[0], starting_position[0] + dist_x, asize)
    array_y = np.linspace(starting_position[1], starting_position[1] + dist_y, asize)
    array_v = v * np.ones(asize, dtype=np.int64)
    array_yaw = theta * np.ones(array_time.size)

    return np.concatenate((array_time, array_x, array_y, array_v, array_yaw)).reshape(5, asize)