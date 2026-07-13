import sys
from typing import Tuple
from opensbt.simulation.simulator import SimulationOutputBase, DrivingSimulationOutput
import numpy as np
import math
from opensbt.utils import geometric


class Fitness():
    """This class defines an interface for concrete fitness functions. 
    """
    
    
    @property
    def min_or_max(self):
        """Defines for each objective if it is minimized or maximized. Returns a tuple, where each element holds 
           the value "min" or "max".
        """
        pass

    @property
    def name(self):     
        """Defines the name of the fitness funtion as a tuple, where each element corresponds to the name of the objective
        """
        pass
 
    def eval(self, simout: SimulationOutputBase, **kwargs) -> Tuple[float]:    
        """Returns the fitness value of a given SimulationOutput instance.
    
        :param simout: SimulationOutputBase instance.
        :type simout: SimulationOutputBase 
        :param kwargs: Further optional variables needed for fitness evaluation. 
        :type kwargs: **Dict 
        :return:: Return the fitness value for the given simulation results. Each tuple element corresponds to value for the specific fitness dimension.
        :rtype Tuple[float]
        """
        pass

class MockFitness():
    @property
    def min_or_max(self):
        return "min","min"

    @property
    def name(self):
        return "dimension_1","dimension_2"

    def eval(self, simout: SimulationOutputBase, **kwargs) -> Tuple[float]:
        return (0,0)
        
class FitnessMinDistance(Fitness):
    @property
    def min_or_max(self):
        return ("min",)

    @property
    def name(self):
        return "Min distance"

    def eval(self, simout: DrivingSimulationOutput, **kwargs) -> Tuple[float]:
        if "distance" in simout.otherParams:
            dist = simout.otherParams["distance"]
            result = min(dist)
        else:
            trace_ego = simout.location["ego"]
            trace_ped = simout.location["adversary"]
            result = np.min(geometric.distPair(trace_ego, trace_ped))
        return result

class FitnessMinDistanceVelocity(Fitness):
    @property
    def min_or_max(self):
        return "min", "max"

    @property
    def name(self):
        return "Min distance", "Velocity at min distance"

    def eval(self, simout: DrivingSimulationOutput, **kwargs) -> Tuple[float]:
        if "adversary" in simout.location:
            name_adversary = "adversary"
        else:
            name_adversary = "other"

        trace_ego = simout.location["ego"]
        trace_ped = simout.location[name_adversary]

        ind_min_dist = np.argmin(geometric.distPair(trace_ego, trace_ped))

        # distance between ego and other object
        distance = np.min(geometric.distPair(trace_ego, trace_ped))

        # speed of ego at time of the minimal distance
        speed = simout.speed["ego"][ind_min_dist]

        return (distance, speed)

class FitnessMinDistanceVelocityFrontOnly(Fitness):
    @property
    def min_or_max(self):
        return "min", "max"

    @property
    def name(self):
        return "Min distance", "Velocity at min distance"

    def eval(self, simout: DrivingSimulationOutput, **kwargs) -> Tuple[float]:
        if "adversary" in simout.location:
            name_adversary = "adversary"
        else:
            name_adversary = "other"
        car_length = float(4.0)
        trace_ego = simout.location["ego"]
        trace_ped = simout.location[name_adversary]
        ind_min_dist = np.argmin(geometric.distPair(trace_ego, trace_ped))
        # approx distance between ego's front and other object
        distance = np.min(geometric.distPair(trace_ego, trace_ped))  - car_length/2
        # speed of ego at time of the minimal distance
        speed = simout.speed["ego"][ind_min_dist]
        # value scenarios worse if pedestrian is not in front of the car
        FITNESS_WORSE = 1000
        if (trace_ego[ind_min_dist][0] -  trace_ped[ind_min_dist][0] < car_length/2):
            distance = FITNESS_WORSE
            speed = -FITNESS_WORSE
        return (distance, speed)

class FitnessMinTTC(Fitness):
    @property
    def min_or_max(self):
        return "min",

    @property
    def name(self):
        return "Min TTC",

    def eval(self, simout: DrivingSimulationOutput, **kwargs) -> Tuple[float]:
        all_ttc = []
        if "adversary" in simout.location:
            name_adversary = "adversary"
        else:
            name_adversary = "other"

        for i in range(2, len(simout.times)):
            ego_location = simout.location["ego"]
            adv_location = simout.location[name_adversary]

            colpoint = geometric.intersection(
                (ego_location[i], ego_location[i-1]), (adv_location[i], adv_location[i-1]))

            if colpoint == []:
                all_ttc.append(sys.maxsize)
            else:
                velocity_ego = simout.speed["ego"]
                velocity_adv = simout.speed[name_adversary]
                dist_ego_colpoint = geometric.dist(colpoint, ego_location[i])
                dist_adv_colpoint = geometric.dist(colpoint, adv_location[i])

                if colpoint == []:
                    all_ttc.append(sys.maxsize)
                    continue
                if velocity_ego[i] == 0 or velocity_adv[i] == 0:
                    all_ttc.append(sys.maxsize)
                    continue

                t_col_ego = dist_ego_colpoint/velocity_ego[i]
                t_col_ped = dist_adv_colpoint/velocity_adv[i]
                t_tolerance = 0.5  # time tolerance for missed collision
                if abs(t_col_ego - t_col_ped) < t_tolerance:
                    all_ttc.append(t_col_ego)
                else:
                    all_ttc.append(t_col_ego)

        min_ttc = np.min(all_ttc)
        return min_ttc

class FitnessMinTTCVelocity(Fitness):
    @property
    def min_or_max(self):
        return "min", "max"

    @property
    def name(self):
        return "Min TTC", "Critical Velocity"

    def eval(self, simout: DrivingSimulationOutput, **kwargs) -> float:
        if "adversary" in simout.location:
            name_adversary = "adversary"
        else:
            name_adversary = "other"

        for i in range(2, len(simout.times)):
            ego_location = simout.location["ego"]
            adv_location = simout.location[name_adversary]

            colpoint = geometric.intersection(
                (ego_location[i], ego_location[i-1]), (adv_location[i], adv_location[i-1]))
            all_ttc = []
            # If no collision, return huge value
            if colpoint == []:
                min_ttc = sys.maxsize
                velocity_min_ttc = sys.maxsize
            else:
                velocity_ego = simout.speed["ego"]
                velocity_adv = simout.speed[name_adversary]

                dist_ego_colpoint = geometric.dist(colpoint, ego_location[i])
                dist_adv_colpoint = geometric.dist(colpoint, adv_location[i])

                if colpoint == []:
                    all_ttc.append(sys.maxsize)
                    continue
                if velocity_ego[i] == 0 or velocity_adv[i] == 0:
                    all_ttc.append(sys.maxsize)
                    continue

                t_col_ego = dist_ego_colpoint/velocity_ego[i]
                t_col_ped = dist_adv_colpoint/velocity_adv[i]

                t_tolerance = 0.5  # time tolerance for missed collision
                if abs(t_col_ego - t_col_ped) < t_tolerance:
                    all_ttc.append(t_col_ego)
                else:
                    all_ttc.append(t_col_ego)

                time_min_ttc = np.argmin(all_ttc)
                min_ttc = all_ttc[time_min_ttc]
                velocity_min_ttc = velocity_ego[time_min_ttc]
        result = (min_ttc, velocity_min_ttc)
        return result

class FitnessAdaptedDistSpeedRelVelocity(Fitness):
    @property
    def min_or_max(self):
        return "max", "max", "max"

    @property
    def name(self):
        return "Critical Adapted distance", "Velocity at critical distance", "Relative velocity at critical distance"

    ''' Fitness function to resolve front and rear collisions'''

    def fitness_parallel(self, z_parallel, car_length):
        """
        Input:
            z_parallel, which is a projection of relative position of a car's front bumper and a pedestrian to the
            axis, parallel to a car velocity.
            car_length, which is a length of a car.
        Returns:
            a value between 0 and 1, indicating severeness of the relative position of a car and a pedestrian in the
            parallel direction. The higher the value - the more severe a scenario is.

        The fitness function is composed of exponential functions and constant functions. step_back characterizes
        steepness of decay of the fitness function behind the front bumper. step_front characterizes steepness of decay
        of the fitness function in front of the front bumper. The value of the fitness function for positions of a
        pedestrian behind the back bumper is 0.
        """
        steep_back = 10
        steep_front = 2
        # z = yPed - yEgo - car_length/2
        n = len(z_parallel)
        result = np.zeros(n)
        for i in range(n):
            if z_parallel[i] < -car_length:
                result[i] = 0
            elif z_parallel[i] < 0:
                result[i] = (np.exp(steep_back * (z_parallel[i] + car_length)) - np.exp(0)) / np.exp(
                    steep_back * car_length)
            else:
                result[i] = np.exp(-steep_front * z_parallel[i]) / np.exp(0)
        return result

    def fitness_perpendicular(self, z_perpendicular, car_width):
        """
        Input:
            z_perpendicular, which is a projection of relative position of a car's center and a pedestrian to the axis,
            perpendicular to a car velocity.
            car_width, which is a width of a car.

        Returns:
            a value between 0 and 1, indicating severeness of the relative position of a car and a pedestrian in the
            perpendicular direction. The higher the value - the more severe a scenario is.

        The fitness function is composed of a "bell-shaped" function segments on the sides, and a constant function in
        the middle equal to 1. A "bell-shaped" function is a gaussian function. sigma is proportional to the width of a
        "bell". A constant segment has a length of the car_width. The parallel fitness function, combined with the
        perpendicular fitness function, e.g. by multiplication, results in a proximity function, which defines a
        severeness of the relative position of a car and a pedestrian.
        """
        sigma = 0.05
        # z = xPed - xEgo
        n = len(z_perpendicular)
        result = np.zeros(n)
        for i in range(n):
            if abs(z_perpendicular[i]) < car_width / 2:
                result[i] = 1
            else:
                result[i] = np.exp(-(0.5 / sigma) *
                                   (abs(z_perpendicular[i]) - car_width / 2) ** 2)
        return result

    def eval(self, simout: DrivingSimulationOutput, **kwargs) -> float:
        if "car_length" in simout.otherParams:
            car_length = float(simout.otherParams["car_length"])
        else:
            car_length = float(4.3)

        if "car_width" in simout.otherParams:
            car_width = float(simout.otherParams["car_width"])
        else:
            car_width = float(1.8)

        if "adversary" in simout.location:
            name_adversary = "adversary"
        else:
            name_adversary = "other"

        # time series of Ego position
        trace_ego = np.array(simout.location["ego"])
        # time series of Ped position
        trace_adv = np.array(simout.location[name_adversary])

        # time series of Ego position
        velocity_ego = np.array(simout.velocity["ego"])
        # time series of Ped position
        velocity_adv = np.array(simout.velocity[name_adversary])
        velocity_relative = velocity_adv - velocity_ego

        # time series of Ego velocity
        speed_ego = np.array(simout.speed["ego"])
        yaw_ego = np.array(simout.yaw["ego"])  # time series of Ego velocity

        # Global coordinates 
        x_ego = trace_ego[:, 0]
        y_ego = trace_ego[:, 1]
        x_adv = trace_adv[:, 0]
        y_adv = trace_adv[:, 1]

        # Coordinates, with respect to ego: e2 is parallel to the direction of ego
        e2_x = np.cos(yaw_ego * math.pi / 180)
        e2_y = np.sin(yaw_ego * math.pi / 180)
        e1_x = e2_y
        e1_y = -e2_x

        z_parallel = (x_adv - x_ego) * e2_x + \
            (y_adv - y_ego) * e2_y - car_length / 2
        z_perpendicular = (x_adv - x_ego) * e1_x + (y_adv - y_ego) * e1_y

        f_1 = self.fitness_parallel(z_parallel, car_length)
        f_2 = self.fitness_perpendicular(z_perpendicular, car_width)
        critical_iteration = np.argmax(f_1 * f_2)

        vector_fitness = (f_1[critical_iteration] * f_2[critical_iteration],
                          speed_ego[critical_iteration],
                          np.linalg.norm(velocity_relative[critical_iteration]))

        # Maybe speed and relative velocity should not be taken at the critical iteration? But at some steps before that.
        # For some reason, when the front collision happens,
        # F1 is not equal to 1 for carla simulation, but somewhere in between 0.5 and 1.0.
        # It happens for f_1. It happens because of poor sampling at the moment of collision.
        # The parallel fitness function should be modified to account for that, or carla settings.
        return vector_fitness


class FitnessAdaptedDistanceSpeed(Fitness):
    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Critical adapted distance", "Velocity at critical distance"

    def eval(self, simout: DrivingSimulationOutput, **kwargs) -> float:
        # use only adapted distance and velocity of the fitness comupation of existing function
        vector_fitness_all = FitnessAdaptedDistSpeedRelVelocity().eval(simout)
        adapted_distance = vector_fitness_all[0]
        speed = vector_fitness_all[1]
        return adapted_distance, speed


class FitnessAdaptedDistanceSpeedTTC(Fitness):
    @property
    def min_or_max(self):
        return "min", "max", "min"

    @property
    def name(self):
        return "Critical adapted distance", "Velocity at critical distance", "Min TTC"

    def eval(self, simout: DrivingSimulationOutput, **kwargs) -> float:
        min_ttc = FitnessMinTTC().eval(simout)
        pos_crit = FitnessAdaptedDistanceSpeed().eval(simout)
        return pos_crit[0], pos_crit[1], min_ttc