from opensbt.simulation.simulator import SimulationOutputBase, DrivingSimulationOutput
import numpy as np
from typing import List

class Critical():
    """This class defines an interface for concrete oracle functions. 
       The functions name and eval have to be implemented.
    """
    @property
    def name(self):
        """The name of the criticality function. To be overriden for specific name. Otherwise class name is used.

        :return: Name of the criticality function.
        :rtype: str
        """
        return self.__class__.__name__

    def eval(self, vector_fitness: np.ndarray, simout: SimulationOutputBase) -> bool:
        """Evaluate whether a test has failed or passed. To be implemented based on specific conditions.
        
        :param vector_fitness: List of fitness values.
        :type vector_fitness: np.ndarray
        :param simout: SimulationOutputBase instance
        :type simout: SimulationOutputBase
        :return: Returns True if simulation is critical/failed or otherwise False.
        :rtype: bool
        """
        pass

class MockCritical():
    @property
    def name(self):
        return self.__class__.__name__

    def eval(self, vector_fitness: np.ndarray, simout: SimulationOutputBase) -> bool:
        return True

class CriticalAdasExplicitRearCollision(Critical):

    ''' ADAS problems '''
    def eval(self, vector_fitness: List[float], simout: DrivingSimulationOutput = None):
        safety_distance = 0.50 # all dimensions, radial, in m

        if simout is not None and "is_collision" in simout.otherParams:
            isCollision = simout.otherParams['isCollision']
        else:
            isCollision = None
        # a) collision occurred (not from the rear) and velocity of ego > 0
        # b) safety distance to pedestrian violated
        if isCollision:
            loc_ego = simout.location["ego"]
            loc_ped = simout.location["other"]
            diff = [np.subtract(pos_e,pos_p) for pos_e, pos_p in zip(loc_ego, loc_ped)]
            distance = np.linalg.norm(diff)
            ind_col= np.argmin(distance)
            dist_x = abs(simout.location["ego"][ind_col][0] - simout.location["other"][ind_col][0])
            raw_dist_y = simout.location["ego"][ind_col][1] - simout.location["other"][ind_col][1]
            if dist_x < simout.otherParams["height"]/2 and raw_dist_y > 0 and vector_fitness[1] < 0:
                return True
            return False
        elif (vector_fitness[0]  - safety_distance <= 0 and vector_fitness[1] < 0):
            return True
        else:
            return False

class CriticalAdasFrontCollisions(Critical):
    def eval(self, vector_fitness, simout: DrivingSimulationOutput = None):
        if simout is not None and "is_collision" in simout.otherParams:
            isCollision = simout.otherParams['isCollision']
        else:
            isCollision = None

        if (isCollision == True) or (vector_fitness[0] < 0.5) and (vector_fitness[1] < 0):
            return True
        else:
            return False

class CriticalAdasTTCVelocity(Critical):
    def eval(self, vector_fitness, simout: DrivingSimulationOutput = None):
        if simout is not None and "is_collision" in simout.otherParams:
            isCollision = simout.otherParams['isCollision']
        else:
            isCollision = None

        if(isCollision == True) or (vector_fitness[0] < 5) and (vector_fitness[1] < -1):
            return True
        else:
            return False
        
class CriticalAdasTTC(Critical):
    def eval(self, vector_fitness, simout: DrivingSimulationOutput = None):
        if (vector_fitness[0] < 0.01):
            return True
        else:
            return False


    '''
        f[0] - min distance ego <-> pedestrian
        f[1] - velocity at time of minimal distance

        # Scenario critical <->

        # a) collision ocurred
        # b) minimal distance between ego and other vehicle < 0.3m
        # c) velcoty at time of minimal distance is > 1 m/s
    '''
class CriticalAdasDistanceVelocity(Critical):
    def eval(self, vector_fitness, simout: DrivingSimulationOutput = None):
        if simout is not None and "is_collision" in simout.otherParams:
            isCollision = simout.otherParams['isCollision']
        else:
            isCollision = None

        if(isCollision == True) or (vector_fitness[0] < 0.3) and (vector_fitness[1] < -1):
            return True
        else:
            return False
        
''' Test problems '''
class CriticalBnhDivided(Critical):
    def eval(self, vector_fitness: np.ndarray, simout=None):
        return  (vector_fitness[0] < 10 ) and \
                (vector_fitness[1] < 50) and  (vector_fitness[1] > 20) or \
                (vector_fitness[0] < 140 ) and (vector_fitness[0] > 40 )  and \
                (vector_fitness[1] < 7) and  (vector_fitness[1] > 0)  or \
                (vector_fitness[0] < 40 ) and (vector_fitness[0] > 20 )  and \
                (vector_fitness[1] < 20) and  (vector_fitness[1] > 15)

class CriticalBnh(Critical):
    def eval(self, vector_fitness):
        return (vector_fitness[0] < 60) and (vector_fitness[1] < 20)

class CriticalRastrigin(Critical):
    def eval(self, fitness):
        return fitness < 2 and fitness > -2
