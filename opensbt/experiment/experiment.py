from dataclasses import dataclass
from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.model_ga.problem import SimulationProblem as Problem

@dataclass
class Experiment(object):
    """ The experiment class holds all information to execute a testing experiment and 
        allows to maintain internally different experiment configurations to seperate experiments 
        from each other.
    """
    def __init__(self, name: str, problem: Problem, algorithm: str, search_configuration: SearchConfiguration):        
        """Create an Experiment.

        :param name: Name of the experiment to identify it.
        :type name: str
        :param problem: The corresponding testing problem.
        :type problem: Problem
        :param algorithm: The algorithm string tag (enum) to use for testing.
        :type algorithm: str
        :param search_configuration: The search configuration.
        :type search_configuration: SearchConfiguration
        """
        self.name = name
        self.problem = problem
        self.algorithm = algorithm
        self.search_configuration = search_configuration

