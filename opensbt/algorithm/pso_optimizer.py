from opensbt.utils.operators import select_operator
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.termination import get_termination
from opensbt.algorithm.optimizer import Optimizer
from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.model_ga.result import *
import logging as log

from opensbt.utils.archive import MemoryArchive

class PSOOptimizer(Optimizer):
    """
        This class provides search with the Particle Swarm Optimization algorithm which is already implemented in pymoo.
    """

    algorithm_name = "PSO"

    def __init__(self,
                problem: Problem,
                config: SearchConfiguration):
        """Initializes the particle swarm opimization approach for testing.

        :param problem: The testing problem to be solved.
        :type problem: SimulationProblem
        :param config: The configuration for the search.
        :type config: SearchConfiguration
        """
        log.info("Initialized PSO Optimizer")
        
        self.config = config
        self.problem = problem
        self.res = None

        # TODO set other PSO parameters by user
        self.parameters = {
            "Population size" : str(config.population_size),
            "Max number of generations" : str(config.n_generations),
        }

        # initialize algorithm
        self.algorithm = PSO(
            pop_size = config.population_size,
            sampling = select_operator("init",config),
            archive = MemoryArchive()
        )

        ''' Prioritize max search time over set maximal number of generations'''
        if config.maximal_execution_time is not None:
            self.termination = get_termination("time", config.maximal_execution_time)
        else:
            self.termination = get_termination("n_gen", config.n_generations)

        self.save_history = True