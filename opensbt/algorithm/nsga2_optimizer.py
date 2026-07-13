import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.utils.operators import select_operator
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from opensbt.algorithm.optimizer import Optimizer
from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.model_ga.result import *
from pymoo.util.archive import MultiObjectiveArchive

from opensbt.utils.archive import MemoryArchive

class NsgaIIOptimizer(Optimizer):
    """ This optimizer class provides the NSGA-II algorithm which is already implemented in pymoo.
    """
    
    algorithm_name = "NSGA-II"

    def __init__(self,
                problem: Problem,
                config: SearchConfiguration,
                **kwargs):
        self.config = config
        self.problem = problem
        self.res = None

        if self.config.prob_mutation is None:
            self.config.prob_mutation = 1 / problem.n_var

        self.parameters = {
            "Population size" : str(config.population_size),
            "Number of generations" : str(config.n_generations),
            "Number of offsprings": str(config.num_offsprings),
            "Crossover probability" : str(config.prob_crossover),
            "Crossover eta" : str(config.eta_crossover),
            "Mutation probability" : str(config.prob_mutation),
            "Mutation eta" : str(config.eta_mutation),
            "Seed" : str(config.seed),
            "Operators" : config.operators
        }

        operators = config.operators

        self.algorithm = NSGA2(
            pop_size=config.population_size,
            n_offsprings=config.num_offsprings,
            sampling = operators.sampling,
            crossover = operators.crossover,
            mutation = operators.mutation,
            eliminate_duplicates = operators.duplicate_elimination,
            repair = operators.repair,
            archive=MemoryArchive(),
            **kwargs
        )

        ''' Prioritize max search time over set maximal number of generations'''
        if config.maximal_execution_time is not None:
            self.termination = get_termination("time", config.maximal_execution_time)
        else:
            self.termination = get_termination("n_gen", config.n_generations)

        self.save_history = True
        
        self.parameters["algorithm_name"] = self.algorithm_name

