import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.algorithm.ps import PureSampling
from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.model_ga.problem import SimulationProblem
from opensbt.model_ga.result import SimulationResult

from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling

class PureSamplingRand(PureSampling):
    """ 
    This class provides the random sampling algorithm which generates random test inputs in the search space.
    """
    def __init__(self,
                problem: Problem,
                config: SearchConfiguration,
                sampling_type = FloatRandomSampling,
                **kwargs):
        """Initializes the random sampling optimizer.
        
        :param problem: The testing problem to be solved.
        :type problem: Problem
        :param config: The configuration for the search.
        :type config: SearchConfiguration
        :param sampling_type: Sets by default sampling type to RS.
        :type sampling_type: _type_, optional
        """
        super().__init__(
            problem = problem,
            config = config,
            sampling_type = sampling_type,
            **kwargs) 

        self.algorithm_name = "RS"

        self.parameters["algorithm_name"] = self.algorithm_name
