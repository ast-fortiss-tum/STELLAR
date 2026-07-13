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
pymoo.core.problem.Problem = SimulationProblem
from pymoo.core.problem import Problem
from opensbt.utils.sampling import CartesianSampling

class PureSamplingGrid(PureSampling):
    """ 
    This class provides the Grid Sampling algorithm which generate aquidistant test inputs placed on a grid in the search space.
    """
    def __init__(self,
                    problem: Problem,
                    config: SearchConfiguration,
                    sampling_type = CartesianSampling,
                    **kwargs):
        """Initializes the grid search sampling optimizer.

        :param problem: The testing problem to be solved.
        :type problem: Problem
        :param config: The configuration for the search. The number of samples is equaly for each axis. The axis based sampling number is defined via the population size.
        :type config: SearchConfiguration
        :param sampling_type: Sets by default sampling type to Cartesian Sampling.
        :type sampling_type: _type_, optional
        """
        super().__init__(
            problem = problem,
            config = config,
            sampling_type = sampling_type,
            **kwargs)    
    
        self.algorithm_name = "GS"

        self.parameters["algorithm_name"] = self.algorithm_name
