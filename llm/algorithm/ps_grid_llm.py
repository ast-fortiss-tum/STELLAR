import random
import time
from llm.algorithm.cartesian_sampling_utterance import CartesianSamplingUtterance
from llm.llms import LLMType
from opensbt.utils.evaluation import evaluate_individuals
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

class PureSamplingGridLLM(PureSampling):
    """ 
    This class provides the Grid Sampling algorithm which generate aquidistant test inputs placed on a grid in the search space.
    """
    def __init__(self,
                    problem: Problem,
                    config: SearchConfiguration,
                    llm_type = LLMType.GPT_4O):
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
            sampling_type = CartesianSamplingUtterance())    
        self.llm_type = llm_type
        self.algorithm_name = "GS_LLM"

        self.parameters["algorithm_name"] = self.algorithm_name
        
    def run(self) -> SimulationResult:
        """Overrides the run method of Optimizer by providing custom evaluation of samples and division in "buckets" for further analysis with pymoo.
           (s. n_splits variable)
        :return: Return a SimulationResults object which holds all information from the simulation.
        :rtype: SimulationResult
        """
        config = self.config
        random.seed(config.seed)
        
        problem = self.problem
        n_samples_per_feature = config.n_samples_per_feature  # TODO FIXME we use instead the sample_size before; need to find some common generic class
        sampled = self.sampling_type(problem,n_samples_per_feature, 
                                     **dict(llm_type=self.llm_type))
        n_splits = self.n_splits
        start_time = time.time()

        pop = evaluate_individuals(sampled, problem)

        execution_time = time.time() - start_time

        # create result object
        self.res = self._create_result(problem, pop, execution_time, n_splits)
        
        return self.res 
    