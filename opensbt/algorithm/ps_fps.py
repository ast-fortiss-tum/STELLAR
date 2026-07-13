import pymoo

from opensbt.algorithm.ps import PureSampling
from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.model_ga.problem import SimulationProblem
from opensbt.utils.fps import FPS
pymoo.core.problem.Problem = SimulationProblem
from pymoo.core.problem import Problem

class PureSamplingFPS(PureSampling):
    """
    This class provides the Farthest Point Sampling algorithm (FPS) [1].
    FPS generate new inputs with the largest distance from the existing inputs to achieve 
    a uniform distribution of test inputs in the search space.

    [1] Eldar Y, Lindenbaum M, Porat M, Zeevi Y (1997) The farthest point strategy for 
    progressive image sampling. IEEE Trans Image Process 6:1305–15, DOI 10.1109/83.623193
    """
    
    algorithm_name = "FPS"

    def __init__(self,
                    problem: Problem,
                    config: SearchConfiguration,
                    sampling_type = FPS,
                    **kwargs):
        """Initializes the furthes point sampling optimizer.

        :param problem: The testing problem to be solved.
        :type problem: Problem
        :param config: The configuration for the search.
        :type config: SearchConfiguration
        :param sampling_type: Sets by default sampling type to FPS.
        :type sampling_type: _type_, optional
        """
        super().__init__(
            problem = problem,
            config = config,
            sampling_type = sampling_type,
            **kwargs)
        
        self.algorithm_name = "FPS"
        self.parameters["algorithm_name"] = self.algorithm_name
