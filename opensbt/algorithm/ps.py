

from opensbt.config import EXPERIMENTAL_MODE
from opensbt.visualization.visualizer import create_save_folder
import pymoo
import random
import time
import logging as log

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from pymoo.core.algorithm import Algorithm
from opensbt.algorithm.optimizer import Optimizer

from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling

from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.utils.evaluation import evaluate_individuals

from opensbt.utils.sorting import get_nondominated_population
from pymoo.termination import get_termination

class PureSampling(Optimizer):
    """
    This class provides the parent class for all sampling based search algorithms.
    """
    algorithm_name = "RS"
    time_termination = None

    save_folder: str = None

    def __init__(self,
                problem: Problem,
                config: SearchConfiguration,
                sampling_type = FloatRandomSampling,
                **kwargs):
        """Initializes pure sampling approaches.
        
        :param problem: The testing problem to be solved.
        :type problem: Problem
        :param config: The configuration for the search.
        :type config: SearchConfiguration
        :param sampling_type: Sets by default sampling type to RS.
        :type sampling_type: _type_, optional
        """
        self.config = config
        self.problem = problem
        self.res = None
        self.sampling_type = sampling_type
        self.sample_size = config.population_size
        self.n_splits = 10 # divide the population by this size 
                         # to make the algorithm iterative for further analysis
        self.parameters = { 
                            "sample_size" : self.sample_size
        }

        self.parameters["algorithm_name"] = self.algorithm_name
        self.parameters.update(kwargs)
        self.callback = kwargs.get("callback", None)
        self.algorithm = Algorithm(archive=PopulationExtended([]))
        log.info(f"Initialized algorithm with config: {config.__dict__}")

    def run(self) -> SimulationResult:
        """Run optimization with iterative evaluation of samples using pymoo's termination 
        criteria (can be based on evaluations, search time, or other criteria).

        :return: A SimulationResult object which holds all information from the simulation.
        :rtype: SimulationResult
        """
        config = self.config
        random.seed(config.seed)

        problem = self.problem
        sample_size = self.sample_size
        n_splits = self.n_splits

        self.save_folder = create_save_folder(self.problem, 
                                    self.config.results_folder,
                                    algorithm_name=self.algorithm_name,
                                    is_experimental=EXPERIMENTAL_MODE)
        
        # time has priority over number of samples
        if config.maximal_execution_time is not None:
            self.time_termination = get_termination("time", config.maximal_execution_time)
     
        self.max_time = config.maximal_execution_time

        start_time = time.time()
        self.start_time = start_time
        evaluated_pop = PopulationExtended()
        n_evals = 0

         # Iteratively evaluate individuals until termination is satisfied;  not more than sample size can be evaluated
        for _ in range(sample_size):
            sampled = config.operators.sampling(problem, 1)
            evaluated = evaluate_individuals(PopulationExtended(individuals=sampled), problem)
            evaluated_pop = PopulationExtended.merge(evaluated_pop, evaluated)
            n_evals += 1
            self.algorithm.archive = evaluated_pop
            self.algorithm.pop = evaluated_pop
            if self.callback is not None:
                self.callback(self.algorithm)

            if self.time_termination is not None:
                self.time_termination.update(self)
                if self.time_termination.has_terminated():
                    break
                
        execution_time = time.time() - start_time

        # Create result object after iterative evaluation
        self.res = self._create_result(problem, evaluated_pop, execution_time, n_splits)

        return self.res

    def _create_result(self, problem, pop, execution_time, n_splits):
        res_holder = SimulationResult()
        res_holder.algorithm = Algorithm()
        res_holder.algorithm.pop = pop
        res_holder.algorithm.archive = pop
        res_holder.algorithm.evaluator.n_eval = len(pop)
        res_holder.problem = problem
        res_holder.algorithm.problem = problem
        res_holder.exec_time = execution_time
        res_holder.opt = get_nondominated_population(pop)
        res_holder.algorithm.opt = res_holder.opt
        res_holder.archive = pop

        res_holder.history = []  # history is the same instance 
        n_bucket = len(pop) // n_splits
    
        pop_sofar = 0
        for i in range(0,n_splits):
            
            algo = Algorithm()
            algo.pop = pop[(i*n_bucket):min((i+1)*n_bucket,len(pop))]
            algo.archive = pop[:min((i+1)*n_bucket,len(pop))]
            pop_sofar += len(algo.pop)
            algo.evaluator.n_eval = pop_sofar
            if len(algo.pop) > 0:
                algo.opt = get_nondominated_population(algo.pop)
                res_holder.history.append(algo)
        
        return res_holder