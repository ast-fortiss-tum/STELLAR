import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.algorithm.optimizer import Optimizer
from opensbt.utils.operators import select_operator

from llm.model.qa_simout import QASimulationOutput

from algorithm.nsga2d.nsga2d import NSGA2D

from opensbt.utils.archive import MemoryArchive

import os
import sys
from pathlib import Path
from typing import List, Callable

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from opensbt.algorithm.classification.decision_tree.decision_tree import *
from opensbt.evaluation.critical import Critical, CriticalBnhDivided
from opensbt.experiment.search_configuration import DefaultSearchConfiguration, SearchConfiguration
from opensbt.problem.pymoo_test_problem import PymooTestProblem
import logging as log
from opensbt.problem import *
from opensbt.model_ga.result import *
from config import *
from pymoo.operators.sampling.lhs import LHS
from llm.utils.embeddings_openai import get_similarity
from llm.adapter.embeddings_local_adapter import get_similarity_individual

class NSGAIIDOptimizer(Optimizer):
    
    algorithm_name =  "NSGA2D"

    save_folder: str

    def __init__(self,
                problem: Problem,
                config: SearchConfiguration,
                dist_function: Callable = get_similarity_individual,
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

        self.algorithm = NSGA2D(
            pop_size=config.population_size,
            n_offsprings=config.num_offsprings,
            sampling = operators.sampling,
            crossover = operators.crossover,
            mutation = operators.mutation,
            eliminate_duplicates = operators.duplicate_elimination,
            n_repopulate_max=config.n_repopulate_max,
            archive_threshold=config.archive_threshold,
            seed = config.seed,
            archive= MemoryArchive(),
            dist_fnc = dist_function,
            bounds_normalize=
                [
                    self.problem.xu,
                    self.problem.xl
                ],
            **kwargs
            )

        ''' Prioritize max search time over set maximal number of generations'''
        if config.maximal_execution_time is not None:
            self.termination = get_termination("time", config.maximal_execution_time)
        else:
            self.termination = get_termination("n_gen", config.n_generations)

        self.save_history = True

        self.parameters["algorithm_name"] = self.algorithm_name

        log.info(f"Initialized algorithm with config: {config.__dict__}")
    
if __name__ == "__main__":        
            
    problem = PymooTestProblem(
        'bnh', critical_function=CriticalBnhDivided())
    config = DefaultSearchConfiguration()

    config.population_size = 100
    config.n_generations = 10
    config.prob_mutation = 0.5
    config.n_func_evals_lim = 20
    config.n_repopulate_max = 10

    optimizer = NSGAIIDOptimizer(problem,config)
    optimizer.run()
    optimizer.write_results(
        ref_point_hv=np.asarray([200,200]), 
        ideal = np.asarray([0,0]), 
        nadir = np.asarray([200,200]), 
        results_folder = RESULTS_FOLDER)