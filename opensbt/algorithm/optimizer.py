import os

from abc import ABC, abstractclassmethod, abstractmethod
from typing import Dict

from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.model_ga.problem import SimulationProblem
from opensbt.model_ga.result import SimulationResult
from pymoo.optimize import minimize
from pymoo.core.problem import Problem  
from pymoo.core.algorithm import Algorithm

import dill
from opensbt.config import BACKUP_FOLDER, RESULTS_FOLDER, EXPERIMENTAL_MODE, BACKUP_ITERATIONS
from opensbt.visualization.visualizer import create_save_folder, backup_object, delete_backup

class Optimizer(ABC):
    """ Base class for all optimizers in OpenSBT.  Subclasses need to   
        implement the __init__ method. The run method has to be overriden when non pymoo implemented algorithms are used.
        For reference consider the implementation of the NSGA-II-DT optimizer in opensbt/algorithm/nsga2dt_optimizer.py
    """
    
    algorithm_name: str
    parameters: Dict
    config: SearchConfiguration
    problem: Problem
    algorithm: Algorithm
    termination: object
    save_history: bool
    
    parameters: str
    save_folder: str = None
    
    @abstractmethod
    def __init__(self, problem: SimulationProblem, config: SearchConfiguration, **kwargs):
        """Initialize here the Optimization algorithm to be used for search-based testing.

        :param problem: The testing problem to be solved.
        :type problem: SimulationProblem
        :param config: The configuration for the search.
        :type config: SearchConfiguration
        """
        pass

    def run(self) -> SimulationResult:
        # create a backup during the search for each generation
        algorithm = self.algorithm
        algorithm.setup(problem = self.problem, 
                        termination = self.termination,
                        save_history = self.save_history)
        self.save_folder = create_save_folder(self.problem, 
                                self.config.results_folder,
                                algorithm_name=self.algorithm_name,
                                is_experimental=EXPERIMENTAL_MODE)
        print("[Optimizer] save_folder created:", self.save_folder)
        while(algorithm.termination.do_continue()):
            algorithm.next()
            if BACKUP_ITERATIONS:
                n_iter = algorithm.n_iter - 1
                dill.detect.trace(True)   # will show what dill is trying to pickle
                # store backup for current iteration
                backup_object(algorithm, 
                            self.save_folder, 
                            name = f"algorithm_iteration_{n_iter}")
                
                # delete old backup
                delete_backup(save_folder = self.save_folder + BACKUP_FOLDER,
                            name = f"algorithm_iteration_{n_iter - 1}")
                                                
        res = SimulationResult.from_result(algorithm.result())
        return res