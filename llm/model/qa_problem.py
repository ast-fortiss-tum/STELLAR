from dataclasses import dataclass
from typing import Dict, List, Set, Optional
from pymoo.core.problem import Problem
import numpy as np
import wandb
from opensbt.evaluation.critical import Critical
from opensbt.evaluation.fitness import Fitness
import logging as log
from llm.config import CONTEXT
from llm.utils.seed import set_seed
from llm.features import FeatureHandler
from llm.utterance_generation import SeedSampler, UtteranceGenerator

@dataclass
class QAProblem(Problem):
    """ Basic problem class for Conversational Systems Testing problems """
    
    def __init__(self,
        xl: List[float],
        xu: List[float],
        scenario_path: str,
        fitness_function: Fitness,
        simulate_function,
        critical_function: Critical,
        simulation_variables: Optional[List[str]] = None,
        design_names: List[str] = None,
        objective_names: List[str] = None,
        problem_name: str = None,
        other_parameters: Dict = None,
        seed_utterances: List[str] = ["I am hungry"],
        context: str = CONTEXT,
        temperature: float = 0,
        names_dim_utterance: List[str] = None,  # for the original discretization approach
        feature_handler_config_path: Optional[str] = None,
        seed_sampler: Optional[SeedSampler] = None,
        question_generator: Optional[UtteranceGenerator] = None,
        seed: int = 0):   
    
        super().__init__(n_var=len(xl),
                         n_obj=len(fitness_function.name),
                         xl=xl,
                         xu=xu)

        assert xl is not None
        assert xu is not None
        assert scenario_path is not None
        assert fitness_function is not None
        assert simulate_function is not None
        assert np.equal(len(xl), len(xu))
        assert np.less_equal(xl, xu).all()
        assert len(fitness_function.min_or_max) == len(fitness_function.name)

        self.fitness_function = fitness_function
        self.simulate_function = simulate_function
        self.critical_function = critical_function
        self.simulation_variables = simulation_variables

        if feature_handler_config_path is not None:
            self.feature_handler = FeatureHandler.from_json(feature_handler_config_path)
            wandb.log(self.feature_handler.model_dump())
            
        self.seed_sampler = seed_sampler
        self.question_generator = question_generator
        if self.question_generator is not None:
            self.question_generator.feature_handler = self.feature_handler

        self.seed = seed
 
        set_seed(seed)

        if design_names is not None:
            self.design_names = design_names
        else:
            self.design_names = simulation_variables

        if objective_names is not None:
            self.objective_names = objective_names
        else:
            self.objective_names = fitness_function.name

        self.scenario_path = scenario_path
        self.problem_name = problem_name
        self.seed_utterances = seed_utterances
        self.context = context
        self.temperature = temperature
        self.names_dim_utterance = names_dim_utterance

        if other_parameters is not None:
            self.other_parameters = other_parameters

        self.signs = []
        for value in self.fitness_function.min_or_max:
            if value == 'max':
                self.signs.append(-1)
            elif value == 'min':
                self.signs.append(1)
            else:
                raise ValueError(
                    "Error: The optimization property " + str(value) + " is not supported.")

        self.counter = 0

    def _evaluate(self, x, out, *args, **kwargs):
        self.counter = self.counter + 1
        log.info(f"Running evaluation number {self.counter}")
        #log.info("[Evaluate] utterances", x)
        try:
            log.info(f"context: {self.context}")
            # it is assumed that the i-th simout in the simout list correponds to i-th scenario in the list x
            simout_list = self.simulate_function(x, 
                                                 self.simulation_variables, 
                                                 self.scenario_path, 
                                                 sim_time=-1,
                                                 time_step=-1, 
                                                 do_visualize=False,
                                                 temperature = self.temperature,
                                                 context = self.context)
        except Exception as e:
            log.info("Exception during simulation ocurred: ")
            # TODO handle exception, terminate, so that results are stored
            raise e
        out["SO"] = []
        vector_list = []
        label_list = []

        for i, simout in enumerate(simout_list):
            kwargs["individual"] = x[i]

            out["SO"].append(simout)
            
            vector_fitness = np.asarray(
                self.signs) * np.array(self.fitness_function.eval(simout, **kwargs))

            vector_list.append(np.array(vector_fitness))
            label_list.append(self.critical_function.eval(vector_fitness, simout = simout))

        out["F"] = np.vstack(vector_list)
        out["CB"] = label_list
        
    def is_simulation(self):
        return True
