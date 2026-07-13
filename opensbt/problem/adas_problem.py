from dataclasses import dataclass
from typing import Dict, List
from pymoo.core.problem import Problem
import numpy as np
from opensbt.evaluation.critical import Critical
from opensbt.evaluation.fitness import Fitness
import logging as log

@dataclass
class ADASProblem(Problem):
    """ Basic problem class for ADAS problems """
    
    def __init__(self,
        xl: List[float],
        xu: List[float],
        scenario_path: str,
        fitness_function: Fitness,
        simulate_function,
        critical_function: Critical,
        simulation_variables: List[str],
        simulation_time: float = 10,
        sampling_time: float = 100,
        design_names: List[str] = None,
        objective_names: List[str] = None,
        problem_name: str = None,
        other_parameters: Dict = None,
        approx_eval_time: float = None,
        do_visualize: bool = False):
        """Generates a simulation-based testing problem.

        :param xl: Lower bound for the search domain.
        :type xl: List[float]
        :param xu: Upper bound for the search domain.
        :type xu: List[float]
        :param scenario_path: The path to the scenario to be simulated.
        :type scenario_path: str
        :param fitness_function: The instance of the fitness function to evaluate simulations.
        :type fitness_function: Fitness
        :param simulate_function: The pointer to the simulate function of the simulator.
        :type simulate_function: _type_
        :param critical_function: The instance of the oracle function to assign pass fail verdicts.
        :type critical_function: Critical
        :param simulation_variables: The name of the simulation variables to alter for test generation.
        :type simulation_variables: List[float]
        :param simulation_time: The simulation time for the test execution, defaults to 10
        :type simulation_time: float, optional
        :param sampling_time: The sampling time for the simulator, defaults to 100
        :type sampling_time: float, optional
        :param design_names: The name for the search variable to use in the output. If none is said, simulation_variables is used, defaults to None
        :type design_names: List[str], optional
        :param objective_names: The name of the objectives to use in the output, defaults to None
        :type objective_names: List[str], optional
        :param problem_name: The name of the problem, defaults to None
        :type problem_name: str, optional
        :param other_parameters: Other necessary parameters, defaults to None
        :type other_parameters: Dict, optional
        :param approx_eval_time: The approximate evaluation time for one test execution, defaults to None
        :type approx_eval_time: float, optional
        :param do_visualize: Visualize test execution or not, defaults to False
        :type do_visualize: bool, optional
        :raises ValueError: Raises ValueError if min_or_max of fitness function is not correctly configured.
        """

        super().__init__(n_var=len(xl),
                         n_obj=len(fitness_function.name),
                         xl=xl,
                         xu=xu)

        assert xl is not None
        assert xu is not None
        assert scenario_path is not None
        assert fitness_function is not None
        assert simulate_function is not None
        assert simulation_time is not None
        assert sampling_time is not None
        assert np.equal(len(xl), len(xu))
        assert np.less_equal(xl, xu).all()
        assert len(fitness_function.min_or_max) == len(fitness_function.name)

        self.fitness_function = fitness_function
        self.simulate_function = simulate_function
        self.critical_function = critical_function
        self.simulation_time = simulation_time
        self.sampling_time = sampling_time
        self.simulation_variables = simulation_variables
        self.do_visualize = do_visualize
  
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

        if other_parameters is not None:
            self.other_parameters = other_parameters

        if approx_eval_time is not None:
            self.approx_eval_time = approx_eval_time

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
        try:
            # it is assumed that the i-th simout in the simout list correponds to i-th scenario in the list x
            simout_list = self.simulate_function(x, 
                                                 self.simulation_variables, 
                                                 self.scenario_path, 
                                                 sim_time=self.simulation_time,
                                                 time_step=self.sampling_time, 
                                                 do_visualize=self.do_visualize)
        except Exception as e:
            log.info("Exception during simulation ocurred: ")
            # TODO handle exception, terminate, so that results are stored
            raise e
        out["SO"] = []
        vector_list = []
        label_list = []

        for simout in simout_list:
            out["SO"].append(simout)
            vector_fitness = np.asarray(
                self.signs) * np.array(self.fitness_function.eval(simout, **kwargs))
            vector_list.append(np.array(vector_fitness))
            label_list.append(self.critical_function.eval(vector_fitness, simout = simout))

        out["F"] = np.vstack(vector_list)
        out["CB"] = label_list

    def is_simulation(self):
        return True
