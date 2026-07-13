import os
from opensbt.visualization import combined
from pymoo.core.population import Population
from opensbt.visualization.configuration import *
from opensbt.utils.sorting import *
from opensbt.model_ga.problem import *
from opensbt.model_ga.result import *
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import copy

def create_result(problem, hist_holder, inner_algorithm, execution_time):
    """ Generates a result object from generation data to adhere to pymoos result model. Consider the implementation of NSGA-II-DT optimizer for reference. """
    # TODO calculate res.opt
    I = 0
    for algo in hist_holder:
        I += len(algo.pop)
        algo.evaluator.n_eval = I
        algo.start_time = 0
        algo.problem = problem
        algo.result()

    res_holder = SimulationResult()
    res_holder.algorithm = inner_algorithm
    res_holder.algorithm.evaluator.n_eval = I
    res_holder.problem = problem
    res_holder.algorithm.problem = problem
    res_holder.history = hist_holder
    res_holder.exec_time = execution_time

    # calculate total optimal population using individuals from all iterations
    opt_all = Population()
    for algo in hist_holder:
        opt_all = Population.merge(opt_all, algo.pop)
    # log.info(f"opt_all: {opt_all}")
    opt_all_nds = get_nondominated_population(opt_all)
    res_holder.opt = opt_all_nds

    return res_holder

def create_result_from_generations(path_generations, problem):
    """Generates a result object from recorded generation files which contain test inputs from the specific generations in human readable format.
    """
    
    n_generations = len(os.listdir(path_generations))

    inner_algorithm = NSGA2(
        pop_size=None,
        n_offsprings=None,
        sampling=None,
        crossover=SBX(),
        mutation=PM(),
        eliminate_duplicates=True)
    
    hist_holder = [copy.deepcopy(inner_algorithm) for i in range(n_generations)]

    for i in range(n_generations):
        path_gen = path_generations + f'gen_{i+1}.csv'
        pop_gen = combined.read_pf_single(filename=path_gen)
        hist_holder[i].pop = pop_gen
        opt_pop = Population(individuals=calc_nondominated_individuals(pop_gen))
        hist_holder[i].opt = opt_pop

    return create_result(problem=problem,
                        hist_holder=hist_holder,
                        inner_algorithm=inner_algorithm,
                        execution_time=0)