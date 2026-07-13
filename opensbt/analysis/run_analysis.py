import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.algorithm import nsga2_optimizer, nsga2dt_optimizer

from default_experiments import *
from opensbt.experiment.search_configuration import *
import argparse
import logging as log
from opensbt.analysis.Analysis import Analysis
from opensbt.utils.log_utils import *
from opensbt.config import *
from opensbt.experiment.experiment_store import experiments_store
from opensbt.config import METRIC_CONFIG

"""
    This script defines a concrete analysis problem to be executed.
"""
disable_pymoo_warnings()

if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description="Pass parameters for analysis.")
    parser.add_argument('-r', dest='n_runs', type=int, default=None, action='store',
                        help='Number runs to perform each algorithm for statistical analysis.')
    parser.add_argument('-p', dest='folder_runs',nargs="+", type=str, default=None, action='store',
                        help='The folder of the results written after executed runs of both algorithms. Path needs to end with "/".')
    parser.add_argument('-c', dest='path_metrics', type=str, default=None, action='store',
                        help='Path to csv file with metric results to regenerate comparison plot')
    
    args = parser.parse_args()
    
    ############# Set default experiment

    # We need to pass several exps, to be able to compare searches with different fitnessfnc 
    # If the problem is the same, just pass the experiment number twice
        
    ##### Dummy
    exp_numbers_default = [5,5]  # Test

    ############### Specify the algorithms

    class_algos = [  
        nsga2_optimizer.NsgaIIOptimizer,
        nsga2dt_optimizer.NsgaIIDTOptimizer
    ]

    ################ For Visualization

    algo_names = [
        "NSGA-II",
        "NSGA-II-DT"
    ]
    #############################
     
    n_runs_default = 2
    
    # this variable is required by an analysis function; 
    # the results will be capped by that number (the search time of some algorithms can be not controllable)
    # TODO refactor 
    n_func_evals_lim = 200 
    analyse_runs = n_runs_default

    distance_tick = 0.1*n_func_evals_lim
    exp_numbers = exp_numbers_default

    if args.n_runs is None:
        n_runs = n_runs_default
    else:
        n_runs = args.n_runs

    folder_runs =  args.folder_runs
    path_metrics = args.path_metrics

    ###################
    problems = []
    configs = []

    for exp_name in exp_numbers:
        exp = experiments_store.load(str(exp_name))
        problem = exp.problem
        problems.append(problem)
        configs.append(exp.search_configuration)

    log.info("Experiment loaded.")
    ##################### Override config
    DO_OVERRIDE_CONFIG = True

    if DO_OVERRIDE_CONFIG:
        config_1 = DefaultSearchConfiguration()
        config_1.population_size = 10
        config_1.n_generations = 20

        ####################
        config_2 = DefaultSearchConfiguration()
        config_2.population_size = 10
        #config_2.ref_points = np.asarray([[0,-4]])
        config_2.inner_num_gen = 5
        config_2.n_func_evals_lim = n_func_evals_lim
        configs = [config_1,config_2]

    ########## HV config #############
        
    # Math Kursawe
    # ideal = np.asarray([-20,-20])
    # ref_point_hv = np.asarray([-10,0]) 
    # nadir = ref_point_hv
        
    ideal = METRIC_CONFIG["DUMMY"]["ideal"]
    ref_point_hv =  METRIC_CONFIG["DUMMY"]["ref_point_hv"]
    nadir =  ref_point_hv

    ################ Naming
    analysis_name = None

    ############# ONLY EVALUATE #######################
    output_folder = None
    #output_folder = os.getcwd() + os.sep + "results" + os.sep + "output" + os.sep

    #######################
    folder_runs = None


    # folder_runs = [
    #     path,
    #     path,
    #     path
    # ]

    # Use different critical funciton for evaluation (only for algos that don't use crit function for search)  
    crit_function = None

    Analysis.run(
                analysis_name = analysis_name,
                algo_names = algo_names,
                class_algos = class_algos,
                configs = configs,
                n_runs = n_runs,
                problems = problems,
                n_func_evals_lim = n_func_evals_lim, 
                folder_runs = folder_runs,
                path_metrics = path_metrics,
                ref_point_hv = ref_point_hv,
                ideal = ideal,
                nadir = nadir,
                output_folder = output_folder,
                do_coverage_analysis = False,
                do_ds_analysis = True,
                path_critical_set = None,
                debug = DEBUG,
                distance_tick = distance_tick,
                do_evaluation = True,
                crit_function = crit_function,
                color_map=None
    )
