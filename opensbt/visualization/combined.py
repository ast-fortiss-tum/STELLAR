import traceback
import numpy as np
import os
import csv
from matplotlib import pyplot as plt
import pandas as pd
from opensbt.model_ga.individual import Individual
from matplotlib.ticker import (AutoMinorLocator)
from opensbt.model_ga.population import Population
from opensbt.utils.sorting import get_nondominated_population
from opensbt.visualization.visualizer import *
from opensbt.analysis.quality_indicators.quality import EvaluationResult
from matplotlib import pyplot as plt
import scipy
import matplotlib
from opensbt.utils.duplicates import duplicate_free
import logging as log
from opensbt.analysis.statistics import wilcoxon
from opensbt.analysis.quality_indicators.metrics import ncrit

from opensbt.config import BACKUP_FOLDER, N_CELLS

"""This module provides functions for the analysis of completed runs with different search approaches.
"""
def calculate_combined_crit_pop(run_paths):
    """Unions critical solutions from all runs to approximate "real" critical design space.
    """
    len(f"run_paths: {run_paths}")
    crit_pop = Population()
    for run_path in run_paths:
        crit_run = read_pf_single(
            run_path + os.sep + "all_critical_testcases.csv")
        crit_pop = Population.merge(crit_pop, crit_run)
    # assign critical label for the visualization
    for i in range(0, len(crit_pop)):
        crit_pop[i].set("CB", True)
    return crit_pop

def calculate_combined_pf(run_paths, critical_only=False):
    # TODO add information on the deviation of the values wrt. to differen runs in the plots

    pf_pop = Population()
    for run_path in run_paths:
        pf_run = read_pf_single(run_path + os.sep + "optimal_testcases.csv")
        pf_pop = Population.merge(pf_pop, pf_run)
    # log.info(f"len: {len(pf_pop)}")
    pf_pop = get_nondominated_population(pf_pop)
    if critical_only:
        crit_inds = np.where(pf_pop.get("CB"))[0]
        pf_pop = pf_pop[crit_inds]
    inds = [ind.get("F").tolist() for ind in pf_pop]
    pf = np.array(inds, dtype=float)
    # log.info(f"pf: {pf}")
    return pf, pf_pop


def plot_combined_analysis_last_min_max(metric_name, run_paths_array, save_folder):
    """
        Outputs mean/std/min/max for final metric value instead for several number of evaluations as in plot_combined_analysis
    """
    plot_array = []

    for key, (algo, run_paths) in enumerate(run_paths_array.items()):
        values_all = []

        y_stacked = np.zeros((len(run_paths), 1))
        f = plt.figure(figsize=(7, 5))

        for key_run, run_path in enumerate(run_paths):
            eval_res = EvaluationResult.load(
                run_path + os.sep + BACKUP_FOLDER, metric_name)
            n_evals, hv = eval_res.steps, eval_res.values
            values_all.append(hv[-1])
            plt.plot(n_evals, hv, marker='.', linestyle='--',
                     label='run ' + str(key_run + 1))

        max_value = max(values_all)
        min_value = min(values_all)

        max_value_runs = [max_value]
        min_value_runs = [min_value]

        def std_dev(y_values):
            y_mean = np.sum(y_values, axis=0) / \
                np.count_nonzero(y_values, axis=0)
            y_error = []
            for i in range(len(y_mean)):
                square_sum = 0
                n = 0
                for j in range(len(run_paths)):
                    if np.all(y_values[j, i]):
                        n += 1
                        deviation = y_values[j, i] - y_mean[i]
                        square_sum += deviation ** 2
                if n <= 1:
                    variance = 0
                else:
                    variance = square_sum / (n - 1)
                standart_deviation = np.sqrt(variance)
                y_error.append(standart_deviation / np.sqrt(n))
            return y_error, y_mean

        y_stacked[:, 0] = values_all
        y_error, y_mean = std_dev(y_stacked)

        x_plot = n_evals  # TODO make labels RS and NSGA-II
        plt.plot(x_plot, y_mean[0:len(x_plot)],
                 color='black', marker='o', lw=2, label='combined')
        plt.errorbar(x_plot, y_mean[0:len(x_plot)],
                     y_error[0:len(x_plot)], fmt='.k', capsize=5)
        plt.legend(loc='best')

        Path(save_folder).mkdir(parents=True, exist_ok=True)

        plt.title(f"{metric_name.upper()} Analysis ({algo})")
        plt.xlabel("Evaluations")
        plt.ylabel(f"{metric_name.upper()}")
        plt.savefig(
            save_folder + f'{metric_name}_combined_single' + str(algo) + '.png')
        plt.clf()
        plt.close(f)
        plot_array.append([x_plot, y_mean[0:len(x_plot)], y_error[0:len(
            x_plot)], min_value_runs, max_value_runs])
    return plot_array

def write_last_metric_values(metric_name_load, run_paths_array, save_folder, metric_name_label=None):
    values_algo = {}
    algos = list(run_paths_array.keys())
    for key, (algo, run_paths) in enumerate(run_paths_array.items()):
        values_algo[algo] = []
        for run_path in run_paths:
            eval_res = EvaluationResult.load(
                run_path + os.sep + BACKUP_FOLDER, metric_name_load)
            _, v = eval_res.steps, eval_res.values
            values_algo[algo].append(v[-1])

    Path(save_folder).mkdir(parents=True, exist_ok=True)

    with open(save_folder + f'overview_{metric_name_label}.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)
        header = ['run']
        for algo in algos:
            header.append(algo)
                          
        write_to.writerow(header)

        for i in range(0, len(values_algo[algo])):
            line =  [f'{i+1}']
            for algo in algos:
                line.append(values_algo[algo][i])
            write_to.writerow(line)
        
        f.close()
            
def plot_combined_analysis(metric_name_load, run_paths_array, save_folder, n_func_evals_lim, n_fitting_points, metric_name_label=None, step_chkp=None, error_mean=False):
    plot_array = []

    for key, (algo, run_paths) in enumerate(run_paths_array.items()):
        num_evals_limit = []

        for run_path in run_paths:
            eval_res = EvaluationResult.load(
                run_path + os.sep + BACKUP_FOLDER, metric_name_load)
            n_evals, hv = eval_res.steps, eval_res.values
            num_evals_limit.append(n_evals[-1])
        min_num_evals = n_func_evals_lim + 1
        step = min_num_evals // n_fitting_points

        if step_chkp is not None:
            step = step_chkp

        x = np.arange(step, min_num_evals, step=step)
        y_stacked = np.zeros((len(run_paths), len(x)))

        f = plt.figure(figsize=(7, 5))
        for key_run, run_path in enumerate(run_paths):
            eval_res = EvaluationResult.load(
                run_path + os.sep + BACKUP_FOLDER, metric_name_load)
            n_evals, hv = eval_res.steps, eval_res.values
            spl = scipy.interpolate.interp1d(
                np.array(n_evals), np.array(hv), fill_value="extrapolate")
            x_run = np.arange(step, min_num_evals, step)
            y = spl(x_run)
            y_stacked[key_run, 0:len(y)] = y
 
            plt.plot(n_evals, hv, marker='.', linestyle='--',
                     label='run ' + str(key_run + 1))

        y_mean = np.sum(y_stacked, axis=0) / \
            np.count_nonzero(y_stacked, axis=0)
        y_error = []
        for i in range(len(y_mean)):
            square_sum = 0
            n = 0
            for j in range(len(run_paths)):
                if np.all(y_stacked[j, i]):
                    n += 1
                    deviation = y_stacked[j, i] - y_mean[i]
                    square_sum += deviation ** 2
            if n <= 1:
                variance = 0
            else:
                variance = square_sum / (n - 1)
            standard_deviation = np.sqrt(variance)
            if error_mean:
                y_error.append(standard_deviation / np.sqrt(n))
            else:
                y_error.append(standard_deviation)

        x_plot = np.arange(step, min_num_evals, step=step)
        plt.plot(x_plot, y_mean[0:len(x_plot)],
                 color='black', marker='o', lw=2, label='combined')
        plt.errorbar(x_plot, y_mean[0:len(x_plot)],
                     y_error[0:len(x_plot)], fmt='.k', capsize=5)
        plt.legend(loc='best')

        Path(save_folder).mkdir(parents=True, exist_ok=True)

        if metric_name_label is None:
            metric_name_label = metric_name_load.upper()

        plt.title(f"{metric_name_label} Analysis ({algo})")
        plt.xlabel("Evaluations")
        plt.ylabel(f"{metric_name_label}")
        plt.savefig(
            save_folder + f'{metric_name_label}_combined_' + str(algo) + '.png')
        plt.clf()
        plt.close(f)
        plot_array.append(
            [x_plot, y_mean[0:len(x_plot)], y_error[0:len(x_plot)]])

    return plot_array

def plot_combined_hypervolume_lin_analysis(run_paths_array, save_folder):
    for key, (algo, run_paths) in enumerate(run_paths_array.items()):
        if len(run_paths) == 0:
            log.info("Path list is empty")
            return

        f = plt.figure(figsize=(7, 5))
        plt.title(f"Performance Analysis ({algo})")
        plt.xlabel("Evaluations")
        plt.ylabel("Hypervolume")

        n_runs = len(run_paths)
        hv_run = []
        evals_run = []

        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, n_runs)]

        for ind, run_path in enumerate(run_paths):
            eval_res = EvaluationResult.load(
                run_path + os.sep + BACKUP_FOLDER, "hv")
            n_evals, hv = eval_res.steps, eval_res.values
            plt.plot(n_evals, hv, marker='o', linestyle=":", linewidth=0.5, markersize=3, color=colors[ind],
                     label=f"run: " + str(ind + 1))
            hv_run.append(hv)
            evals_run.append(n_evals)

        def get_interpol_value(pos, n_evals, hv):
            for ind, eval in enumerate(n_evals):
                if n_evals[ind] > pos:
                    if ind == 0:
                        value = (hv[ind]) / 2
                    else:
                        diff = pos - n_evals[ind - 1]
                        grad = (hv[ind] - hv[ind - 1]) / \
                            (n_evals[ind] - n_evals[ind - 1])
                        value = hv[ind - 1] + grad * diff
                    return value

        step = 10
        last_n_evals = [n_evals[-1] for n_evals in evals_run]
        # n_steps = floor(min(last_n_evals)/step)
        last_n_eval = min(last_n_evals)
        n_evals_comb = np.arange(0, last_n_eval, step)

        for ind in range(0, n_runs):
            n_evals = evals_run[ind]
            hv = hv_run[ind]
            hv_run[ind] = [get_interpol_value(
                val, n_evals, hv) for val in n_evals_comb]

        hv_comb_all = np.sum(hv_run, axis=0)
        hv_comb = np.asarray(hv_comb_all) / n_runs

        plt.plot(n_evals_comb, hv_comb, marker='o', linewidth=1,
                 markersize=3, color='black', label=f"combined")

        plt.legend(loc='best')
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(save_folder + 'hypervolume_combined_' + str(algo) + '.png')
        plt.clf()
        plt.close(f)


def write_analysis_results(result_runs_all, 
                            save_folder, 
                            nadir, 
                            ideal):
    evaluations_all = {}
    evaluations_all_no_duplicates = {}

    critical_all = {}
    critical_all_no_duplicates = {}

    n_critical_all = {}
    n_critical_all_no_duplicates = {}

    mean_critical_all = {}
    mean_critical_all_no_duplicates = {}

    mean_evaluations_all = {}
    mean_evaluations_all_no_duplicates = {}

    ratio_critical_single_run = {}
    ratio_critical_single_run_no_duplicates = {}
    # temporary
    # algo1 = list(result_runs_all.keys())[0]
    # algo2 = list(result_runs_all.keys())[1]

    n_crit_no_dup_best = {}
    n_dominated_best_crit = {}

    grid_crit_distinct = {}
    n_crit_distinct_avg = {}
    n_crit_distinct_all = {}

    grid_crit_distinct_X = {}
    n_crit_distinct_avg_X = {}
    n_crit_distinct_all_X = {}
    
    base_algo = None
    
    for algo, result_runs in result_runs_all.items():
        if len(result_runs) == 0:
            log.info("Result list is empty")
            return
        n_runs = len(result_runs)

        # n evaluations analysis
        n_evals = [res.algorithm.evaluator.n_eval for res in result_runs]
        min_n_evals = np.min(n_evals)
        max_n_evals = np.max(n_evals)
        mean_n_evals = np.sum(n_evals) / len(result_runs)

        # time analysis
        exec_time = [res.exec_time for res in result_runs]
        min_exec_time = np.min(exec_time)
        max_exec_time = np.max(exec_time)
        mean_exec_time = np.sum(exec_time) / len(result_runs)

        # criticality analysis
        evaluations_all[algo] = [res.obtain_all_population()
                                 for res in result_runs]
        evaluations_all_no_duplicates[algo] = [duplicate_free(
            res.obtain_all_population()) for res in result_runs]
        
        critical_all[algo] =  [evals.divide_critical_non_critical()[0] for evals in evaluations_all[algo]]
        
        critical_all_no_duplicates[algo] = [evals.divide_critical_non_critical()[0] for evals in evaluations_all_no_duplicates[algo]]

        n_critical_all[algo] = np.asarray(
            [len(evals) for evals in   critical_all[algo]], dtype=object)
        n_critical_all_no_duplicates[algo] = np.asarray(
            [len(evals) for evals in critical_all_no_duplicates[algo]], dtype=object)

        # mean_critical_all[algo] = np.sum(
        #     n_critical_all[algo])  # /len(result_runs)
        # mean_critical_all_no_duplicates[algo] = np.sum(
        #     n_critical_all_no_duplicates[algo])  # /len(result_runs)

        # mean_evaluations_all[algo] = np.sum(
        #     len(evals) for evals in evaluations_all[algo])  # /len(result_runs)
        # mean_evaluations_all_no_duplicates[algo] = np.sum(
        #     len(evals) for evals in evaluations_all_no_duplicates[algo])  # /len(result_runs)

        # ratio_critical_single_run[algo] = [
        #     n_critical_all[algo][i] / len(evaluations_all[algo][i]) for i in range(0, n_runs)]
        # ratio_critical_single_run_no_duplicates[algo] = [
        #     n_critical_all_no_duplicates[algo][i] / len(evaluations_all_no_duplicates[algo][i]) for i in range(0, n_runs)]

        n_crit_no_dup_best[algo] = len(get_nondominated_population(
                evaluations_all_no_duplicates[algo][-1].divide_critical_non_critical()[0])
                )
        
        # criticality analysis using grid-based distinctness in fitness space

        grid_crit_distinct[algo] = [  ncrit.get_n_crit_grid(evals.get("F"), 
                                            b_min=ideal,
                                            b_max=nadir,
                                            n_cells=N_CELLS)[1]
                                        for evals in critical_all_no_duplicates[algo]]

        n_crit_distinct_avg[algo] = sum(
                                    np.asarray(
                                        [
                                            np.count_nonzero(grid) for grid in grid_crit_distinct[algo]
                                        ]
                                        )
                                    ) / n_runs

        # conjunction over all grids
        n_crit_distinct_all[algo] = np.count_nonzero(np.logical_or.reduce(grid_crit_distinct[algo]))

        
        # # criticality analysis using grid-based distinctness in input space
        grid_crit_distinct_X[algo] = [  ncrit.get_n_crit_grid(evals.get("X"), 
                                            b_min=result_runs[-1].problem.xl,
                                            b_max=result_runs[-1].problem.xu,
                                            n_cells=N_CELLS)[1]
                                        for evals in critical_all_no_duplicates[algo]]

        n_crit_distinct_avg_X[algo] = sum(
                                    np.asarray(
                                        [
                                            np.count_nonzero(grid) for grid in grid_crit_distinct_X[algo]
                                        ]
                                        )
                                    ) / n_runs

        # conjunction over all grids
        n_crit_distinct_all_X[algo] = np.count_nonzero(np.logical_or.reduce(grid_crit_distinct_X[algo]))

        # grid_crit_distinct_X[algo] = [  ncrit.get_n_crit_grid(evals.get("X"), 
        #                                     b_min=result_runs[-1].problem.xl,
        #                                     b_max=result_runs[-1].problem.xu,
        #                                     n_cells=N_CELLS)[1]
        #                                 for evals in critical_all_no_duplicates[algo]]


        if base_algo is None:
            base_algo = algo

        '''Output of summery of the performance'''
        with open(save_folder + f'analysis_results_{algo}.csv', 'w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)

            header = ['Attribute', 'Value']
            write_to.writerow(header)
            write_to.writerow(['min_n_evals', min_n_evals])
            write_to.writerow(['max_n_evals', max_n_evals])
            write_to.writerow(['mean_n_evals', mean_n_evals])

            write_to.writerow(['min_exec_time [s]', min_exec_time])
            write_to.writerow(['max_exec_time [s]', max_exec_time])
            write_to.writerow(['mean_exec_time [s]', mean_exec_time])


            write_to.writerow([f'num_critical_distinct_sum_F ({algo})',   n_crit_distinct_all[algo]])
            write_to.writerow([f'num_critical_distinct_avg_F ({algo})',    n_crit_distinct_avg[algo]])

            write_to.writerow([f'num_critical_distinct_sum_X ({algo})',   n_crit_distinct_all_X[algo]])
            write_to.writerow([f'num_critical_distinct_avg_X ({algo})',    n_crit_distinct_avg_X[algo]])
            # write_to.writerow([f'num_opt_critical_distinct_sum_F ({algo})',  int(sum(n_crit_distinct_opt[algo]))])
            # write_to.writerow([f'num_opt_critical_distinct_avg_F ({algo})',  int(sum(n_crit_distinct_opt[algo])/n_runs)])

            # write_to.writerow([f'num_critical_distinct_sum_X ({algo})',  sum(n_crit_distinct_X[algo])])
            # write_to.writerow([f'num_critical_distinct_avg_X ({algo})',  int(sum(n_crit_distinct_X[algo])/n_runs)])
            # write_to.writerow([f'num_opt_critical_distinct_sum_X ({algo})',  int(sum(n_crit_distinct_opt_X[algo]))])
            # write_to.writerow([f'num_opt_critical_distinct_avg_X ({algo})',  int(sum(n_crit_distinct_opt_X[algo])/n_runs)])
            # write_to.writerow(['Mean number critical Scenarios', len(mean_critical)])
            # write_to.writerow(['Mean evaluations all scenarios', len(mean_evaluations_all)])
            # write_to.writerow(['Mean ratio critical/all scenarios', '{0:.2f}'.format(len(critical_all) / len(all_population))])
            # write_to.writerow(['Mean ratio best Critical/best Scenarios', '{0:.2f}'.format(len(critical_best) / len(best_population))])
            f.close()

        # ratio_critical_both = np.sum(n_critical_all[algo]) if np.sum(n_critical_all[algo]) == 0 \
        #     else np.sum(n_critical_all[algo]) / np.sum(n_critical_all[algo])

        # ratio_critical_both_no_duplicates = np.sum(n_critical_all_no_duplicates[algo]) if np.sum(n_critical_all_no_duplicates[algo]) == 0 \
        #     else np.sum(n_critical_all_no_duplicates[algo]) / np.sum(n_critical_all_no_duplicates[algo])

        # ratio_critical_both_average = np.sum(
        #     ratio_critical_single_run[algo]) / np.sum(ratio_critical_single_run[algo])
        # ratio_critical_both_average_no_duplicates = np.sum(
        #     ratio_critical_single_run_no_duplicates[algo]) / np.sum(ratio_critical_single_run_no_duplicates[algo])

    '''Output of summery of the performance'''
    with open(save_folder + f'analysis_combined.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)
        
        algo_names = list(result_runs_all.keys()) 

        for algo in algo_names:
            write_to.writerow([f'Critical Scenarios {algo}', np.sum(n_critical_all[algo])])
            write_to.writerow([f'Critical Scenarios {algo} (duplicate free)', np.sum(
                n_critical_all_no_duplicates[algo])])
            
            write_to.writerow([f'Non-dominated Critical Scenarios {algo} (duplicate free)',n_crit_no_dup_best[algo]])
        # write_to.writerow([f'Best Critical Scenarios {algo} Dominated By {base_algo} (duplicate free)',n_dominated_best_crit[algo]])

        # write_to.writerow(
        #     [f'Critical Scenarios {algo}', np.sum(n_critical_all[algo])])
        # write_to.writerow([f'Critical Scenarios {algo} (duplicate free)', np.sum(
        #     n_critical_all_no_duplicates[algo])])

        # write_to.writerow(
        #     [f'Ratio Critical Scenarios {algo}/{algo} (union)', '{0:.2f}'.format(ratio_critical_both)])
        # write_to.writerow(
        #     [f'Ratio Critical Scenarios {algo}/{algo} (union, duplicate free)', '{0:.2f}'.format(ratio_critical_both_no_duplicates)])

        # write_to.writerow(
        #     [f'Ratio Critical Scenarios {algo}/{algo} (average)', '{0:.2f}'.format(ratio_critical_both_average)])
        # write_to.writerow([f'Ratio Critical Scenarios {algo}/{algo} (average, duplicate free)',
        #                 '{0:.2f}'.format(ratio_critical_both_average_no_duplicates)])

        # write_to.writerow([f'Mean evaluations all scenarios {algo}', mean_evaluations_all[algo]])
        # write_to.writerow([f'Mean evaluations all scenarios {algo2}', mean_evaluations_all[algo2]])
        # write_to.writerow([f'Mean critical all {algo}', '{0:.2f}'.format(mean_critical_all[algo])])
        # write_to.writerow([f'Mean critical all {algo2}', '{0:.2f}'.format(mean_critical_all[algo2])])
        # write_to.writerow(['Mean ratio best Critical/best Scenarios', '{0:.2f}'.format(len(critical_best) / len(best_population))])

        f.close()


def read_metric_single(filename, metric_name):
    table = pd.read_csv(filename, names=["n_eval", metric_name])
    n_evals = np.array(table["n_eval"][1:].values.tolist(), dtype=float)
    hv = np.array(table[metric_name][1:].values.tolist(), dtype=float)
    return n_evals, hv

def read_pf_single(filename):
    individuals = []
    table = pd.read_csv(filename)
    n_var = -1
    k = 0
    # identify number of objectives
    for col in table.columns[1:]:
        if col.startswith("Fitness_"):
            n_var = k
            break
        k = k + 1
    for i in range(len(table)):
        X = table.iloc[i, 1:n_var + 1].to_numpy()
        F = table.iloc[i, n_var + 1:-1].to_numpy()
        CB = table.iloc[i, -1]
        ind = Individual()
        ind.set("X", X)
        ind.set("F", F)
        ind.set("CB", CB)
        individuals.append(ind)
    return Population(individuals=individuals)

def make_comparison_single(max_evaluations, save_folder, subplot_metrics, subplot_names, algo_names, suffix=""):
    font = {'family': 'sans-serif', 'size': 17, 'weight': 'bold'}
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.weight'] = "bold"
    matplotlib.rcParams['font.size'] = 35

    offset_x_ax = 0.1 * max_evaluations
    stop = max_evaluations + offset_x_ax
    step = 200

    algos = algo_names
    colors = ['#9a226a', '#1347ac']
    n_subplots = len(subplot_metrics)
    fig, ax = plt.subplots(n_subplots, 1, figsize=(9, 9))
    for i in range(len(algos)):
        for key, metric in enumerate(subplot_metrics):
            # plt.subplot(n_subplots, 1, key + 1)
            ax[key].plot(metric[i][0], metric[i][1],
                         color=colors[i], marker=None, lw=1)
            ax[key].errorbar(metric[i][0], metric[i][1], metric[i]
                             [2], ecolor=colors[i], fmt='.k', capsize=5)
            ax[key].set_xticks(np.arange(0, stop + step, step))
            ax[key].set_xlim([0, stop])
            ax[key].locator_params(tight=True, axis='y', nbins=4)
            if key != 2:
                ax[key].xaxis.set_ticklabels([])

    for i in range(n_subplots):
        ax[i].set_ylabel(subplot_names[i], **font)
        # ax[i].xaxis.set_minor_locator()
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].tick_params(which='major', axis='y', length=7)
        ax[i].tick_params(which='minor', axis='y', length=4)
        ax[i].tick_params(which='minor', axis='x', length=0)
    ax[n_subplots - 1].set_xlabel("Number of Evaluations", **font)
    ax[n_subplots - 1].legend(algos, loc='best')

    ax[n_subplots - 1].set_xticks(np.arange(0, stop+step, step))
    ax[n_subplots - 1].set_xlim([0, stop])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.savefig(save_folder + f'subplots_combined{suffix}.png')
    plt.clf()
    plt.close(fig)

def make_comparison_plot(max_evaluations, 
                        save_folder, 
                        subplot_metrics, 
                        subplot_names, 
                        algo_names, 
                        distance_tick,
                        shift_error=False, 
                        suffix="",
                        colors=None,
                        cmap=None,
                        figsize=(7,5),
                        linestyles=None,
                        alpha=None):

    font = {'family': 'sans-serif', 'size': 16, 'weight': 'bold'}
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.weight'] = "bold"
    matplotlib.rcParams['font.size'] = 15
    size_title = 12
    size_ticks = 10

    if shift_error is True:
        offset_algo_x = 0.010 * max_evaluations
        offset_algo_y = 0.012
    else: 
        offset_algo_x = 0
        offset_algo_y = 0
    
    marker_error = ''
    line_width = 2
    error_lw = 2
    error_ls = ''

    offset_x_ax = 0.1 * max_evaluations
    stop = max_evaluations + offset_x_ax
    if distance_tick is None:
        step = 0.1*max_evaluations
    else:
        step = distance_tick

    algos = algo_names
    if colors is None:
            colors = ['#9a226a',
                    '#1347ac',
                    '#ffbb00',
                    '#a0052d',
                    '#666666']
    n_subplots = len(subplot_metrics)

    # only one metric to plot
    if n_subplots == 1:
        fig, ax = plt.subplots(n_subplots, figsize=figsize)
        for i in range(len(algos)):
            for key, metric in enumerate(subplot_metrics):
                ax.plot(metric[i][0], 
                        metric[i][1],
                        color= cmap[algos[i]] if cmap is not None else colors[i], 
                        marker = None,
                        linestyle=linestyles[i] if linestyles is not None else None,
                        lw=1)
                ax.errorbar(np.asarray(metric[i][0]) + offset_algo_x*i, 
                           np.asarray(metric[i][1]) + offset_algo_y*i, 
                            metric[i][2], 
                            ecolor= cmap[algos[i]] if cmap is not None else colors[i], 
                            marker=marker_error,
                            capsize=5,
                            linestyle=error_ls,
                            linewidth=error_lw)
                ax.set_xticks(np.arange(0, max_evaluations, step))
                ax.set_xlim([0, stop])
                ax.locator_params(tight=True, axis='y', nbins=4)
        
        ax.set_ylabel(subplot_names[0], **font)
        # ax[i].xaxis.set_minor_locator()
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='major', axis='y', length=7, labelsize=size_ticks)
        ax.tick_params(which='minor', axis='y', length=4, labelsize=size_ticks)
        ax.tick_params(which='minor', axis='x', length=0, labelsize=size_ticks)

        ax.set_xlabel("Number of Evaluations", **font)
        ax.title.set_size(size_title)
        ax.legend(algos, loc='best',alpha=0.2)
        ax.set_xticks(np.arange(0, stop, step))
        ax.set_xlim([0, stop])
    else:
        # several metrics to plot
        fig, ax = plt.subplots(n_subplots, 1, figsize=figsize)
        for i in range(len(algos)):
            for key, metric in enumerate(subplot_metrics):
                # plt.subplot(n_subplots, 1, key + 1)
                # log.info(f"metric: {metric}")
                ax[key].plot(np.asarray(metric[i][0]), 
                             np.asarray(metric[i][1]),
                             color= cmap[algos[i]] if cmap is not None else colors[i],
                             marker = None,
                             linestyle=linestyles[i] if linestyles is not None else None,
                             lw=line_width)
                ax[key].errorbar(
                                 np.asarray(metric[i][0])+ offset_algo_x*i, 
                                 np.asarray(metric[i][1]) + offset_algo_y*i, 
                                 metric[i][2], 
                                 marker=marker_error,
                                 ecolor= cmap[algos[i]] if cmap is not None else colors[i], 
                                 capsize=5,
                                 linestyle=error_ls,
                                 linewidth=error_lw)
                ax[key].set_xticks(np.arange(0,stop+step -1, step))
                ax[key].set_xlim([0, stop])
                ax[key].locator_params(tight=True, axis='y', nbins=4)

                if key != 2:
                    ax[key].xaxis.set_ticklabels([])

        for i in range(n_subplots):
            ax[i].set_ylabel(subplot_names[i], **font)
            # ax[i].xaxis.set_minor_locator()
            ax[i].xaxis.set_minor_locator(AutoMinorLocator())
            ax[i].tick_params(which='major', axis='y', length=7)
            ax[i].tick_params(which='minor', axis='y', length=4)
            ax[i].tick_params(which='minor', axis='x', length=0)

        ax[n_subplots - 1].set_xlabel("Number of Evaluations", **font)
        ax[n_subplots - 1].legend(algos, loc='best',  framealpha=alpha)

        ax[n_subplots - 1].set_xticks(np.arange(0, stop, step))
        ax[n_subplots - 1].set_xlim([0, stop])

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    Path(save_folder).mkdir(parents=True, exist_ok=True)

    plt.savefig(save_folder + f'subplots_combined{suffix}.png')
    plt.savefig(save_folder + f'subplots_combined{suffix}.pdf', format="pdf")

    plt.clf()
    plt.close(fig)
    
def make_subplots(max_evaluations, save_folder, subplot_metrics, subplot_names, algo_names, distance_tick, suffix=""):
    font = {'family': 'sans-serif', 'size': 17, 'weight': 'bold'}
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.weight'] = "bold"
    matplotlib.rcParams['font.size'] = 35

    offset_x_ax = 0.1 * max_evaluations
    stop = max_evaluations + offset_x_ax
    if distance_tick is not None:
        step = distance_tick
    else:
        step = 0.1 * max_evaluations

    algos = algo_names
    colors = ['#9a226a', '#1347ac']
    n_subplots = len(subplot_metrics)
    fig, ax = plt.subplots(n_subplots, 1, figsize=(9, 9))
    for i in range(len(algos)):
        for key, metric in enumerate(subplot_metrics):
            # plt.subplot(n_subplots, 1, key + 1)
            ax[key].plot(metric[i][0], metric[i][1],
                         color=colors[i], marker=None, lw=1)
            ax[key].errorbar(metric[i][0], metric[i][1], metric[i]
                             [2], ecolor=colors[i], fmt='.k', capsize=5)
            ax[key].set_xticks(np.arange(0, stop + step, step))
            ax[key].set_xlim([0, stop])
            ax[key].locator_params(tight=True, axis='y', nbins=4)
            if key != 2:
                ax[key].xaxis.set_ticklabels([])

    for i in range(n_subplots):
        ax[i].set_ylabel(subplot_names[i], **font)
        # ax[i].xaxis.set_minor_locator()
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].tick_params(which='major', axis='y', length=7)
        ax[i].tick_params(which='minor', axis='y', length=4)
        ax[i].tick_params(which='minor', axis='x', length=0)
    ax[n_subplots - 1].set_xlabel("Number of Evaluations", **font)
    ax[n_subplots - 1].legend(algos, loc='best')

    ax[n_subplots - 1].set_xticks(np.arange(0, stop+step, step))
    ax[n_subplots - 1].set_xlim([0, stop])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.savefig(save_folder + f'subplots_combined{suffix}.png')
    plt.clf()
    plt.close(fig)


''' 
    Write plot idge plot data in csv form. (TODO make more generic, i.e. independent of number of metrics)
'''
def write_metric_data_to_csv_cid(save_folder, metric_name, algo_names, plot_array, suffix=""):
    algo_1 = algo_names[0]
    algo_2 = algo_names[1]

    m = metric_name

    header_igd = [f'{algo_1}_n_evals',
                  f'{algo_1}_{m}',
                  f'{algo_1}_{m}_sigma',
                  # f'{algo_1}__{m}_min',
                  # f'{algo_1}__{m}_max',
                  f'{algo_2}__n_evals',
                  f'{algo_2}__{m}',
                  f'{algo_2}__{m}_sigma',
                  # f'{algo_2}__{m}_min',
                  # f'{algo_2}__{m}_max'
                  ]
    # metric_names = ['hv', 'gd', 'sp']
    paths = []
    metric = plot_array
    filename = save_folder + metric_name + os.sep + \
        'combined_' + metric_name + suffix + '.csv'
    with open(filename, 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)
        write_to.writerow(header_igd)
        for i in range(len(metric[0][0])):
            write_to.writerow([metric[0][0][i],  # algo1
                               metric[0][1][i],
                               metric[0][2][i],
                               # metric[0][3][i],
                               # metric[0][4][i],
                               metric[1][0][i],  # algo2
                               metric[1][1][i],
                               metric[1][2][i],
                               # metric[1][3][i],
                               # metric[1][4][i]
                               ])
        f.close()
        paths.append(filename)

    return paths


def write_metric_data_to_csv(save_folder, metric_names, algo_names, plot_array_hv, plot_array_igd, plot_array_sp, suffix=""):
    headers = []
    for m in metric_names:
        header = []
        for algo in algo_names:
            header += [f'{algo}_n_evals', f'{algo}_{m}', f'{algo}_{m}_sigma']
        headers.append(header)
    paths = []

    for key, metric in enumerate([plot_array_hv, plot_array_igd, plot_array_sp]):
        filename = save_folder + \
            metric_names[key] + os.sep + 'combined_' + \
            metric_names[key] + suffix + '.csv'
        with open(filename, 'w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)
            write_to.writerow(headers[key])
            for i, value in enumerate(metric[0][0]):
                res_algo = []
                for j, _ in enumerate(algo_names):
                    res_algo += [metric[j][0][i],
                                 metric[j][1][i], metric[j][2][i]]
                write_to.writerow(res_algo)
            f.close()
            paths.append(filename)
    return paths


def retrieve_metric_data_from_csv(paths, n_algos):
    storing_arrays = []
    for key, path in enumerate(paths):
        table = pd.read_csv(path)
        by_metric = []
        for i in range(n_algos):
            metric_algo = [
                        table.iloc[:, 3*i].values.tolist(), 
                        table.iloc[:,3*i + 1].values.tolist(), 
                        table.iloc[:, 3*i + 2].values.tolist()
            ]
            by_metric.append(metric_algo)
        storing_arrays.append(by_metric)
    return storing_arrays  # plot_array_hv, plot_array_igd, plot_array_sp

def statistical_analysis(metric_name_load, 
                        runs_bases,
                        runs_test,
                        algo_test,
                        save_folder, 
                        metric_name_label=None):
    def get_last_metric_value(path,name):
        eval_res = EvaluationResult.load(path + os.sep + BACKUP_FOLDER, name)
        _, hv = eval_res.steps, eval_res.values
        return hv[-1]

    m_test = []
    m_bases = {}

    for run_path in runs_test:
        m_test.append(get_last_metric_value(run_path,metric_name_load))
    log.info(f"metric values test: {m_test}")
    algos = runs_bases.keys()

    for algo in algos:
        # We allow to have the algorithm under test in the list
        if algo == algo_test:
            continue
        m_bases[algo] = []  
        results = {}
        for run_path in runs_bases[algo]:
            m_bases[algo].append(get_last_metric_value(run_path,metric_name_load))
        
        assert(len(m_bases[algo]) == len(m_test))
        for i,algo in enumerate(m_bases):
            log.info(f"metric values base_{i}: {m_bases[algo]}")

        p_val, effect =  wilcoxon.run_wilcoxon_and_delaney(m_test,m_bases[algo])
        results[algo] = [p_val, effect] 

    log.info(m_bases)
   
    # write to file
    Path(save_folder).mkdir(parents=True, exist_ok=True)   
    
    with open(save_folder + f'{metric_name_label}_significance.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)
        header = [f'algo (subject: {algo_test})', 'p_value', 'effect']
        write_to.writerow(header)
        for algo in algos:
            if algo == algo_test:
                continue
            write_to.writerow([f'{algo}', results[algo][0],   results[algo][1]])
        f.close()

def statistical_analysis_from_overview(metric_name, 
                        input_folder,
                        save_folder):
    try:
        file_name = input_folder + f"overview_{metric_name}.csv"
        table = pd.read_csv(file_name)

        m_test = []
        m_bases = {}
        results = {}

        algos =list(table.columns.values)[1:]
        log.info(algos)

        algo = algos[0]
        m_test = table[algos[0]].to_list() #table.iloc[:, 1].to_numpy()
        for algo in algos[1:]:
            m_bases[algo] = table[algo].to_list() #table.iloc[:, i + 1].to_numpy()
            # assert(len(m_bases[col]) == len(m_test))
            d1 = m_test
            d2 = m_bases[algo]
            log.info(f"test: {d1}")
            log.info(f"base: {d2}")
            p_val, effect =  wilcoxon.run_wilcoxon_and_delaney(d1,d2)
            results[algo] = [p_val, effect] 

        log.info(m_bases)
        # write to file
        Path(save_folder).mkdir(parents=True, exist_ok=True)   

        with open(save_folder + f'{metric_name}_significance.csv', 'w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)
            header = [f'algo (subject: {algos[0]})', 'p_value', 'effect']
            write_to.writerow(header)
            for algo in algos[1:]:
                write_to.writerow(
                                [
                                    f'{algo}', 
                                    results[algo][0],   
                                    results[algo][1]
                                    ]
                                )
            f.close()

    except ValueError as e:
        # we do this to catch this exception from wilcoxon: 
        #  ValueError("zero_method 'wilcox' and 'pratt' do not "
        #                              "work if x - y is zero for all elements.")
        traceback.print_exc()
        print("No statistical analysis is possible. No files generated.")


