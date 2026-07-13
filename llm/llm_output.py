from dataclasses import fields
from llm.eval.diversity import cluster_utterances_vars
from llm.model.qa_problem import QAProblem
from opensbt.model_ga.individual import IndividualSimulated
from llm.model.models import LOS
from pymoo.core.result import Result
from opensbt.utils.duplicates import duplicate_free
from opensbt.model_ga.population import PopulationExtended
import csv
from itertools import product
from pydantic import BaseModel
import shutil
from llm.utils.math import euclid_distance
import logging as log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from pathlib import Path
import os
import json
import os
from typing import List, Dict, Optional, Any
from llm.model.models import Utterance, ContentInput
from llm.model.qa_problem import QAProblem
from llm.eval.utterances_distance import UtterancesDistance

def compute_pairwise_dissimilarity(utterances):
    q_dissimilarities = []
    a_dissimilarities = []
    vars_dissimilarities = []

    for i in range(len(utterances)):
        for j in range(i + 1, len(utterances)):
            distance = UtterancesDistance.calculate(
                utterances[i].get("X")[0],
                utterances[j].get("X")[0]
            )
            q_dissimilarities.append(distance.embeddings_distance)
            a_dissimilarities.append(distance.embeddings_answer_distance)
            vars_dissimilarities.append(distance.vars_distance)

    avg_q_dissim = -1 if not q_dissimilarities else np.mean(q_dissimilarities)
    avg_a_dissim = -1 if not a_dissimilarities else np.mean(a_dissimilarities)
    avg_vars_dissim = -1 if not vars_dissimilarities else np.mean(vars_dissimilarities)

    return avg_q_dissim, avg_vars_dissim, avg_a_dissim

def compute_average_max_dissimilarity(utterances):
    max_q = -1
    max_a = -1
    max_var = -1

    # Compute all pairwise distances for this test
    for i in range(len(utterances)):
        for j in range(i + 1, len(utterances)):
            distance = UtterancesDistance.calculate(
                utterances[i].get("X")[0],
                utterances[j].get("X")[0]
            )
            max_q = max(max_q, distance.embeddings_distance)
            max_a = max(max_a, distance.embeddings_answer_distance)
            max_var = max(max_var, distance.vars_distance)

    # Wrap into lists for averaging (keeping same structure as your original version)
    max_qs = [max_q] if max_q >= 0 else []
    max_as = [max_a] if max_a >= 0 else []
    max_vars = [max_var] if max_var >= 0 else []

    if not max_qs and not max_as and not max_vars:
        return -1, -1, -1

    avg_max_q = np.mean(max_qs) if max_qs else -1
    avg_max_a = np.mean(max_as) if max_as else -1
    avg_max_vars = np.mean(max_vars) if max_vars else -1

    return avg_max_q, avg_max_vars, avg_max_a

class TestReport(BaseModel):
    utterance: Utterance
    features_dict: Dict[str, Any] = dict()
    fitness: Dict[str, float] = dict()
    is_critical: bool = False
    poi_exists: bool = False
    other: Dict = None

def write_tests_to_json(res, save_folder: str, critical_only: bool = False, filename: Optional[str] = None):
    all_population = res.archive
    problem: QAProblem = res.problem
    if critical_only:
        all_population, _ = all_population.divide_critical_non_critical()

    if filename is None:
        filename = "critical_utterances" if critical_only else "all_utterances"

    # Ensure the save directory exists
    os.makedirs(save_folder, exist_ok=True)

    all_utterances: List[TestReport] = []

    for idx, ind in enumerate(all_population):
        utter: Utterance = ind.get("X")[0]  # Assuming "X" holds vars and is a list of lists
        poi_exists = ind.get("SO").poi_exists
        other = ind.get("SO").other

        fitness_names = list(problem.fitness_function.name)
        scores_dict = {name: float(score) for name, score in zip(fitness_names, ind.get("F").tolist())}
        is_critical = bool(ind.get("CB"))

        if hasattr(problem, "feature_handler"):
            feature_handler = problem.feature_handler
            features_dict = feature_handler.get_feature_values_dict(
                utter.ordinal_vars,
                utter.categorical_vars,
            )
        else:
            features_dict = {}

        all_utterances.append(
            TestReport(
                utterance=utter,
                features_dict=features_dict,
                fitness=scores_dict,
                is_critical=is_critical,
                other = other,
                poi_exists=poi_exists,
            ).model_dump(serialize_as_any=True)
        )
    # Construct a filename: e.g., utterance_0001.json
    filename = os.path.join(save_folder, f"{filename}.json")

    # Write JSON file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_utterances, f, ensure_ascii=False, indent=4)

def calculate_diversity(res, save_folder):
    """
    Calculate and store diversity statistics:
    - Input: Question diversity (semantic dissimilarity based on question embeddings)
    - Output: Answer diversity (semantic dissimilarity based on answer embeddings / output diversity)
    - Input: Variable-level diversity
    For both critical and all utterances.
    """
    all_population = res.archive
    critical, _ = all_population.divide_critical_non_critical()
    critical_clean = duplicate_free(critical)
    
    # --- Average pairwise dissimilarities ---
    avg_q_critical, avg_vars_critical, avg_a_critical = compute_pairwise_dissimilarity(critical_clean)
    avg_q_all, avg_vars_all, avg_a_all = compute_pairwise_dissimilarity(all_population)

    # --- Average max dissimilarities ---
    avg_max_q_all, avg_max_vars_all, avg_max_a_all= compute_average_max_dissimilarity(all_population)
    avg_max_q_critical, avg_max_vars_critical, avg_max_a_critical= compute_average_max_dissimilarity(critical_clean)

    if avg_q_critical == -1:
        log.info("Average Question Dissimilarity (Critical): Not available")
    if avg_a_critical == -1:
        log.info("Average Answer Dissimilarity (Critical): Not available")
    if avg_vars_critical == -1:
        log.info("Average Variable-Level Dissimilarity (Critical): Not available")

    if avg_q_all == -1:
        log.info("Average Question Dissimilarity (All): Not available")
    if avg_a_all == -1:
        log.info("Average Answer Dissimilarity (All): Not available")
    if avg_vars_all == -1:
        log.info("Average Variable-Level Dissimilarity (All): Not available")

    # # --- Cluster-level diversity ---
    # cluster_critical = cluster_utterances_vars([ind.get("X")[0] for ind in critical_clean])
    # cluster_all = cluster_utterances_vars([ind.get("X")[0] for ind in all_population])

    # --- Write results to CSV ---
    with open(save_folder + 'diversity.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Attribute', 'Value'])
        
        # Average pairwise distances
        writer.writerow(['Average Question Dissimilarity Critical', avg_q_critical])
        writer.writerow(['Average Answer Dissimilarity Critical', avg_a_critical])  
        writer.writerow(['Average Dissimilarity Critical (Vars)', avg_vars_critical])

        writer.writerow(['Average Question Dissimilarity All', avg_q_all])
        writer.writerow(['Average Answer Dissimilarity All', avg_a_all])            
        writer.writerow(['Average Dissimilarity All (Vars)', avg_vars_all])

        # Average max distances
        writer.writerow(['Average Max Question Dissimilarity Critical', avg_max_q_critical])
        writer.writerow(['Average Max Answer Dissimilarity Critical', avg_max_a_critical]) 
        writer.writerow(['Average Max Dissimilarity Critical (Vars)', avg_max_vars_critical])

        writer.writerow(['Average Max Question Dissimilarity All', avg_max_q_all])
        writer.writerow(['Average Max Answer Dissimilarity All', avg_max_a_all])           
        writer.writerow(['Average Max Dissimilarity All (Vars)', avg_max_vars_all])
        
        # # Cluster-based metrics
        # writer.writerow(['Critical Clusters: Best k', cluster_critical['best_k']])
        # writer.writerow(['Critical Clusters: Avg Medoid Distance', cluster_critical['avg_medoid_distance']])
        # writer.writerow(['Critical Clusters: Max Medoid Distance', cluster_critical['max_medoid_distance']])
        # writer.writerow(['Critical Clusters: Avg Max Medoid Distance', cluster_critical['avg_max_medoid_distance']])

        # writer.writerow(['All Clusters: Best k', cluster_all['best_k']])
        # writer.writerow(['All Clusters: Avg Medoid Distance', cluster_all['avg_medoid_distance']])
        # writer.writerow(['All Clusters: Max Medoid Distance', cluster_all['max_medoid_distance']])
        # writer.writerow(['All Clusters: Avg Max Medoid Distance', cluster_all['avg_max_medoid_distance']])

def write_failures_over_time(res, save_folder, interval = 1200):
    """
    Compute cumulative failures over time from a pymoo result archive, preserving execution order.

    Parameters
    ----------
    res : pymoo Result
        Result object containing archive.
    save_folder : str
        Folder to save the interpolated CSV.
    total_search_time : int
        Total search time in seconds.
    interval : int
        Interpolation step in seconds.

    Returns
    -------
    pd.DataFrame
        DataFrame with Time_s, Budget_%, FailuresFound.
    """
    os.makedirs(save_folder, exist_ok=True)

    total_search_time = res.exec_time

    all_population = res.archive  # Keep all individuals in order
    
    all_population = duplicate_free(all_population)

    # Store critical labels in order
    critical_labels = []
    for ind in all_population:
        # Assuming 'CB' is 1 for critical (failure) and 0 for non-critical
        critical_labels.append(ind.get("CB"))

    # Total number of individuals
    n_tests = len(all_population)
    time_per_test = total_search_time / n_tests

    # Discovery times
    discovery_times = [(i + 1) * time_per_test for i in range(n_tests)]

    # Cumulative failures
    cumulative_failures = np.cumsum(critical_labels)

    # Interpolation points
    time_points = np.arange(0, total_search_time + interval, interval)
    failures_over_time = np.interp(time_points, discovery_times, cumulative_failures)

    # Create result DataFrame
    result = pd.DataFrame({
        "Time_s": time_points,
        "Budget_%": (time_points / total_search_time) * 100,
        "FailuresFound": failures_over_time
    })

    # Save CSV
    csv_path = os.path.join(save_folder, "failures_over_time.csv")
    result.to_csv(csv_path, index=False)
    print(f"Interpolated failures saved to {csv_path}")

    return result

def copy_prompts(save_folder, source_path = "./llm/prompts.py"):
    destination_path = save_folder + "/prompts.txt"
    shutil.copy2(source_path, destination_path)
    return destination_path

def copy_config(save_folder, source_path = "./llm/config.py"):
    destination_path = save_folder + "/config.txt"
    shutil.copy2(source_path, destination_path)
    return destination_path

def show_critical_heatmap_features(res, save_folder, max_pairs = 3):
    problem = res.problem
    pop = res.obtain_archive()

    if len(pop) > 0 and pop[0].get("X")[0].ordinal_vars is not None:
        feature_names = problem.names_dim_utterance
        feature_bins = get_features()

        exclude_feature = "venue"
        exclude_idx = feature_names.index(exclude_feature) if exclude_feature in feature_names else -1

        all_pairs = list(combinations(range(len(feature_names)), 2))
        selected_pairs = [pair for pair in all_pairs if exclude_idx not in pair]

        print(f"Writing only max {max_pairs} heatmaps from {len(selected_pairs)}")
        selected_pairs = selected_pairs[:max_pairs]

        # === Extract samples ===
        X_all = np.array([ind.get("X")[0].ordinal_vars for ind in pop])  # all tests
        X_critical = np.array([ind.get("X")[0].ordinal_vars for ind in pop if ind.get("CB")])  # only critical

        for i, j in selected_pairs:
            fname_i, fname_j = feature_names[i], feature_names[j]
            bins_i = feature_bins[fname_i].categories
            bins_j = feature_bins[fname_j].categories
            n_bins_i, n_bins_j = len(bins_i), len(bins_j)

            def bin_index(val, n_bins):
                return min(max(int(val * n_bins), 0), n_bins - 1)

            # Heatmaps
            heatmap = np.zeros((n_bins_i, n_bins_j), dtype=int)
            count_heatmap = np.zeros((n_bins_i, n_bins_j), dtype=int)

            # Count critical failures
            for row in X_critical:
                xi = bin_index(row[i], n_bins_i)
                yj = bin_index(row[j], n_bins_j)
                heatmap[xi, yj] += 1

            # Count total tests
            for row in X_all:
                xi = bin_index(row[i], n_bins_i)
                yj = bin_index(row[j], n_bins_j)
                count_heatmap[xi, yj] += 1

            # Annotation: show total test count per cell
            annot_labels = np.empty_like(heatmap, dtype=object)
            for xi in range(n_bins_i):
                for yj in range(n_bins_j):
                    failures = heatmap[xi, yj]
                    total = count_heatmap[xi, yj]
                    annot_labels[xi, yj] = f"{failures}/{total}" if total > 0 else "0/0"

            # === Plot heatmap ===
            plt.figure(figsize=(max(10, len(bins_j) * 0.6), max(6, len(bins_i) * 0.4)))
            sns.heatmap(
                heatmap,
                annot=annot_labels,
                fmt="",
                cmap="YlGnBu",
                xticklabels=bins_j,
                yticklabels=bins_i,
                cbar=True
            )
            plt.title(f"Heatmap (Critical Failures/Number Tests)\n{fname_i} vs {fname_j}")
            plt.xlabel(fname_j)
            plt.ylabel(fname_i)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()

            save_folder_heatmap = save_folder + os.sep + "htmp/"
            Path(save_folder_heatmap).mkdir(exist_ok=True, parents=True)
            plt.savefig(save_folder_heatmap + f"heatmap_critical_{feature_names[i]}_{feature_names[j]}.png", format="png")    
