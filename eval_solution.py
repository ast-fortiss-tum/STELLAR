from tarot.solutions.baseline import CustomNLUUtteranceGenerator, CustomIntentMatchFitness

# Set the test parameters first to try different search configurations

population_size = 5           # number of tests per generation
max_time = "00:01:00"         # total execution time for the search, e.g., "00:00:10" for 10 seconds
n_generations = 20            # if max_time is set, n_generations will be ignored
seed = 1


from argparse import Namespace
import csv
import json
import os
from pathlib import Path
import sys
import warnings

import wandb
from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.algorithm.ps_rand import PureSamplingRand
from opensbt.config import LOG_FILE
from opensbt.utils.log_utils import (
    disable_pymoo_warnings,
    log,
    setup_logging,
 )

candidate_roots = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
project_root = next((path for path in candidate_roots if (path / "llm").exists()), None)

if project_root is None:
    raise FileNotFoundError("Could not locate the STELLAR project root")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.chdir(project_root)

from llm.eval.critical import CriticalByFitnessThreshold, CriticalMerged
from llm.llms import LLMType
from llm.model.qa_problem import QAProblem
from llm.model.search_configuration import QASearchConfiguration, QASearchOperators
from llm.operators.utterance_crossover_discrete import UtteranceCrossoverDiscrete
from llm.operators.utterance_duplicates import UtteranceDuplicateEliminationDistance
from llm.operators.utterance_mutator_discrete import UtteranceMutationDiscrete
from llm.operators.utterance_repair import UtteranceRepairQuestionGenerator
from llm.operators.utterance_sampling_discrete import (
    UtteranceSamplingDiscrete,
    UtteranceSamplingGrid,
 )
from tarot.nlu_bot.ipa_nlu_bot import IPA_NLU_BOT

def configure_runtime() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    setup_logging(LOG_FILE)
    disable_pymoo_warnings()
    if wandb.run is None:
        wandb.init(mode="disabled")

def build_search_config(args):
    llm_generator = LLMType(args.generator)

    operators = QASearchOperators(
        crossover=UtteranceCrossoverDiscrete(
            llm_type=llm_generator,
            generate_question=True,
        ),
        sampling=(
            UtteranceSamplingGrid(
                llm_type=llm_generator,
                total_samples=args.population_size,
                t=4,
            )
            if args.algorithm == "random"
            else UtteranceSamplingDiscrete(
                llm_type=llm_generator,
                generate_question=True,
            )
        ),
        mutation=UtteranceMutationDiscrete(
            llm_type=llm_generator,
            generate_question=True,
        ),
        duplicate_elimination=UtteranceDuplicateEliminationDistance(),
        repair=UtteranceRepairQuestionGenerator(llm_type=llm_generator),
    )

    config = QASearchConfiguration(operators=operators)
    config.population_size = args.population_size
    config.n_generations = args.n_generations
    config.maximal_execution_time = args.max_time
    config.results_folder = args.results_folder
    return config

def build_optimizer(args, problem, config):
    if args.algorithm == "nsga2":
        return NsgaIIOptimizer(problem=problem, config=config)

    if args.algorithm == "random":
        return PureSamplingRand(problem=problem, config=config)

    raise ValueError("Unknown algorithm")

def report_execution_metrics(results_folder: str) -> dict:
    results_path = Path(results_folder)

    summary_results_path = results_path / "summary_results.csv"
    all_critical_utterances_path = results_path / "all_critical_utterances.json"
    llm_usage_summary_path = results_path / "llm_usage_summary.json"

    fail_rate = None
    with summary_results_path.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("Attribute") == "Ratio Critical/All scenarios (duplicate free)":
                fail_rate = float(row["Value"])
                break

    if fail_rate is None:
        raise ValueError(
            "Could not find 'Ratio Critical/All scenarios (duplicate free)' in summary_results.csv"
        )

    with all_critical_utterances_path.open("r", encoding="utf-8") as json_file:
        all_critical_utterances = json.load(json_file)

    diverse_intents = set()
    for entry in all_critical_utterances:
        utterance = entry.get("utterance", {})
        content_input = utterance.get("content_input", {})
        intent = content_input.get("intent")
        if intent:
            diverse_intents.add(intent)

    with llm_usage_summary_path.open("r", encoding="utf-8") as json_file:
        llm_usage_summary = json.load(json_file)

    tokens_spent = llm_usage_summary["total"]["tokens"]

    metrics = {
        "fail_rate": fail_rate,
        "num_diverse_fails": len(diverse_intents),
        "tokens_spent": tokens_spent,
    }

    metrics_path = results_path / "execution_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as json_file:
        json.dump(metrics, json_file, indent=4)

    print(json.dumps(metrics, indent=4))
    return metrics

feature_config_path = project_root / "tarot" / "configs" / "features_nlu.json"

args = Namespace(
    population_size=population_size,
    n_generations=n_generations,
    seed=seed,
    max_time=max_time,
    algorithm="nsga2",
    results_folder="results",
    features_config=str(feature_config_path),
    generator=LLMType.GPT_4O_MINI,
 )

configure_runtime()
log.info("Starting notebook NLU search experiment")

config = build_search_config(args)
fitness = CustomIntentMatchFitness()
critical = CriticalMerged(
    fitness_names=fitness.name,
    criticals=[
        (CriticalByFitnessThreshold(mode="<", score=1), ["intent_match_score"]),
    ],
    mode="or",
 )

problem = QAProblem(
    problem_name=(
        f"NLU_test_{args.algorithm}_p{config.population_size}"
        f"_g{config.n_generations}_t-{args.max_time}_s{args.seed}"
    ).replace(":", "-").replace("/", "-"),
    scenario_path=str(project_root),
    xl=[0],
    xu=[1],
    simulation_variables=["utterance"],
    fitness_function=fitness,
    critical_function=critical,
    simulate_function=IPA_NLU_BOT.simulate,
    seed_utterances=["play some radio"],
    context={},
    seed=args.seed,
    names_dim_utterance=["utterance"],
    feature_handler_config_path=str(feature_config_path),
    question_generator=CustomNLUUtteranceGenerator(),
 )

optimizer = build_optimizer(args, problem, config)
result = optimizer.run()
result.write_results(
    results_folder=optimizer.save_folder,
    params=optimizer.parameters,
    search_config=config,
 )

results_root = Path(optimizer.save_folder)
if results_root.is_absolute():
    try:
        results_root = results_root.relative_to(project_root)
    except ValueError:
        results_root = Path(os.path.relpath(results_root, project_root))
os.environ["STELLAR_RESULTS_ROOT"] = str(results_root)

metrics = report_execution_metrics(optimizer.save_folder)