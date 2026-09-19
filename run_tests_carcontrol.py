import argparse
import os
import warnings
from datetime import datetime
from functools import partial

import wandb
import weave

from examples.car_control.cc_utterance_generator import CCUtteranceGenerator
from examples.car_control.fitness import CCFitnessAnswerValidationDimensions
from llm.eval.critical import CriticalByFitnessThreshold, CriticalMerged
from llm.eval.fitness import FitnessDiverse, FitnessMerged
from llm.eval.utterances_distance import get_feature_distance_individual
from llm.llms import ALL_MODELS, LLMType
from llm.model.qa_problem import QAProblem
from llm.model.search_configuration import QASearchConfiguration, QASearchOperators
from llm.operators.utterance_crossover_discrete import UtteranceCrossoverDiscrete
from llm.operators.utterance_duplicates_discrete import (
    UtteranceDuplicateEliminationLocalDiscreteWithContent,
)
from llm.operators.utterance_mutator_discrete import UtteranceMutationDiscrete
from llm.operators.utterance_sampling_discrete import (
    UtteranceSamplingDiscrete,
    UtteranceSamplingDiscreteDiverse,
    UtteranceSamplingGrid,
)
from llm.sut.ipa import IPA
from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.algorithm.nsga2d_optimizer import NSGAIIDOptimizer
from opensbt.algorithm.ps_rand import PureSamplingRand
from opensbt.config import LOG_FILE, RESULTS_FOLDER
from opensbt.utils.log_utils import disable_pymoo_warnings, log, setup_logging
from opensbt.utils.wandb import logging_callback_archive


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-turn Car Controls tests.")
    parser.add_argument("--population_size", type=int, default=2)
    parser.add_argument("--n_generations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--max_time", type=str, default=None)
    parser.add_argument(
        "--algorithm",
        choices=["rs", "gs", "nsga2", "nsga2d", "nsga2ds"],
        default="nsga2d",
    )
    parser.add_argument("--results_folder", type=str, default=RESULTS_FOLDER)
    parser.add_argument("--sut", choices=ALL_MODELS, default=LLMType.GPT_4O_MINI.value)
    parser.add_argument("--generator", choices=ALL_MODELS, default=LLMType.GPT_4O_MINI.value)
    parser.add_argument("--judge", choices=ALL_MODELS, default=LLMType.GPT_4O_MINI.value)
    parser.add_argument(
        "--features_config",
        type=str,
        default="configs/car_control_features.json",
    )
    parser.add_argument("--th_answer", type=float, default=0.6)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default="opentest")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="stellar-car-control-algorithm-smoke",
    )
    parser.add_argument("--use_diverse_sampling", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.chmod(os.getcwd(), 0o777)
    setup_logging(LOG_FILE)
    disable_pymoo_warnings()
    llm_generator = LLMType(args.generator)

    operators = QASearchOperators(
        crossover=UtteranceCrossoverDiscrete(llm_type=llm_generator),
        sampling=(
            UtteranceSamplingGrid(
                llm_type=llm_generator,
                total_samples=args.population_size,
                t=2,
            )
            if args.algorithm == "gs"
            else (
                UtteranceSamplingDiscreteDiverse(llm_type=llm_generator)
                if args.use_diverse_sampling or args.algorithm == "nsga2ds"
                else UtteranceSamplingDiscrete(llm_type=llm_generator)
            )
        ),
        mutation=UtteranceMutationDiscrete(llm_type=llm_generator),
        duplicate_elimination=UtteranceDuplicateEliminationLocalDiscreteWithContent(),
    )
    config = QASearchConfiguration(operators=operators)
    config.population_size = args.population_size
    config.n_generations = args.n_generations
    config.maximal_execution_time = args.max_time
    config.n_repopulate_max = 0.5
    config.results_folder = args.results_folder

    fitnesses = [CCFitnessAnswerValidationDimensions(llm_type=LLMType(args.judge))]
    if args.algorithm in {"nsga2d", "nsga2ds"}:
        fitnesses.append(FitnessDiverse(dist_fnc=get_feature_distance_individual))
    fitness = FitnessMerged(fitnesses)
    critical = CriticalMerged(
        fitness_names=fitness.name,
        criticals=[
            (CriticalByFitnessThreshold(mode="<", score=args.th_answer), ["answer_fitness"]),
        ],
        mode="or",
    )

    problem_name = (
        f"car_control_{args.sut}_{args.population_size}n_{args.n_generations}i_"
        f"{args.seed}seed_{args.algorithm.upper()}"
    )
    problem = QAProblem(
        problem_name=problem_name,
        scenario_path=os.getcwd(),
        xl=[0],
        xu=[1],
        simulation_variables=["utterance"],
        fitness_function=fitness,
        critical_function=critical,
        simulate_function=partial(IPA.simulate, llm_type=LLMType(args.sut)),
        seed_utterances=["Adjust the car settings."],
        context={},
        seed=args.seed,
        names_dim_utterance=["utterance"],
        feature_handler_config_path=args.features_config,
        question_generator=CCUtteranceGenerator(),
    )

    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        weave.init(args.wandb_project)
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=problem_name,
            group=datetime.now().strftime("%d-%m-%Y"),
            tags=[f"{key}:{value}" for key, value in vars(args).items()],
        )

    if args.algorithm == "nsga2":
        optimizer = NsgaIIOptimizer(problem, config, callback=logging_callback_archive)
    elif args.algorithm in {"nsga2d", "nsga2ds"}:
        optimizer = NSGAIIDOptimizer(
            problem,
            config,
            callback=logging_callback_archive,
            dist_function=get_feature_distance_individual,
        )
    elif args.algorithm in {"rs", "gs"}:
        optimizer = PureSamplingRand(problem, config, callback=logging_callback_archive)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    result = optimizer.run()
    result.write_results(
        results_folder=optimizer.save_folder,
        params=optimizer.parameters,
        search_config=config,
    )
    log.info("====== Algorithm search time: %.2f sec", result.exec_time)
    wandb.finish()