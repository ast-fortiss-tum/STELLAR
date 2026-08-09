import os
import warnings

from opensbt.utils.log_utils import log, setup_logging, disable_pymoo_warnings
from opensbt.config import RESULTS_FOLDER, LOG_FILE

from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.algorithm.ps_rand import PureSamplingRand

from llm.model.search_configuration import QASearchConfiguration, QASearchOperators
from llm.model.qa_problem import QAProblem

from llm.operators.utterance_crossover_discrete import UtteranceCrossoverDiscrete
from llm.operators.utterance_duplicates import UtteranceDuplicateEliminationDistance
from llm.operators.utterance_mutator_discrete import UtteranceMutationDiscrete
from llm.operators.utterance_repair import (
    NoUtteranceRepair,
    UtteranceRepairQuestionGenerator,
)
from llm.operators.utterance_sampling_discrete import (
    UtteranceSamplingDiscrete,
    UtteranceSamplingGrid,
)
from llm.eval.critical import CriticalMerged, CriticalByFitnessThreshold

from tarot.misc.fitness import IntentMatchFitness
from tarot.misc.nlu_utterance_generator import NLUUtteranceGenerator

from jupyter.nlu_bot.ipa_nlu_bot import IPA_NLU_BOT
from llm.llms import ALL_MODELS, LLMType

import wandb

import warnings
warnings.filterwarnings(
"ignore",
message="CUDA initialization: The NVIDIA driver on your system is too old.*",
category=UserWarning,
)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    # when using random search, set the number of samples via population size variable
    parser.add_argument("--population_size", type=int, default=2)
    parser.add_argument("--n_generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--max_time", type=str, default="00:01:00")

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["random", "nsga2"],
        default="nsga2",
    )

    parser.add_argument("--results_folder", type=str, default=RESULTS_FOLDER)
    parser.add_argument("--features_config", type=str, default="tarot/configs/features_nlu.json")
    parser.add_argument(
        "--generator",
        type=str,
        choices=ALL_MODELS,
        default=LLMType.GPT_4O_MINI,
        help="The LLM used to generate inputs",
    )
    return parser.parse_args()


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
        repair=(
            UtteranceRepairQuestionGenerator(llm_type=llm_generator)
        ),
    )

    config = QASearchConfiguration(operators=operators)
    config.population_size = args.population_size
    config.n_generations = args.n_generations
    config.maximal_execution_time = args.max_time
    config.results_folder = args.results_folder

    return config


def build_fitness_and_critical():
    fitness = IntentMatchFitness()

    critical = CriticalMerged(
        fitness_names=fitness.name,
        criticals=[
            (CriticalByFitnessThreshold(mode="<", score=1), ["intent_match_score"]),
        ],
        mode="or",
    )

    return fitness, critical


def build_problem(args, config, fitness, critical):
    return QAProblem(
        problem_name=fr"NLU_{args.algorithm}_p{config.population_size}_g{config.n_generations}_t-{args.max_time}_s{args.seed}_f-{args.features_config}".replace(":", "-").replace("/", "-"),
        scenario_path=os.getcwd(),
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
        feature_handler_config_path=args.features_config,
        question_generator=NLUUtteranceGenerator(),
    )


def build_optimizer(args, problem, config):
    if args.algorithm == "nsga2":
        return NsgaIIOptimizer(
            problem=problem,
            config=config,
        )

    if args.algorithm == "random":
        return PureSamplingRand(
            problem=problem,
            config=config,
        )

    raise ValueError("Unknown algorithm")
