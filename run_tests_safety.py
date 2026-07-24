import argparse
import os
import warnings
from datetime import datetime
import weave

import wandb
from examples.safety.eval import AstralFitnessAnswerValidation, CriticalAstral
from examples.safety.utterance_generator import AstralUtteranceGenerator
from llm.llms import ALL_MODELS, LLMType
from llm.model.qa_problem import QAProblem
from llm.model.search_configuration import QASearchConfiguration, QASearchOperators
from llm.operators.utterance_crossover_discrete import UtteranceCrossoverDiscrete
from llm.operators.utterance_duplicates import UtteranceDuplicateElimination, UtteranceDuplicateEliminationDistance
from llm.operators.utterance_duplicates_discrete import (
    UtteranceDuplicateEliminationDiscrete,
)
from llm.operators.utterance_mutator_discrete import UtteranceMutationDiscrete
from llm.operators.utterance_repair import (
    NoUtteranceRepair,
    UtteranceRepairQuestionGenerator,
)
from llm.operators.utterance_sampling_discrete import (
    UtteranceSamplingDiscrete,
    UtteranceSamplingGrid,
)
from llm.sut.io_simulation import IOSimulator
from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.algorithm.nsga2d_optimizer import NSGAIIDOptimizer
from opensbt.algorithm.optimizer import Optimizer
from opensbt.algorithm.ps import PureSampling
from opensbt.config import LOG_FILE, RESULTS_FOLDER
from opensbt.utils.log_utils import disable_pymoo_warnings, log, setup_logging
from opensbt.utils.wandb import logging_callback_archive, TableCallback
from opensbt.utils.callback import merged_callbacks


def parse_args():
    parser = argparse.ArgumentParser(description="Run Navi test with selectable SUT")
    parser.add_argument(
        "--population_size",
        type=int,
        default=4,
        help="Population size for GA (default: 4)",
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=1,
        help="Number of generations (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=4, help="Seed")
    parser.add_argument(
        "--max_time",
        type=str,
        default=None,
        help="Maximal execution time as string 'hh:mm:ss'",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["rs", "gs", "nsga2", "nsga2d"],
        default="nsga2",
        help="Algorithm.",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default=RESULTS_FOLDER,
        help="Path to the folder where results or test cases will be stored.",
    )
    parser.add_argument(
        "--sut",
        type=str,
        choices=ALL_MODELS,
        default=LLMType.DEEPSEEK_V2.value,
        help="The LLM tested.",
    )
    parser.add_argument(
        "--generator",
        type=str,
        choices=ALL_MODELS,
        default=LLMType.DOLPHIN3.value,
        help="The LLM used to generate inputs",
    )
    parser.add_argument(
        "--fitness",
        type=str,
        choices=ALL_MODELS,
        default=LLMType.GPT_4O_MINI.value,
        help="The LLM used to calculate fitness",
    )
    parser.add_argument(
        "--judge",
        type=str,
        choices=ALL_MODELS,
        default=LLMType.GPT_4O_MINI.value,
        help="The LLM used to select failures",
    )
    parser.add_argument(
        "--use_repair",
        action="store_true",
        help="Use the repair operator instead of generating utterances during crossover and mutation",
    )
    parser.add_argument(
        "--features_config",
        type=str,
        default="configs/safety_features.json",
        help="Path to the file with feature config",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Turn off wanbd logging"
    )
    parser.add_argument(
        "--no_rag",
        action="store_true",
        help="Turn off rag"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    warnings.filterwarnings(
        "ignore", category=FutureWarning, message=".*encoder_attention_mask.*"
    )

    os.chmod(os.getcwd(), 0o777)
    logger = log.getLogger(__name__)
    setup_logging(LOG_FILE)
    disable_pymoo_warnings()
    llm_generator = LLMType(args.generator)

    search_operatoes = QASearchOperators(
        crossover=UtteranceCrossoverDiscrete(
            llm_type=llm_generator, generate_question=not args.use_repair
        ),
        sampling=(
            UtteranceSamplingGrid(
                llm_type=llm_generator, total_samples=args.population_size
            )
            if args.algorithm == "gs"
            else UtteranceSamplingDiscrete(llm_type=llm_generator,
                generate_question=not args.use_repair)
        ),
        mutation=UtteranceMutationDiscrete(
            llm_type=llm_generator, generate_question=not args.use_repair
        ),
        duplicate_elimination=UtteranceDuplicateEliminationDistance(),
        repair=(
            UtteranceRepairQuestionGenerator(llm_type=llm_generator)
            if args.use_repair
            else NoUtteranceRepair()
        ),
    )

    config = QASearchConfiguration(
        operators=search_operatoes,
        population_size=args.population_size,
        n_generations=args.n_generations,
        maximal_execution_time=args.max_time,
        results_folder=args.results_folder,
        n_repopulate_max=0.5,
    )

    fitness = AstralFitnessAnswerValidation(llm_type=LLMType(args.fitness))
    critical = CriticalAstral(llm_type=LLMType(args.judge))

    # we update the name based on the sut used
    problem_name = (
        f"{args.sut}"
        + f"_{config.population_size}n"
        + (f"_{config.n_generations}i" if config.n_generations is not None else "")
        + (
            f"_{config.maximal_execution_time.replace(':', '_')}t"
            if config.maximal_execution_time is not None
            else ""
        )
        + f"_{args.seed}seed"
        + f"_{args.algorithm.upper()}"
        + f"_{args.features_config.split('.')[0].replace('/', '_')}"
    )

    tags = [f"{k}:{v}" for k, v in vars(args).items() if k != "features_config"]
    tags.append(f"features:{'astral' if 'astral' in args.features_config else 'extended'}")

    wandb_entity = os.getenv("WANDB_ENTITY", "opentest")
    wandb_project = os.getenv("WANDB_PROJECT", "demo")

    if not args.no_wandb:
        weave.init(wandb_project)
        wandb.init(
            entity=wandb_entity,                 # team
            project=wandb_project,               # the project name
            name=problem_name,                  # run name
            group=datetime.now().strftime("%d-%m-%Y"),  # group by date
            tags=tags,
        )
    else:
        wandb.init(mode="disabled")

    problem = QAProblem(
        problem_name=problem_name,
        scenario_path=os.getcwd() + "",
        xl=[0],
        xu=[1],
        simulation_variables=["utterance"],
        fitness_function=fitness,
        critical_function=critical,
        simulate_function=IOSimulator(llm_type=LLMType(args.sut)).simulate,
        seed_utterances=[
            "",
        ],
        context={},
        seed=args.seed,
        names_dim_utterance=["utterance"],
        feature_handler_config_path=args.features_config,
        question_generator=AstralUtteranceGenerator(use_rag=not args.no_rag),
    )

    seed_name = "_".join(problem.seed_utterances[0][:-1].split())
    callback = merged_callbacks(
        logging_callback_archive, TableCallback().log
    )

    optimizer_map = {
        "rs": PureSampling,
        "gs": PureSampling,
        "nsga2": NsgaIIOptimizer,
        "nsga2d": NSGAIIDOptimizer,
    }
    if args.algorithm not in optimizer_map:
        raise ValueError("Algorithm not supported")
    optimizer_class: type[Optimizer] = optimizer_map[args.algorithm]
    optimizer = optimizer_class(problem, config, callback=callback, algorithm_name=args.algorithm)
    res = optimizer.run()
    res.write_results(
        results_folder=optimizer.save_folder, params=optimizer.parameters, search_config=config
    )

    log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")
