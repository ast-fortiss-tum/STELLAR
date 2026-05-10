import os
from llm.adapter.embeddings_local_adapter import get_disimilarity_individual
from llm.llms import ALL_MODELS, LLMType
from opensbt.utils.wandb import logging_callback_archive
import pymoo
import wandb
import weave

from llm.sut.ipa_los import IPA_LOS
from llm.sut.ipa_yelp import IPA_YELP
from opensbt.algorithm.nsga2d_optimizer import NSGAIIDOptimizer
from opensbt.algorithm.ps_rand import PureSamplingRand

from llm.model.search_configuration import QASearchConfiguration, QASearchOperators
from llm.model.qa_problem import QAProblem
from llm.sut.ipa import IPA
from examples.navi.navi_utterance_generator import NaviUtteranceGenerator
from llm.eval.fitness import FitnessMerged, FitnessDiverse, FitnessNumberOfWords
from examples.navi.fitness import NaviFitnessAnswerValidationDimensions, NaviFitnessContentComparison
from llm.eval.critical import CriticalMerged, CriticalByFitnessThreshold, CriticalAnswerLength
from llm.operators.utterance_crossover_discrete import UtteranceCrossoverDiscrete
from llm.operators.utterance_sampling_discrete import UtteranceSamplingDiscrete
from llm.operators.utterance_mutator_discrete import UtteranceMutationDiscrete
from llm.operators.utterance_duplicates_discrete import UtteranceDuplicateEliminationDiscreteWithContent, UtteranceDuplicateEliminationLocalDiscreteWithContent
from opensbt.utils.log_utils import log, setup_logging, disable_pymoo_warnings
from opensbt.config import RESULTS_FOLDER, LOG_FILE
from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from llm.utils.name import create_problem_name
from llm.sut.ipa_los import IPA_LOS
from llm.sut.ipa_yelp import IPA_YELP

from datetime import datetime
import argparse
import warnings

import warnings

def parse_args():
    parser = argparse.ArgumentParser(description="Run Navi test with selectable SUT")
    parser.add_argument(
        "--sut",
        type=str,
        default="IPA_LOS",
        choices=["IPA_LOS", "IPA_YELP"],
        help="Select the System Under Test (SUT) to use"
    )

    parser.add_argument(
        "--population_size",
        type=int,
        default=2,
        help="Population size for GA (default: 4)"
    )

    parser.add_argument(
        "--n_generations",
        type=int,
        default=2,
        help="Number of generations (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4,
        help="Seed"
    )
    parser.add_argument(
        "--max_time",
        type=str,
        default=None,
        help="Maximal execution time as string 'hh:mm:ss'"
    )
    parser.add_argument(
        "--th_answer",
        type=float,
        default=0.75,
        help="Threshold answer."
    )
    parser.add_argument(
        "--th_content",
        type=float,
        default=0.75,
        help="Threshold content."
    )
    parser.add_argument("--algorithm", type=str, 
        choices=["rs", "nsga2", "nsga2d"], 
        default="nsga2d",
        help="Algorithm."
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default=RESULTS_FOLDER,
        help="Path to the folder where results or test cases will be stored."
    )   
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Turn off wanbd logging"
    )
    parser.add_argument(
        "--use_rag",
        action="store_true",
        help="Turn off rag in test generation"
    )
    parser.add_argument(
        "--judge",
        type=str,
        choices=ALL_MODELS,
        default=LLMType.GPT_4O_MINI.value,
        help="The LLM used for judgement.",
    )
    parser.add_argument(
        "--features_config",
        type=str,
        default="configs/features_simple_judge.json",
        help="Path to the file with feature config",
    )
    parser.add_argument(
        "--judge_weights",
        nargs="+",
        default=[1, 0, 0.5],
        help="Weights for the judge dimension.",
    )
    return parser.parse_args()

args = parse_args()

warnings.filterwarnings("ignore", category=FutureWarning, message=".*encoder_attention_mask.*")

os.chmod(os.getcwd(), 0o777)
logger = log.getLogger(__name__)
setup_logging(LOG_FILE)
disable_pymoo_warnings()

SUT_MAP = {
    "IPA_LOS": IPA_LOS,
    "IPA_YELP": IPA_YELP
}

SUT_CLASS = SUT_MAP[args.sut]

search_operatoes = QASearchOperators(
    crossover=UtteranceCrossoverDiscrete(),
    sampling=UtteranceSamplingDiscrete(),
    mutation=UtteranceMutationDiscrete(),
    duplicate_elimination=UtteranceDuplicateEliminationLocalDiscreteWithContent(),
)

config = QASearchConfiguration(
    operators=search_operatoes,
)
config.population_size = args.population_size
config.n_generations = args.n_generations
config.maximal_execution_time = args.max_time
config.n_repopulate_max = 0.5
config.results_folder = args.results_folder

fitness = FitnessMerged([
    NaviFitnessAnswerValidationDimensions(weights=args.judge_weights,
                                        llm_type=LLMType(args.judge)),
    NaviFitnessContentComparison(),
    FitnessDiverse(),
    # FitnessNumberOfWords(),
])

critical = CriticalMerged(
    fitness_names=fitness.name,
    criticals=[ # TODO select min scores here
        (CriticalByFitnessThreshold(mode = "<", score=args.th_answer), ["answer_fitness"]),
        (CriticalByFitnessThreshold(mode = "<", score=args.th_content), ["content_fitness"]),
        # (CriticalByFitnessThreshold(mode = "<", score=0.75), ["raw_output_fitness"]),
        #(CriticalAnswerLength(limit = 30), [])
    ],
    mode="or",
)

simulate_function = SUT_CLASS.simulate
seed = args.seed

problem_name = create_problem_name(
        simulate_function,
        suffix=(
            f"{config.population_size}n"
            + (f"_{config.n_generations}i" if config.n_generations is not None else "")
            + (f"_{config.maximal_execution_time.replace(':','-')}t" if config.maximal_execution_time is not None else "")
            + f"_{seed}seed"
            + f"_{args.algorithm.upper()}"
        )
    )

tags = [f"{k}:{v}" for k, v in vars(args).items() if k != "features_config"]

if not args.no_wandb:
    weave.init("dev")
    wandb.init(
        entity="opentest",                  # team
        project="dev",                      # the project name
        name=problem_name,          # run name
        group=datetime.now().strftime("%d-%m-%Y"),  # group by date
        tags=tags
    )   
else:
    wandb.init(mode="disabled")

problem = QAProblem(
            problem_name="Test",
            scenario_path=os.getcwd() + "",
            xl=[0],
            xu=[1],
            simulation_variables=["utterance"],
            fitness_function=fitness,  
            critical_function=critical,
            simulate_function=simulate_function,
            seed_utterances = [
                "hmm I need some food",
                "yeah I need some burgers",
                "oh some burger would be great"
            ],
            context ={
                        "location" : {
                            "position" : "Amathountos Avenue 502, 4520, Limassol, Cyprus",
                            "date" : "2025-03-19T0",
                            "time" : "09:00:00",
                        },
                        "person" : {
                            "gender" : "female",
                            "age" : 30
                        }
                    },
            seed = seed,
            names_dim_utterance=["utterance"],
            feature_handler_config_path=args.features_config,
            question_generator=NaviUtteranceGenerator(use_rag=args.use_rag),
)
problem.problem_name = problem_name

# callback for dynamic logging (works only if wandb logging is activated)

if args.algorithm == "nsga2":
    optimizer = NsgaIIOptimizer(
                        problem=problem,
                        config=config,
                        callback=logging_callback_archive)
    
elif args.algorithm == "nsga2d":
    optimizer = NSGAIIDOptimizer(
                            problem=problem,
                            config=config,
                            callback = logging_callback_archive,
                            dist_function=get_disimilarity_individual)
elif args.algorithm == "rs":
    optimizer = PureSamplingRand(
                            problem=problem,
                            config=config,
                            sampling_type=UtteranceSamplingDiscrete,
                            callback = logging_callback_archive)
else:
    raise ValueError("Algorithm not known.")

res = optimizer.run()
print(f"{optimizer.__dict__}")

res.write_results(results_folder=optimizer.save_folder,
                  params = optimizer.parameters,
                  search_config=config)

log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")
