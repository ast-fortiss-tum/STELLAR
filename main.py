import os
os.environ["WANDB__SERVICE"] = "false"

import argparse
import logging
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import pymoo
import wandb
import weave

from opensbt.algorithm.nsga2d_optimizer import NSGAIIDOptimizer
from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.algorithm.ps_rand import PureSamplingRand
from opensbt.utils.log_utils import log, setup_logging, disable_pymoo_warnings
from opensbt.config import RESULTS_FOLDER, LOG_FILE
from opensbt.utils.wandb import logging_callback_archive

from llm.model.qa_problem import QAProblem
from llm.eval.critical import CriticalByFitnessThreshold, CriticalMerged
from llm.eval.fitness import FitnessMerged
from llm.llms import ALL_MODELS, LLMType


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified runner for single-turn / multi-turn × navi / cc."
    )
    # General settings
    parser.add_argument("--case_study", type=str, choices=["cc", "navi", "safety"], required=True, help="Case study.")
    parser.add_argument("--mode", type=str, choices=["single-turn", "multi-turn"], required=True, help="Execution mode.")
    parser.add_argument("--algorithm", type=str, choices=["rs", "gs", "nsga2", "nsga2d", "nsga2ds"], default="nsga2d", help="Algorithm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (default depends on case_study/mode).")
    parser.add_argument("--wandb_entity", type=str, default="opentest", help="W&B entity or team name.")
    parser.add_argument("--features_config", type=str, default=None, help="Path to features config JSON (default depends on case_study/mode).")
    parser.add_argument("--max_time", type=str, default=None, help="Maximal execution time as 'hh:mm:ss'. Use 'None' to disable.")
    parser.add_argument("--sut", type=str, default=None, help="System under test (default depends on case_study/mode).")
    parser.add_argument("--save_folder", type=str, default=RESULTS_FOLDER, help="Results folder.")
    # Numerical hyperparameters
    parser.add_argument("--n", type=int, default=6, help="Population size.")
    parser.add_argument("--i", type=int, default=6, help="Number of generations.")
    parser.add_argument("--weight_clarity", type=float, default=0.5, help="Weight for clarity dimension.")
    parser.add_argument("--weight_request_orientedness", type=float, default=0.5, help="Weight for request-orientedness.")
    parser.add_argument("--judge_weights", nargs="+", type=float, default=[0.55, 0.30, 0.15], help="Weights for judge dimensions (navi single-turn).")
    parser.add_argument("--th_dims", type=float, default=0.7, help="Critical threshold for dimensions fitness.")
    parser.add_argument("--th_efficiency", type=float, default=0.7, help="Critical threshold for efficiency fitness.")
    parser.add_argument("--th_effectiveness", type=float, default=0.7, help="Critical threshold for effectiveness fitness.")
    parser.add_argument("--th_answer", type=float, default=None, help="Threshold answer (default depends on case_study).")
    parser.add_argument("--th_content", type=float, default=None, help="Threshold content (default depends on case_study).")
    parser.add_argument("--min_turns", type=int, default=3, help="Min turns in a conversation.")
    parser.add_argument("--max_turns", type=int, default=6, help="Max turns in a conversation.")
    parser.add_argument("--max_repeats", type=int, default=2, help="Max repeat intents per conversation.")
    parser.add_argument("--archive_threshold", type=float, default=None, help="Archive threshold for NSGA-II-D.")
    # LLMs
    parser.add_argument("--llm_ipa", type=str, default="gpt-5-chat", help="LLM for SUT.")
    parser.add_argument("--llm_intent_classifier", type=str, default=None, help="LLM for intent classification.")
    parser.add_argument("--llm_judge", type=str, default="gpt-5-mini", help="LLM for judge.")
    parser.add_argument("--llm_generator", type=str, default="gpt-5-mini", help="LLM for generation.")
    # Extra generation settings
    parser.add_argument("--use_rag", action="store_true", help="Use RAG in test generation.")
    parser.add_argument("--use_diverse_sampling", action="store_true", help="Use diverse sampling (cc single-turn).")
    parser.add_argument("--use_repair", action="store_true", help="Use repair operator.")
    parser.add_argument("--store_turns_details", action="store_true", help="Store turn-level results.")

    return parser.parse_args()


def run_navi_multi_turn(args):
    from examples.navi.fitness_mt import (
        NaviFitnessConversationEffectiveness,
        NaviFitnessConversationEfficiency,
        NaviFitnessConversationValidationDimensions,
    )
    from llm.adapter.embeddings_openai_adapter import get_similarity_conversation
    from llm.operators.conversation_sampling_discrete import ConversationSamplingDiscrete
    from llm.operators.conversation_mutator_discrete import ConversationMutationDiscrete
    from llm.operators.conversation_crossover_discrete import ConversationCrossoverDiscrete
    from llm.operators.conversation_duplicates import ConversationDuplicateEliminationVars
    from llm.operators.conversation_repair import ConversationRepairConversationGenerator, NoConversationRepair
    from examples.navi.navi_conversation_generator import NaviConversationGenerator
    from llm.features import FeatureHandler
    from llm.model.search_configuration import MultiTurnSearchConfiguration, MultiTurnSearchOperators
    from llm.sut.ipa import IPA
    from llm.sut.ipa_yelp import IPA_YELP
    from llm.sut.ipa_los import IPA_LOS
    from llm.mt_navi_runs_utils import save_results_to_json
    import llm.config as llm_config

    if args.sut is None:
        args.sut = "ipa_yelp"
    if args.features_config is None:
        args.features_config = "configs/features_navi_mt.json"
    if args.wandb_project is None:
        args.wandb_project = "MultiTurnTestDiscrete"
    if args.archive_threshold is None:
        args.archive_threshold = 0.15
    if args.max_time is None:
        args.max_time = "00:01:30"
    if args.llm_intent_classifier is None:
        args.llm_intent_classifier = "gpt-4o-mini"

    if args.llm_ipa:
        llm_config.LLM_IPA = args.llm_ipa
        os.environ["LLM_IPA"] = args.llm_ipa
        print(f"Set LLM_IPA to {args.llm_ipa}")
    if args.llm_intent_classifier:
        llm_config.LLM_CLASSIFIER = args.llm_intent_classifier
        os.environ["LLM_CLASSIFIER"] = args.llm_intent_classifier
        print(f"Set LLM_CLASSIFIER to {args.llm_intent_classifier}")
    if args.llm_judge:
        llm_config.LLM_VALIDATOR = args.llm_judge
        os.environ["LLM_VALIDATOR"] = args.llm_judge
        print(f"Set LLM_VALIDATOR to {args.llm_judge}")
    if args.llm_generator:
        llm_config.LLM_GENERATOR = args.llm_generator
        os.environ["LLM_GENERATOR"] = args.llm_generator
        print(f"Set LLM_GENERATOR to {args.llm_generator}")

    feature_handler = FeatureHandler.from_json(args.features_config)

    def _create_problem_name(args, config) -> str:
        return (
            f"{args.algorithm.upper()}"
            + f"_{args.sut}"
            + f"_{config.population_size}n"
            + (f"_{config.n_generations}i" if config.n_generations is not None else "")
            + (
                f"_{config.maximal_execution_time.replace(':', '_')}t"
                if config.maximal_execution_time is not None
                else ""
            )
            + f"_{args.seed}seed"
            + "_Discrete"
            + f"_gen-{args.llm_generator}"
            + f"_judge-{args.llm_judge}"
        )

    operators = MultiTurnSearchOperators(
        crossover=ConversationCrossoverDiscrete(generate_conversation=not args.use_repair),
        sampling=ConversationSamplingDiscrete(generate_conversation=not args.use_repair, variable_length=True),
        mutation=ConversationMutationDiscrete(generate_conversation=not args.use_repair),
        duplicate_elimination=ConversationDuplicateEliminationVars(),
        repair=ConversationRepairConversationGenerator() if args.use_repair else NoConversationRepair(),
    )

    config = MultiTurnSearchConfiguration(operators=operators)
    config.population_size = args.n
    config.n_generations = args.i
    config.archive_threshold = args.archive_threshold
    config.maximal_execution_time = (
        None if (args.max_time is None or str(args.max_time).lower() in ("none", "")) else args.max_time
    )
    config.n_repopulate_max = 0.2
    config.results_folder = RESULTS_FOLDER

    sut_map = {
        "openai": IPA.simulate_conversation,
        "openai_los": IPA_LOS.simulate_conversation,
        "ipa_yelp": IPA_YELP.simulate_conversation,
    }
    if args.sut not in sut_map:
        raise ValueError(f"Invalid SUT: {args.sut}. Valid: {list(sut_map)}")
    simulate_function = sut_map[args.sut]

    fitness = FitnessMerged([
        NaviFitnessConversationValidationDimensions(weights=[args.weight_clarity, args.weight_request_orientedness]),
        NaviFitnessConversationEfficiency(),
        NaviFitnessConversationEffectiveness(),
    ])

    critical = CriticalMerged(
        fitness_names=fitness.name,
        criticals=[
            (CriticalByFitnessThreshold(mode="<", score=args.th_dims), ["dimensions_fitness"]),
            (CriticalByFitnessThreshold(mode="<", score=args.th_efficiency), ["efficiency_fitness"]),
            (CriticalByFitnessThreshold(mode="<", score=args.th_effectiveness), ["effectiveness_fitness"]),
        ],
        mode="or",
    )

    problem = QAProblem(
        problem_name="MT_Test",
        scenario_path=os.getcwd(),
        xl=[0],
        xu=[1],
        simulation_variables=["conversation"],
        fitness_function=fitness,
        critical_function=critical,
        simulate_function=simulate_function,
        context={
            "location": {
                "position": [48.2628, 11.6687],
                "address": "Am Parkring, Munich, Germany",
                "data": "2025-03-19T0",
                "time": "09:00:00",
            },
            "person": {"gender": "male", "age": 51},
        },
        seed=args.seed,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        max_repeats=args.max_repeats,
    )

    problem.feature_handler = feature_handler
    problem.conversation_generator = NaviConversationGenerator(feature_handler=feature_handler)
    problem.problem_name = _create_problem_name(args, config)

    tags = [f"{k}:{v}" for k, v in vars(args).items() if k not in ["features_config", "save_folder"]]

    if args.use_wandb:
        weave.init(args.wandb_project)
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=problem.problem_name,
            group=datetime.now().strftime("%d-%m-%Y"),
            tags=tags,
        )
    else:
        wandb.init(mode="disabled")

    if args.algorithm == "nsga2":
        optimizer = NsgaIIOptimizer(problem=problem, config=config, callback=logging_callback_archive)
    elif args.algorithm in {"nsga2d", "nsga2ds"}:
        optimizer = NSGAIIDOptimizer(
            problem=problem, config=config, callback=logging_callback_archive,
            dist_function=get_similarity_conversation,
        )
    elif args.algorithm == "rs":
        optimizer = PureSamplingRand(problem=problem, config=config, callback=logging_callback_archive)
    else:
        raise ValueError("Algorithm not known.")

    res = optimizer.run()

    if args.store_turns_details:
        save_results_to_json(res, args, problem, output_dir=optimizer.save_folder)

    res.write_results(results_folder=optimizer.save_folder, params=optimizer.parameters, search_config=config)
    log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")


def run_cc_multi_turn(args):
    from examples.car_control.fitness_mt import (
        CCFitnessConversationEffectiveness,
        CCFitnessConversationEfficiency,
        CCFitnessConversationValidationDimensions,
    )
    from llm.adapter.embeddings_openai_adapter import get_similarity_conversation
    from llm.operators.conversation_sampling_discrete import ConversationSamplingDiscrete
    from llm.operators.conversation_mutator_discrete import ConversationMutationDiscrete
    from llm.operators.conversation_crossover_discrete import ConversationCrossoverDiscrete
    from llm.operators.conversation_duplicates import ConversationDuplicateEliminationVars
    from llm.operators.conversation_repair import ConversationRepairConversationGenerator, NoConversationRepair
    from examples.car_control.cc_conversation_generator import CCConversationGenerator
    from llm.features import FeatureHandler
    from llm.model.search_configuration import MultiTurnSearchConfiguration, MultiTurnSearchOperators
    from llm.sut.ipa import IPA
    from llm.sut.ipa_yelp_cc import IPA_YELP
    from llm.sut.ipa_los import IPA_LOS
    from llm.mt_navi_runs_utils import save_results_to_json
    import llm.config as llm_config

    if args.sut is None:
        args.sut = "ipa_yelp"
    if args.features_config is None:
        args.features_config = "configs/features_cc_mt.json"
    if args.wandb_project is None:
        args.wandb_project = "MultiTurnTestDiscrete"
    if args.archive_threshold is None:
        args.archive_threshold = 0.15
    if args.max_time is None:
        args.max_time = "00:01:30"
    if args.llm_intent_classifier is None:
        args.llm_intent_classifier = "gpt-4o"

    if args.llm_ipa:
        llm_config.LLM_IPA = args.llm_ipa
        os.environ["LLM_IPA"] = args.llm_ipa
        print(f"Set LLM_IPA to {args.llm_ipa}")
    if args.llm_intent_classifier:
        llm_config.LLM_CLASSIFIER = args.llm_intent_classifier
        os.environ["LLM_CLASSIFIER"] = args.llm_intent_classifier
        print(f"Set LLM_CLASSIFIER to {args.llm_intent_classifier}")
    if args.llm_judge:
        llm_config.LLM_VALIDATOR = args.llm_judge
        os.environ["LLM_VALIDATOR"] = args.llm_judge
        print(f"Set LLM_VALIDATOR to {args.llm_judge}")
    if args.llm_generator:
        llm_config.LLM_GENERATOR = args.llm_generator
        os.environ["LLM_GENERATOR"] = args.llm_generator
        print(f"Set LLM_GENERATOR to {args.llm_generator}")

    feature_handler = FeatureHandler.from_json(args.features_config)

    def _create_problem_name(args, config) -> str:
        return (
            f"{args.algorithm.upper()}"
            + f"_{args.sut}"
            + f"_{args.llm_ipa}"
            + f"_{config.population_size}n"
            + (f"_{config.n_generations}i" if config.n_generations is not None else "")
            + (
                f"_{config.maximal_execution_time.replace(':', '_')}t"
                if config.maximal_execution_time is not None
                else ""
            )
            + f"_{args.seed}seed"
        )

    operators = MultiTurnSearchOperators(
        crossover=ConversationCrossoverDiscrete(generate_conversation=not args.use_repair),
        sampling=ConversationSamplingDiscrete(generate_conversation=not args.use_repair, variable_length=True),
        mutation=ConversationMutationDiscrete(generate_conversation=not args.use_repair),
        duplicate_elimination=ConversationDuplicateEliminationVars(),
        repair=ConversationRepairConversationGenerator() if args.use_repair else NoConversationRepair(),
    )

    config = MultiTurnSearchConfiguration(operators=operators)
    config.population_size = args.n
    config.n_generations = args.i
    config.archive_threshold = args.archive_threshold
    config.maximal_execution_time = (
        None if (args.max_time is None or str(args.max_time).lower() in ("none", "")) else args.max_time
    )
    config.n_repopulate_max = 0.2
    config.results_folder = args.save_folder

    sut_map = {
        "openai": IPA.simulate_conversation,
        "openai_los": IPA_LOS.simulate_conversation,
        "ipa_yelp": IPA_YELP.simulate_conversation,
    }
    if args.sut not in sut_map:
        raise ValueError(f"Invalid SUT: {args.sut}. Valid: {list(sut_map)}")
    simulate_function = sut_map[args.sut]

    fitness = FitnessMerged([
        CCFitnessConversationValidationDimensions(weights=[args.weight_clarity, args.weight_request_orientedness]),
        CCFitnessConversationEfficiency(),
        CCFitnessConversationEffectiveness(),
    ])

    critical = CriticalMerged(
        fitness_names=fitness.name,
        criticals=[
            (CriticalByFitnessThreshold(mode="<", score=args.th_dims), ["dimensions_fitness"]),
            (CriticalByFitnessThreshold(mode="<", score=args.th_efficiency), ["efficiency_fitness"]),
            (CriticalByFitnessThreshold(mode="<", score=args.th_effectiveness), ["effectiveness_fitness"]),
        ],
        mode="or",
    )

    problem = QAProblem(
        problem_name="MT_Test",
        scenario_path=os.getcwd(),
        xl=[0],
        xu=[1],
        simulation_variables=["conversation"],
        fitness_function=fitness,
        critical_function=critical,
        simulate_function=simulate_function,
        context={
            "location": {
                "position": [48.2628, 11.6687],
                "address": "Am Parkring, Munich, Germany",
                "data": "2025-03-19T0",
                "time": "09:00:00",
            },
            "person": {"gender": "male", "age": 51},
        },
        seed=args.seed,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        max_repeats=args.max_repeats,
    )

    problem.feature_handler = feature_handler
    problem.conversation_generator = CCConversationGenerator(feature_handler=feature_handler)
    problem.problem_name = _create_problem_name(args, config)

    tags = [f"{k}:{v}" for k, v in vars(args).items() if k not in ("features_config", "save_folder")]

    if args.use_wandb:
        weave.init(args.wandb_project)
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=problem.problem_name,
            group=datetime.now().strftime("%d-%m-%Y"),
            tags=tags,
        )
    else:
        wandb.init(mode="disabled")

    if args.algorithm == "nsga2":
        optimizer = NsgaIIOptimizer(problem=problem, config=config, callback=logging_callback_archive)
    elif args.algorithm in {"nsga2d", "nsga2ds"}:
        optimizer = NSGAIIDOptimizer(
            problem=problem, config=config, callback=logging_callback_archive,
            dist_function=get_similarity_conversation,
        )
    elif args.algorithm == "rs":
        optimizer = PureSamplingRand(problem=problem, config=config, callback=logging_callback_archive)
    else:
        raise ValueError("Algorithm not known.")

    res = optimizer.run()

    if args.store_turns_details:
        save_results_to_json(res, args, problem, output_dir=optimizer.save_folder)

    res.write_results(results_folder=optimizer.save_folder, params=optimizer.parameters, search_config=config)
    log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")



def run_navi_single_turn(args):
    from llm.adapter.embeddings_local_adapter import get_disimilarity_individual
    from llm.operators.utterance_repair import NoUtteranceRepair, UtteranceRepairQuestionGenerator
    from llm.model.search_configuration import QASearchConfiguration, QASearchOperators
    from llm.sut.ipa import IPA
    from llm.sut.ipa_los import IPA_LOS
    from llm.sut.ipa_yelp import IPA_YELP
    from examples.navi.navi_utterance_generator import NaviUtteranceGenerator
    from llm.eval.fitness import FitnessDiverse
    from examples.navi.fitness import NaviFitnessAnswerValidationDimensions, NaviFitnessContentComparison
    from llm.operators.utterance_crossover_discrete import UtteranceCrossoverDiscrete
    from llm.operators.utterance_sampling_discrete import (
        UtteranceSamplingDiscrete,
        UtteranceSamplingDiscreteDiverse,
        UtteranceSamplingGrid,
    )
    from llm.operators.utterance_mutator_discrete import UtteranceMutationDiscrete
    from llm.operators.utterance_duplicates_discrete import UtteranceDuplicateEliminationLocalDiscreteWithContent
    from llm.utils.name import create_problem_name

    if args.sut is None:
        args.sut = "IPA_LOS"
    if args.features_config is None:
        args.features_config = "configs/navi_features.json"
    if args.wandb_project is None:
        args.wandb_project = "ConvLLMFinal"
    if args.archive_threshold is None:
        args.archive_threshold = 0.1
    if args.th_answer is None:
        args.th_answer = 0.75
    if args.th_content is None:
        args.th_content = 0.75

    warnings.filterwarnings("ignore", category=FutureWarning, message=".*encoder_attention_mask.*")

    SUT_MAP = {
        "IPA_LOS": IPA_LOS,
        "IPA_YELP": IPA_YELP,
    }
    if args.sut not in SUT_MAP:
        raise ValueError(f"Invalid SUT: {args.sut}. Valid: {list(SUT_MAP)}")
    SUT_CLASS = SUT_MAP[args.sut]

    search_operators = QASearchOperators(
        crossover=UtteranceCrossoverDiscrete(generate_question=not args.use_repair),
        sampling=(
            UtteranceSamplingGrid(total_samples=args.n, t=4)
            if args.algorithm == "gs"
            else (
                UtteranceSamplingDiscreteDiverse(
                    generate_question=not args.use_repair,
                )
                if args.use_diverse_sampling or args.algorithm == "nsga2ds"
                else UtteranceSamplingDiscrete(generate_question=not args.use_repair)
            )
        ),
        mutation=UtteranceMutationDiscrete(generate_question=not args.use_repair),
        duplicate_elimination=UtteranceDuplicateEliminationLocalDiscreteWithContent(),
        repair=UtteranceRepairQuestionGenerator() if args.use_repair else NoUtteranceRepair(),
    )

    config = QASearchConfiguration(operators=search_operators)
    config.population_size = args.n
    config.n_generations = args.i
    config.maximal_execution_time = args.max_time
    config.n_repopulate_max = 0.2
    config.results_folder = args.save_folder
    config.archive_threshold = args.archive_threshold

    fitness = FitnessMerged([
        NaviFitnessAnswerValidationDimensions(
            weights=args.judge_weights,
            llm_type=LLMType(args.llm_judge),
        ),
        NaviFitnessContentComparison(),
        FitnessDiverse(),
    ])

    critical = CriticalMerged(
        fitness_names=fitness.name,
        criticals=[
            (CriticalByFitnessThreshold(mode="<", score=args.th_answer), ["answer_fitness"]),
            (CriticalByFitnessThreshold(mode="<", score=args.th_content), ["content_fitness"]),
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
        ),
    )

    tags = [f"{k}:{v}" for k, v in vars(args).items() if k != "features_config"]

    if args.use_wandb:
        weave.init(args.wandb_project)
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=problem_name,
            group=datetime.now().strftime("%d-%m-%Y"),
            tags=tags,
        )
    else:
        wandb.init(mode="disabled")

    problem = QAProblem(
        problem_name="Test",
        scenario_path=os.getcwd(),
        xl=[0],
        xu=[1],
        simulation_variables=["utterance"],
        fitness_function=fitness,
        critical_function=critical,
        simulate_function=simulate_function,
        seed_utterances=[
            "hmm I need some food",
            "yeah I need some burgers",
            "oh some burger would be great",
        ],
        context={
            "location": {
                "position": [48.2628, 11.6687],
                "address": "Am Parkring, Munich, Germany",
                "date": "2025-03-19T0",
                "time": "09:00:00",
            },
            "person": {"gender": "female", "age": 30},
        },
        seed=seed,
        names_dim_utterance=["utterance"],
        feature_handler_config_path=args.features_config,
        question_generator=NaviUtteranceGenerator(use_rag=args.use_rag),
    )
    problem.problem_name = problem_name

    if args.algorithm == "nsga2":
        optimizer = NsgaIIOptimizer(problem=problem, config=config, callback=logging_callback_archive)
    elif args.algorithm in {"nsga2d", "nsga2ds"}:
        optimizer = NSGAIIDOptimizer(
            problem=problem, config=config, callback=logging_callback_archive,
            dist_function=get_disimilarity_individual,
        )
    elif args.algorithm in {"rs", "gs"}:
        optimizer = PureSamplingRand(problem=problem, config=config, callback=logging_callback_archive)
    else:
        raise ValueError("Algorithm not known.")

    res = optimizer.run()
    res.write_results(results_folder=optimizer.save_folder, params=optimizer.parameters, search_config=config)



def run_cc_single_turn(args):
    from llm.adapter.embeddings_local_adapter import get_disimilarity_individual
    from llm.operators.utterance_repair import NoUtteranceRepair, UtteranceRepairQuestionGenerator
    from llm.model.search_configuration import QASearchConfiguration, QASearchOperators
    from llm.sut.ipa import IPA
    from llm.sut.ipa_los import IPA_LOS
    from llm.sut.ipa_yelp_cc import IPA_YELP
    from examples.car_control.cc_utterance_generator import CCUtteranceGenerator
    from llm.eval.fitness import FitnessDiverse
    from llm.eval.utterances_distance import get_question_distance
    from examples.car_control.fitness import (
        CCFitnessContentComparison,
        CCFitnessAnswerValidationDimensions,
    )
    from llm.operators.utterance_crossover_discrete import UtteranceCrossoverDiscrete
    from llm.operators.utterance_sampling_discrete import (
        UtteranceSamplingDiscrete,
        UtteranceSamplingGrid,
        UtteranceSamplingDiscreteDiverse,
    )
    from llm.operators.utterance_mutator_discrete import UtteranceMutationDiscrete
    from llm.operators.utterance_duplicates_discrete import UtteranceDuplicateEliminationLocalDiscreteWithContent
    from llm.utils.name import create_problem_name

    if args.sut is None:
        args.sut = "IPA_YELP"
    if args.features_config is None:
        args.features_config = "configs/features_cc_st.json"
    if args.wandb_project is None:
        args.wandb_project = "CarControlShort"
    if args.archive_threshold is None:
        args.archive_threshold = 0.1
    if args.th_answer is None:
        args.th_answer = 0.6
    if args.th_content is None:
        args.th_content = 0.85

    warnings.filterwarnings("ignore", category=FutureWarning, message=".*encoder_attention_mask.*")

    SUT_MAP = {
        "IPA_LOS": IPA_LOS,
        "IPA_YELP": IPA_YELP,
    }
    if args.sut not in SUT_MAP:
        raise ValueError(f"Invalid SUT: {args.sut}. Valid: {list(SUT_MAP)}")
    SUT_CLASS = SUT_MAP[args.sut]

    search_operators = QASearchOperators(
        crossover=UtteranceCrossoverDiscrete(
            llm_type=LLMType(args.llm_generator),
            generate_question=not args.use_repair,
        ),
        sampling=(
            UtteranceSamplingGrid(
                llm_type=LLMType(args.llm_generator),
                total_samples=args.n,
                t=2,
            )
            if args.algorithm == "gs"
            else (
                UtteranceSamplingDiscreteDiverse(
                    llm_type=LLMType(args.llm_generator),
                    generate_question=not args.use_repair,
                )
                if args.use_diverse_sampling or args.algorithm == "nsga2ds"
                else UtteranceSamplingDiscrete(
                    llm_type=LLMType(args.llm_generator),
                    generate_question=not args.use_repair,
                )
            )
        ),
        mutation=UtteranceMutationDiscrete(
            llm_type=LLMType(args.llm_generator),
            generate_question=not args.use_repair,
        ),
        duplicate_elimination=UtteranceDuplicateEliminationLocalDiscreteWithContent(),
        repair=(
            UtteranceRepairQuestionGenerator(llm_type=LLMType(args.llm_generator))
            if args.use_repair
            else NoUtteranceRepair()
        ),
    )

    config = QASearchConfiguration(operators=search_operators)
    config.population_size = args.n
    config.n_generations = args.i
    config.maximal_execution_time = args.max_time
    config.n_repopulate_max = 0.2
    config.results_folder = args.save_folder
    config.archive_threshold = args.archive_threshold

    if args.algorithm not in {"nsga2d", "nsga2ds"}:
        fitness = FitnessMerged([
            CCFitnessContentComparison(llm_type=LLMType(args.llm_judge)),
            CCFitnessAnswerValidationDimensions(llm_type=LLMType(args.llm_judge)),
        ])
    else:
        fitness = FitnessMerged([
            CCFitnessContentComparison(llm_type=LLMType(args.llm_judge)),
            CCFitnessAnswerValidationDimensions(llm_type=LLMType(args.llm_judge)),
            FitnessDiverse(dist_fnc=get_question_distance),
        ])

    critical = CriticalMerged(
        fitness_names=fitness.name,
        criticals=[
            (CriticalByFitnessThreshold(mode="<", score=args.th_content), ["content_fitness"]),
            (CriticalByFitnessThreshold(mode="<", score=args.th_answer), ["answer_fitness"]),
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
        ),
    )

    tags = [f"{k}:{v}" for k, v in vars(args).items() if k != "features_config"]

    if args.use_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=problem_name,
            group=datetime.now().strftime("%d-%m-%Y"),
            tags=tags,
        )
    else:
        wandb.init(mode="disabled")

    problem = QAProblem(
        problem_name="Test",
        scenario_path=os.getcwd(),
        xl=[0],
        xu=[1],
        simulation_variables=["utterance"],
        fitness_function=fitness,
        critical_function=critical,
        simulate_function=simulate_function,
        seed_utterances=[
            "hmm I need some food",
            "yeah I need some burgers",
            "oh some burger would be great",
        ],
        context={
            "location": {
                "position": [48.2628, 11.6687],
                "address": "Am Parkring, Munich, Germany",
                "date": "2025-03-19T0",
                "time": "09:00:00",
            },
            "person": {"gender": "female", "age": 30},
        },
        seed=seed,
        names_dim_utterance=["utterance"],
        feature_handler_config_path=args.features_config,
        question_generator=CCUtteranceGenerator(),
    )
    problem.problem_name = problem_name

    if args.algorithm == "nsga2":
        optimizer = NsgaIIOptimizer(problem=problem, config=config, callback=logging_callback_archive)
    elif args.algorithm in {"nsga2d", "nsga2ds"}:
        optimizer = NSGAIIDOptimizer(
            problem=problem, config=config, callback=logging_callback_archive,
            dist_function=get_disimilarity_individual,
        )
    elif args.algorithm in {"rs", "gs"}:
        optimizer = PureSamplingRand(problem=problem, config=config, callback=logging_callback_archive)
    else:
        raise ValueError("Algorithm not known.")

    res = optimizer.run()
    res.write_results(results_folder=optimizer.save_folder, params=optimizer.parameters, search_config=config)
    log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")


def run_safety_single_turn(args):
    from llm.adapter.embeddings_local_adapter import get_disimilarity_individual
    from llm.operators.utterance_repair import NoUtteranceRepair, UtteranceRepairQuestionGenerator
    from llm.model.search_configuration import QASearchConfiguration, QASearchOperators
    from llm.sut.io_simulation import IOSimulator
    from examples.safety.utterance_generator import AstralUtteranceGenerator
    from examples.safety.eval import AstralFitnessAnswerValidation, CriticalAstral
    from llm.eval.fitness import FitnessDiverse, FitnessMerged
    from llm.eval.utterances_distance import get_question_distance
    from llm.operators.utterance_crossover_discrete import UtteranceCrossoverDiscrete
    from llm.operators.utterance_sampling_discrete import (
        UtteranceSamplingDiscrete,
        UtteranceSamplingGrid,
        UtteranceSamplingDiscreteDiverse,
    )
    from llm.operators.utterance_mutator_discrete import UtteranceMutationDiscrete
    from llm.operators.utterance_duplicates import UtteranceDuplicateEliminationDistance

    if args.sut is None:
        args.sut = args.llm_ipa
    if args.features_config is None:
        args.features_config = "configs/safety_features.json"
    if args.wandb_project is None:
        args.wandb_project = "SafeLLMDiv"
    if args.archive_threshold is None:
        args.archive_threshold = 0.0

    warnings.filterwarnings("ignore", category=FutureWarning, message=".*encoder_attention_mask.*")

    search_operators = QASearchOperators(
        crossover=UtteranceCrossoverDiscrete(
            llm_type=LLMType(args.llm_generator),
            generate_question=not args.use_repair,
        ),
        sampling=(
            UtteranceSamplingGrid(
                llm_type=LLMType(args.llm_generator),
                total_samples=args.n,
                t=4,
            )
            if args.algorithm == "gs"
            else (
                UtteranceSamplingDiscreteDiverse(
                    llm_type=LLMType(args.llm_generator),
                    generate_question=not args.use_repair,
                )
                if args.use_diverse_sampling or args.algorithm == "nsga2ds"
                else UtteranceSamplingDiscrete(
                    llm_type=LLMType(args.llm_generator),
                    generate_question=not args.use_repair,
                )
            )
        ),
        mutation=UtteranceMutationDiscrete(
            llm_type=LLMType(args.llm_generator),
            generate_question=not args.use_repair,
        ),
        duplicate_elimination=UtteranceDuplicateEliminationDistance(),
        repair=(
            UtteranceRepairQuestionGenerator(llm_type=LLMType(args.llm_generator))
            if args.use_repair
            else NoUtteranceRepair()
        ),
    )

    config = QASearchConfiguration(operators=search_operators)
    config.population_size = args.n
    config.n_generations = args.i
    config.maximal_execution_time = args.max_time
    config.n_repopulate_max = 0.5
    config.results_folder = args.save_folder
    config.archive_threshold = args.archive_threshold

    if args.algorithm in {"nsga2d", "nsga2ds"}:
        fitness = FitnessMerged([
            AstralFitnessAnswerValidation(llm_type=LLMType(args.llm_judge)),
            FitnessDiverse(dist_fnc=get_question_distance),
        ])
    else:
        fitness = FitnessMerged([
            AstralFitnessAnswerValidation(llm_type=LLMType(args.llm_judge)),
        ])

    critical = CriticalAstral(llm_type=LLMType(args.llm_judge))

    simulate_function = IOSimulator(llm_type=LLMType(args.sut)).simulate
    seed = args.seed

    problem_name = (
        f"{args.sut}"
        + f"_{config.population_size}n"
        + (f"_{config.n_generations}i" if config.n_generations is not None else "")
        + (
            f"_{config.maximal_execution_time.replace(':', '_')}t"
            if config.maximal_execution_time is not None
            else ""
        )
        + f"_{seed}seed"
        + f"_{args.algorithm.upper()}"
    )

    tags = [f"{k}:{v}" for k, v in vars(args).items() if k != "features_config"]

    if args.use_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=problem_name,
            group=datetime.now().strftime("%d-%m-%Y"),
            tags=tags,
        )
    else:
        wandb.init(mode="disabled")

    problem = QAProblem(
        problem_name=problem_name,
        scenario_path=os.getcwd(),
        xl=[0],
        xu=[1],
        simulation_variables=["utterance"],
        fitness_function=fitness,
        critical_function=critical,
        simulate_function=simulate_function,
        seed_utterances=[""],
        context={},
        seed=seed,
        names_dim_utterance=["utterance"],
        feature_handler_config_path=args.features_config,
        question_generator=AstralUtteranceGenerator(use_rag=args.use_rag),
    )

    if args.algorithm == "nsga2":
        optimizer = NsgaIIOptimizer(problem=problem, config=config, callback=logging_callback_archive)
    elif args.algorithm in {"nsga2d", "nsga2ds"}:
        optimizer = NSGAIIDOptimizer(
            problem=problem, config=config, callback=logging_callback_archive,
            dist_function=get_question_distance,
        )
    elif args.algorithm in {"rs", "gs"}:
        optimizer = PureSamplingRand(problem=problem, config=config, callback=logging_callback_archive)
    else:
        raise ValueError("Algorithm not known.")

    res = optimizer.run()
    res.write_results(results_folder=optimizer.save_folder, params=optimizer.parameters, search_config=config)
    log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")


if __name__ == "__main__":
    args = parse_args()

    os.chmod(os.getcwd(), 0o777)
    setup_logging(LOG_FILE)
    disable_pymoo_warnings()

    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure").disabled = True

    dispatch = {
        ("navi", "multi-turn"): run_navi_multi_turn,
        ("cc", "multi-turn"): run_cc_multi_turn,
        ("navi", "single-turn"): run_navi_single_turn,
        ("cc", "single-turn"): run_cc_single_turn,
        ("safety", "single-turn"): run_safety_single_turn,
    }

    fn = dispatch[(args.case_study, args.mode)]
    fn(args)
