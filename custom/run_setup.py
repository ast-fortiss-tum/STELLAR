"""Assemble the LUNAR search from the ``Custom*`` components.

This wires the four ingredients of a search:

* **problem**   — ties the generator, the SUT and the fitness together over the
  feature space;
* **generator** — :class:`~custom.custom_generator.CustomUtteranceGenerator`;
* **SUT**       — :meth:`~custom.custom_sut.CustomSUT.simulate`;
* **fitness / critical** — :class:`~custom.custom_fitness.CustomFitness` and the
  threshold that marks a test as a failure.

The generator uses an LLM when credentials are configured and samples from
intent-specific fallback pools otherwise. Static perturbations preserve the
offline behavior, so the full tutorial can always run locally.
"""
import os
import warnings

from opensbt.utils.log_utils import log, setup_logging, disable_pymoo_warnings
from opensbt.config import LOG_FILE, RESULTS_FOLDER

from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.algorithm.ps_rand import PureSamplingRand

from llm.model.search_configuration import QASearchConfiguration, QASearchOperators
from llm.model.qa_problem import QAProblem
from llm.operators.utterance_crossover_discrete import UtteranceCrossoverDiscrete
from llm.operators.utterance_duplicates import UtteranceDuplicateEliminationMock
from llm.operators.utterance_mutator_discrete import UtteranceMutationDiscrete
from llm.operators.utterance_repair import NoUtteranceRepair
from llm.operators.utterance_sampling_discrete import (
    UtteranceSamplingDiscrete,
    UtteranceSamplingGrid,
)
from llm.eval.critical import CriticalMerged, CriticalByFitnessThreshold
from llm.llms import LLMType

from custom.custom_fitness import CustomFitness
from custom.custom_generator import CustomUtteranceGenerator
from custom.custom_sut import CustomSUT
from custom.presets import Hyperparameters

import wandb


def configure_runtime(hp: Hyperparameters) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    setup_logging(LOG_FILE)
    disable_pymoo_warnings()
    if wandb.run is None:
        init_kwargs = {"mode": hp.wandb_mode}
        if hp.wandb_project:
            init_kwargs["project"] = hp.wandb_project
        if hp.wandb_entity:
            init_kwargs["entity"] = hp.wandb_entity
        wandb.init(**init_kwargs)


def build_search_config(hp: Hyperparameters):
    generator_llm = LLMType(hp.generator_llm)

    operators = QASearchOperators(
        crossover=UtteranceCrossoverDiscrete(llm_type=generator_llm, generate_question=True),
        sampling=UtteranceSamplingDiscrete(llm_type=generator_llm, generate_question=True),
        mutation=UtteranceMutationDiscrete(llm_type=generator_llm, generate_question=True),
        duplicate_elimination=UtteranceDuplicateEliminationMock(),
        repair=NoUtteranceRepair(),
    )

    config = QASearchConfiguration(operators=operators)
    config.population_size = hp.population_size
    config.n_generations = hp.n_generations
    config.maximal_execution_time = hp.max_time
    config.results_folder = hp.results_folder or RESULTS_FOLDER

    return config


def build_fitness_and_critical():
    fitness = CustomFitness()

    critical = CriticalMerged(
        fitness_names=fitness.name,
        criticals=[
            (
                CriticalByFitnessThreshold(mode="<", score=0.5),
                ["probability_expected_intent"],
            ),
        ],
        mode="or",
    )

    return fitness, critical


def build_problem(hp: Hyperparameters, config, fitness, critical):
    problem_name = (
        f"CUSTOM_{hp.algorithm}_p{config.population_size}"
        f"_g{config.n_generations}_t-{hp.max_time}_s{hp.seed}"
    ).replace(":", "-")

    return QAProblem(
        problem_name=problem_name,
        scenario_path=os.getcwd(),
        xl=[0],
        xu=[1],
        simulation_variables=["utterance"],
        fitness_function=fitness,
        critical_function=critical,
        simulate_function=CustomSUT.simulate,
        seed_utterances=["turn on the AC"],
        context={},
        seed=hp.seed,
        names_dim_utterance=["utterance"],
        feature_handler_config_path=hp.features_config,
        question_generator=CustomUtteranceGenerator(),
    )


def build_optimizer(hp: Hyperparameters, problem, config):
    if hp.algorithm == "nsga2":
        return NsgaIIOptimizer(problem=problem, config=config)
    if hp.algorithm == "random":
        return PureSamplingRand(problem=problem, config=config)
    raise ValueError(f"Unknown algorithm: {hp.algorithm}")
