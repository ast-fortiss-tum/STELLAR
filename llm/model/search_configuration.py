import dataclasses
import pydantic

from opensbt.experiment.search_configuration import SearchConfiguration, SearchOperators
from llm.operators.base_classes import (
    UtteranceCrossoverBase,
    UtteranceMutationBase,
    UtteranceSamplingBase,
    UtteranceDuplicateEliminationBase,
    UtteranceRepairBase,
)
from llm.operators.utterance_crossover import NoUtteranceCrossover
from llm.operators.utterance_mutator import UtteranceMutation
from llm.operators.utterance_sampling import UtteranceSampling
from llm.operators.utterance_duplicates import UtteranceDuplicateElimination
from llm.operators.utterance_repair import NoUtteranceRepair


class QASearchOperators(SearchOperators):
    crossover: UtteranceCrossoverBase = NoUtteranceCrossover()
    sampling: UtteranceSamplingBase = UtteranceSampling()
    duplicate_elimination: UtteranceDuplicateEliminationBase = UtteranceDuplicateElimination()
    mutation: UtteranceMutationBase = UtteranceMutation()
    repair: UtteranceRepairBase = NoUtteranceRepair()


class QASearchConfiguration(SearchConfiguration):
    operators: QASearchOperators = QASearchOperators()
