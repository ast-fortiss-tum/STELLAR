from typing import List

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX

from llm.config import LLM_CROSSOVER
from llm.llms import LLMType
from llm.model.models import Utterance
from llm.model.qa_problem import QAProblem

from .base_classes import UtteranceCrossoverBase

class NoUtteranceCrossoverDiscrete(UtteranceCrossoverBase):
    def __init__(self):
        # Takes 2 parents, produces 2 offspring (unchanged)
        super().__init__(2, 2)

    def _utterance_crossover(
        self, problem, matings: List[List[Utterance]]
    ) -> List[List[Utterance]]:
        return matings


class UtteranceCrossoverDiscrete(UtteranceCrossoverBase):
    call_counter = 0
    def __init__(
        self, crossover_rate=0.7, temperature=0.3, llm_type=LLMType(LLM_CROSSOVER),
        generate_question: bool = True,
    ):
        super().__init__(2, 2)
        self.crossover_rate = crossover_rate
        self.temperature = temperature
        self.llm_type = llm_type
        self.generate_question = generate_question
        self.sbx = SBX(
            prob=self.crossover_rate, prob_var=self.crossover_rate, eta=30, vtype=float
        )
        self.ux = UX()

    def _instance_crossover(
        self, problem: QAProblem, matings: List[List[Utterance]]
    ) -> List[List[Utterance]]:        
        offspring_ordinal_vars = self._ordinal_vars_crossover(problem, matings, self.sbx)
        offspring_categorical_vars = self._categorical_vars_crossover(problem, matings, self.ux)

        result: List[List[Utterance]] = []
        for _ in matings:
            result.append([])
        for i in range(self.n_offsprings):
            for j in range(len(matings)):
                utterance = self._build_utterance(
                    problem=problem,
                    seed=matings[j][i].seed,
                    ordinal_vars=offspring_ordinal_vars[j][i],
                    categorical_vars=offspring_categorical_vars[j][i]
                )
                result[j].append(utterance)
        return result
