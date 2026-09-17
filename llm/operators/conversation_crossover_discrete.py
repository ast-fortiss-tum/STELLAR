from typing import List

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX

from llm.llms import LLMType
from llm.model.models import Conversation
from llm.model.qa_problem import QAProblem

from .base_classes import ConversationCrossoverBase


class NoConversationCrossoverDiscrete(ConversationCrossoverBase):
    def __init__(self):
        # Takes 2 parents, produces 2 offspring (unchanged)
        super().__init__(2, 2)

    def _conversation_crossover(
        self, problem, matings: List[List[Conversation]]
    ) -> List[List[Conversation]]:
        return matings
    

class ConversationCrossoverDiscrete(ConversationCrossoverBase):
    call_counter = 0
    def __init__(
        self, crossover_rate=0.7,
        generate_conversation: bool = False,
    ):
        super().__init__(2, 2)
        self.crossover_rate = crossover_rate
        self.generate_conversation = generate_conversation
        self.sbx = SBX(
            prob=self.crossover_rate, prob_var=self.crossover_rate, eta=5, vtype=float
        )
        self.ux = UX()

    def _instance_crossover(
        self, problem: QAProblem, matings: List[List[Conversation]]
    ) -> List[List[Conversation]]:        
        offspring_ordinal_vars = self._ordinal_vars_crossover(problem, matings, self.sbx)
        offspring_categorical_vars = self._categorical_vars_crossover(problem, matings, self.ux)
        offspring_continuous_vars = self._continuous_vars_crossover(problem, matings, self.sbx)

        result: List[List[Conversation]] = []
        for _ in matings:
            result.append([])
        for i in range(self.n_offsprings):
            for j in range(len(matings)):
                conversation = self._build_conversation(
                    problem=problem,
                    ordinal_vars=offspring_ordinal_vars[j][i],
                    categorical_vars=offspring_categorical_vars[j][i],
                    continuous_vars=offspring_continuous_vars[j][i],
                )
                result[j].append(conversation)
        return result