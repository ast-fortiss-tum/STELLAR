import copy
import random
from typing import Any, Dict, List, Optional

from pymoo.core.mutation import Mutation
from pymoo.operators.mutation.pm import PolynomialMutation

from llm.model.models import Conversation
from llm.model.qa_problem import QAProblem
from llm.operators.utterance_mutator_discrete import ChoiceMutation

from .base_classes import ConversationMutationBase


class ConversationMutationDiscrete(ConversationMutationBase):
    call_counter = 0
    def __init__(
        self,
        mut_prob=0.9,
        generate_conversation: bool = False
    ):
        super().__init__()
        self.mut_prob = mut_prob
        self.generate_conversation = generate_conversation
        self.poly = PolynomialMutation(prob=self.mut_prob, eta=5)
        self.rm = ChoiceMutation(prob=self.mut_prob)

    def _instance_mutation(
        self, problem: QAProblem, conversations: List[Conversation]
    ) -> List[Conversation]:
        new_ordinal_vars = self._ordinal_vars_mutation(problem, conversations, self.poly)
        new_categorical_vars = self._categorical_vars_mutation(problem, conversations, self.rm)
        new_continuous_vars = self._continuous_vars_mutation(problem, conversations, self.poly)
        self.call_counter = self.call_counter + 1
        
        # print(f"{self.call_counter} mutation calls")
        
        result = []
        for i in range(len(conversations)):
            result.append(self._build_conversation(
                problem=problem,
                ordinal_vars=new_ordinal_vars[i],
                categorical_vars=new_categorical_vars[i],
                continuous_vars=new_continuous_vars[i],
            ))

        return result
