from typing import List

import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.operators.mutation.pm import PolynomialMutation

from llm.config import LLM_MUTATOR
from llm.features.models import Feature
from llm.llms import LLMType
from llm.model.models import Utterance
from llm.model.qa_problem import QAProblem
from llm.prompts import MUTATION_PROMPT_GENERIC

from .base_classes import UtteranceMutationBase

class ChoiceMutation(Mutation):
    def _do(self, problem: QAProblem, X, **kwargs):
        prob_var = min(0.5, 1 / len(problem.feature_handler.categorical_features))
        for k, feature in enumerate(problem.feature_handler.categorical_features.values()):
            mut = np.where(np.random.random(len(X)) < prob_var)[0]
            X[mut, k] = np.random.choice(feature.num_values, len(mut))

        return X


class UtteranceMutationDiscrete(UtteranceMutationBase):
    call_counter = 0
    def __init__(
        self,
        mut_prob=0.9,
        temperature=0.3,
        prompt=MUTATION_PROMPT_GENERIC,
        llm_type=LLMType(LLM_MUTATOR),
        generate_question: bool = True,
    ):  # the last length/2 values are seg length values
        super().__init__()
        self.mut_prob = mut_prob
        self.temperature = temperature
        self.prompt = prompt
        self.llm_type = llm_type
        self.generate_question = generate_question
        self.poly = PolynomialMutation(prob=self.mut_prob, eta=30)
        self.rm = ChoiceMutation(prob=self.mut_prob)

    def _instance_mutation(
        self, problem: QAProblem, utterances: List[Utterance]
    ) -> List[Utterance]:
        new_ordinal_vars = self._ordinal_vars_mutation(problem, utterances, self.poly)
        new_categorical_vars = self._categoricals_vars_mutation(problem, utterances, self.rm)
        self.call_counter = self.call_counter + 1
        print(f"{self.call_counter} mutation calls")
        result = []
        for i in range(len(utterances)):
            result.append(self._build_utterance(
                problem=problem,
                seed=utterances[i].question,
                ordinal_vars=new_ordinal_vars[i],
                categorical_vars=new_categorical_vars[i],
            ))

        return result
