import random
from typing import List

import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem

from llm.config import DEBUG, LLM_MUTATOR
from llm.llms import LLMType, pass_llm
from llm.model.models import Utterance
from llm.prompts import (MUTATION_PROMPT_FAULT, MUTATION_PROMPT_GENERIC,
                         MUTATION_PROMPT_IMPLICIT)

from .base_classes import UtteranceMutationBase

MUTATION_TYPES = [
    "SynonymReplacement",  # Replace a verb, adjective, adverb, or noun with a synonym.
    "ModifierInsertion",  # Add an adjective or adverb to introduce redundancy or emphasis.
    "SentenceExpansion",  # Add clarifying details without changing the core meaning.
    "VoiceTransformation",  # Convert active voice to passive or vice versa.
    "DomainInstanceVariation",  # Change specific attributes (e.g., landmarks, directions) within the same domain.
]


class UtteranceMutation(UtteranceMutationBase):
    def __init__(
        self,
        mut_prob=0.9,
        temperature=0.3,
        prompt=MUTATION_PROMPT_GENERIC,
        llm_type=LLMType(LLM_MUTATOR),
    ):  # the last length/2 values are seg length values
        super().__init__()
        self.mut_prob = mut_prob
        self.temperature = temperature
        self.prompt = prompt
        self.llm_type = llm_type

    def _instance_mutation(
        self, problem: Problem, utterances: List[Utterance]
    ) -> List[Utterance]:
        result = []
        for utterance in utterances:
            mutated_utterance = Utterance(
                question=self.llm_mutator(
                    question=utterance.question, rate=self.mut_prob
                ),
                answer=None,
            )
            result.append(mutated_utterance)
        return result

    def llm_mutator(self, question, rate=0.5, temperature=None):
        mutate_prompt = MUTATION_PROMPT_GENERIC
        if not DEBUG and random.random() < rate:
            mutation_type = random.choice(MUTATION_TYPES)
            prompt = mutate_prompt.format(mutation_type, question)
            return pass_llm(
                prompt,
                temperature=temperature,
                llm_type=self.llm_type,
            )
        else:
            return question


class UtteranceMutationImplicit(UtteranceMutation):
    def __init__(
        self, mut_prob=0.7, temperature=0.3, prompt=MUTATION_PROMPT_IMPLICIT
    ):  # the last length/2 values are seg length values
        super().__init__(mut_prob, temperature, prompt)


class UtteranceMutationFault(UtteranceMutation):
    def __init__(
        self, mut_prob=0.6, temperature=0.3, prompt=MUTATION_PROMPT_FAULT
    ):  # the last length/2 values are seg length values
        super().__init__(mut_prob, temperature, prompt)


if __name__ == "__main__":
    data = [
        "I am quite hungry.",
        "I need to rest.",
        "It is time to buy a new phone.",
        "Would be good to get to the next bus stop.",
        "Have not seen my friend very long time.",
        "I am running out of gas.",
        "I forgot to get some cash.",
        "I need to by some food for my dog.",
    ]
    print("First iter")
    data_1 = []
    for question in data:
        mut = UtteranceMutation()
        res = mut.llm_mutator(question, rate=1, temperature=0)
        data_1.append(res)
        print(res)
    print("2nd iter")
    data_2 = []
    for question in data_1:
        mut = UtteranceMutation()
        res = mut.llm_mutator(question, rate=1, temperature=0)
        data_2.append(res)
        print(res)
