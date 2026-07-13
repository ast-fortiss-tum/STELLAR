import logging as log
import random
from typing import List

import numpy as np
from pymoo.core.crossover import Crossover

from llm.config import DEBUG, LLM_CROSSOVER
from llm.llms import LLMType, pass_llm
from llm.model.models import Utterance
from llm.prompts import CROSSOVER_PROMPT

from .base_classes import UtteranceCrossoverBase


class NoUtteranceCrossover(UtteranceCrossoverBase):
    def __init__(self):
        # Takes 2 parents, produces 2 offspring (unchanged)
        super().__init__(2, 2)

    def _instance_crossover(
        self, problem, matings: List[List[Utterance]]
    ) -> List[List[Utterance]]:
        return matings


class UtteranceCrossover(UtteranceCrossoverBase):
    def __init__(
        self, crossover_rate=0.7, temperature=0.3, llm_type=LLMType(LLM_CROSSOVER)
    ):
        super().__init__(2, 2)
        self.crossover_rate = crossover_rate
        self.temperature = temperature
        self.llm_type = llm_type

    def _instance_crossover(
        self, problem, matings: List[List[Utterance]]
    ) -> List[List[Utterance]]:
        result: List[List[Utterance]] = []
        for mating in matings:
            new_questions = llm_crossover(
                mating[0].question,
                mating[1].question,
                temperature=self.temperature,
                rate=self.crossover_rate,
                llm_type=self.llm_type,
            )
            result.append([Utterance(question=q) for q in new_questions])
        return result


def llm_crossover(
    utterance1,
    utterance2,
    temperature=0,
    rate=0.7,
    llm_type=LLMType(LLM_CROSSOVER),
):
    if random.random() < rate and not DEBUG:
        answer = pass_llm(
            CROSSOVER_PROMPT.format(utterance1, utterance2),
            temperature=temperature,
            llm_type=llm_type,
        )
        processed = [resp.strip().strip('"') for resp in answer.split("\n")]
        # answer.splitlines()#re.findall(r'"(.*?)"', answer)
        utterances = []
        for utter in processed:
            if len(utter) != 0:
                utterances.append(utter)
        if len(utterances) != 2:
            return utterance1, utterance2
        return utterances[0], utterances[1]
    else:
        return utterance1, utterance2


if __name__ == "__main__":
    pairs = [
        ("I need to buy new clothes.", "I think my clothes are outdated."),
        ("It is time for somehting new to wear.", "What about new clothes?"),
        ("I am in the mood for some food.", "I have been craving sushi all day."),
        (
            "I could really go for something hearty.",
            "I am really craving something sweet right now.",
        ),
        ("It is breakfast time.", "It is time to eat for breakfast."),
    ]
    for utterance1, utterance2 in pairs:
        print(llm_crossover(utterance1, utterance2, temperature=0.2, rate=1.0))
