import logging as log
import random
from typing import List

import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling

from llm.llms import LLMType, pass_llm
from llm.model.models import Utterance
from llm.prompts import SAMPLING_PROMPT

from .base_classes import UtteranceSamplingBase


class UtteranceSampling(UtteranceSamplingBase):
    def __init__(self, variable_length=False, llm_type=LLMType.GPT_4O):
        super().__init__()
        self.variable_length = variable_length
        self.llm_type = llm_type

    def _sample_instances(
        self, problem: Problem, n_samples: int, **kwargs
    ) -> List[Utterance]:
        if n_samples > len(problem.seed_utterances):
            utters_raw = problem.seed_utterances  # use the examples we have already
            utters_raw = utters_raw + llm_sample(
                questions=problem.seed_utterances,
                context=problem.context,
                n=n_samples - len(problem.seed_utterances),
                llm_type=self.llm_type,
            )
        else:
            utters_raw = random.sample(
                problem.seed_utterances, k=n_samples
            )  # use the examples we have already
        result = [Utterance(question=q) for q in utters_raw]
        return result


def llm_sample(
    questions,
    context,
    n=20,
    temperature=None,
    max_repeat=5,
    llm_type=LLMType.GPT_4O,
) -> List[Utterance]:
    log.info(f"QUESTIONS: {questions}")
    prompt = SAMPLING_PROMPT
    processed = []
    repeat = 0
    while len(processed) != n and repeat < max_repeat:
        answer = pass_llm(
            prompt.format(n, context, n, questions),
            temperature=temperature,
            max_tokens=1000,
            llm_type=llm_type,
        )
        processed = answer.splitlines()
        log.info("Repeating init sampling.")
        log.info(f"Last response: {processed}")

        if len(processed) >= n:
            processed = processed[:n]
        else:
            # fill up missing or repeat all again
            answer = pass_llm(
                prompt.format(
                    n - len(processed), context, n - len(processed), questions
                ),
                temperature=temperature,
                max_tokens=1000,
                llm_type=llm_type,
            )

            processed_extra = answer.splitlines()
            processed = processed + processed_extra
            log.info("Added extra samples.")
            log.info(f"Last response: {processed}")
    utterances = []
    for utter in processed:
        if len(utter) != 0:
            utterances.append(utter)

    return utterances


if __name__ == "__main__":
    print(llm_sample(question="I have strong headaches.", context="{}", n=5))
