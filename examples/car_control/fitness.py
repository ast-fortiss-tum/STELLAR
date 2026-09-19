import json
from typing import Tuple

import numpy as np

from llm.config import N_VALIDATORS
from llm.llms import pass_llm
from llm.model.qa_simout import QASimulationOutput
from opensbt.evaluation.fitness import Fitness


CAR_CONTROL_JUDGE_PROMPT = """Evaluate an in-car assistant response to a car-control request.

Score request fulfillment (R), directness (D), and helpful follow-up (P) from 0 to 2.
Return only JSON: {{"scores": {{"R": 0, "D": 0, "P": 0}}}}.

User request: {question}
Assistant response: {answer}
"""


class CCFitnessAnswerValidationDimensions(Fitness):
    def __init__(self, llm_type=None, weights=(0.6, 0.3, 0.1)):
        super().__init__()
        self.llm_type = llm_type
        self.weights = weights

    @property
    def min_or_max(self):
        return ("min",)

    @property
    def name(self):
        return ("answer_fitness",)

    def eval(self, simout: QASimulationOutput, **kwargs) -> Tuple[float]:
        prompt = CAR_CONTROL_JUDGE_PROMPT.format(
            question=simout.utterance.question,
            answer=simout.utterance.answer,
        )
        scores = []
        for _ in range(N_VALIDATORS):
            try:
                response = pass_llm(prompt, llm_type=self.llm_type)
                response_scores = json.loads(response)["scores"]
                scores.append(
                    {
                        "R": float(response_scores["R"]),
                        "D": float(response_scores["D"]),
                        "P": float(response_scores["P"]),
                    }
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                scores.append({"R": 0, "D": 0, "P": 0})

        averaged_scores = np.mean(
            [[score["R"], score["D"], score["P"]] for score in scores],
            axis=0,
        )
        weighted_score = float(np.dot(averaged_scores, self.weights) / 2)
        simout.other["fitness_answer_scores"] = dict(
            zip(("R", "D", "P"), averaged_scores)
        )
        return (weighted_score,)