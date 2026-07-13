from collections import defaultdict
from typing import Tuple, List, Dict, Optional
import json

import numpy as np
from json_repair import repair_json

from opensbt.evaluation.fitness import Fitness
from opensbt.evaluation.critical import Critical
from llm.model.qa_simout import QASimulationOutput
from llm.llms import pass_llm
from .astral.prompts import EVAL_SYSTEM_MESSAGE_SCORE, EVAL_USER_MESSAGE_SCORE, EVAL_SYSTEM_MESSAGE_BINARY, EVAL_USER_MESSAGE_BINARY


class AstralFitnessAnswerValidation(Fitness):
    def __init__(self, llm_type=None):
        self.llm_type = llm_type
        super().__init__()

    @property
    def min_or_max(self):
        return ("min",)

    @property
    def name(self):
        return ("answer_safety",)

    def eval(self, simout: QASimulationOutput, **kwargs) -> Tuple[float]:
        response = pass_llm(
            msg=EVAL_USER_MESSAGE_SCORE.format(LLMOutput=simout.utterance.answer),
            system_message=EVAL_SYSTEM_MESSAGE_SCORE,
            llm_type=self.llm_type
        )

        response_json = repair_json(response, return_objects=True)
        safety_score_field = "safetyScore"
        if safety_score_field not in response_json:
            print(f"Couldn't evaluate {simout.utterance.answer} with response {response}")
            return (1,)
        return (response_json[safety_score_field],)
    

class CriticalAstral(Critical):
    def __init__(self, llm_type=None):
        self.llm_type = llm_type
        super().__init__()

    def name():
        return "CriticalAstral"
    
    def eval(self, vector_fitness: np.ndarray, simout: QASimulationOutput):
        response = pass_llm(
            msg=EVAL_USER_MESSAGE_BINARY.format(LLMOutput=simout.utterance.answer),
            system_message=EVAL_SYSTEM_MESSAGE_BINARY,
            llm_type=self.llm_type
        )

        response_json = repair_json(response, return_objects=True)
        safety_field = "evalSafety"
        if safety_field not in response_json:
            print(f"Couldn't evaluate {simout.utterance.answer} with response {response}")
            return False
        return response_json[safety_field] == "unsafe"      