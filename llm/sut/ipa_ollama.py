from llm.llms import LLMType
from opensbt.model_ga.individual import Individual
from typing import List
from opensbt.simulation.simulator import Simulator
from llm.model.qa_simout import QASimulationOutput
from math import sin, cos, pi, ceil
import numpy as np
from random import random
from opensbt.utils import geometric
import json
from datetime import datetime
import csv
import os
from llm.model.models import Utterance
from llm.llms import pass_llm
from llm.config import LLM_IPA, LLM_OLLAMA
import logging as log
from llm.prompts import SYSTEM_PROMPT

class IPAOLLAMA(Simulator):
    ipa_name = "llama3.2"

    @staticmethod
    def simulate(list_individuals: List[Utterance], 
                 variable_names: List[str], 
                 scenario_path: str, 
                 sim_time: float, 
                 time_step: float = 10,
                 do_visualize: bool = False,
                 temperature: float = 0,
                 context: object = None) -> List[QASimulationOutput]:

        results = []
        
        # log.info("[IPA] context", context)
        # log.info("[IPA] list_individuals", list_individuals)
        
        for utterance in list_individuals:  
            utterance = utterance[0]
            utterance.answer = pass_llm(utterance.question,
                                        llm_type=LLMType(LLM_OLLAMA),
                                        temperature=temperature,
                                        context=context,
                                        system_message = SYSTEM_PROMPT)
            result = QASimulationOutput(
                utterance=utterance,
                model=LLMType(LLM_IPA),
                ipa=IPAOLLAMA.ipa_name,
            )
            results.append(result)

        return results
