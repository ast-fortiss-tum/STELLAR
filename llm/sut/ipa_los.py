import re
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
from examples.navi.models import NaviContentOutput
from llm.model.models import Utterance
from llm.llms import pass_llm
from llm.config import LLM_IPA
import logging as log
from llm.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_CONTENT_INPUT
import copy
import traceback
from json_repair import repair_json
from dataclasses import fields

class IPA_LOS(Simulator):
    memory: List = []
    ipa_name = "generic_with_los"

    @staticmethod
    def simulate(list_individuals: List[List[Utterance]], 
                 variable_names: List[str], 
                 scenario_path: str, 
                 sim_time: float, 
                 time_step: float = 10,
                 do_visualize: bool = False,
                 temperature: float = 0,
                 context: object = None) -> List[QASimulationOutput]:

        results = []
        
        # log.info("[IPA] context", context)
        log.info(f"[IPA] list_individuals: {list_individuals}")
        
        for utterance in list_individuals:  
            utterance = utterance[0]

            def check_utterance_in_mem(utterance):
                for u in IPA_LOS.memory:
                    if u.question == utterance.question:
                        return True, u
                return False, utterance
            
            in_memory, memory_utterance = check_utterance_in_mem(utterance)

            if not in_memory:
                max_attempts = 5
                attempt = 0

                while attempt < max_attempts:
                    try:
                        response = pass_llm(
                            msg=utterance.question,
                            llm_type=LLMType(LLM_IPA),
                            temperature=temperature,
                            context=context,
                            system_message=SYSTEM_PROMPT_CONTENT_INPUT
                        )

                        if response is not None:
                            response = repair_json(response)
                            print("IPA LOS Response:", response)
                            response_parsed = json.loads(response)
                            print("response parsed:", response_parsed)

                            utterance.answer = response_parsed["system_response"]
                            
                            if (len(response_parsed["los"])) > 0:
                                utterance.content_output_list = [
                                    NaviContentOutput.model_validate(entry) for entry in response_parsed["los"]
                                ]
                            else:
                                utterance.content_output_list = []

                            break 
                    except Exception as e:
                        traceback.print_exc()
                        print(f"[IPA_LOS] Attempt {attempt + 1} failed with error: {e}")
                    attempt += 1
                IPA_LOS.memory.append(utterance)
            else:
                print("Utterance already in memory")
                utterance.answer = memory_utterance.answer
            print(f"[IPA LOS] {utterance}")
            result = QASimulationOutput(
                utterance=utterance,
                model=LLMType(LLM_IPA),
                ipa=IPA_LOS.ipa_name,
            )
            results.append(result)

        return results


if __name__ == "__main__":
    qus = ["Time for burgers! Need directions, 5 stars!",
           "I need to open my windows on the left.", 
           "It is very cold.",
           "I am very hungry and need a Pizza.",
           "I am very hungry and need a French baguette.",
           "Show options for breakfasts.",
           "Where is the closest public toilet.",
           "Where is the closest Cafe.",
           "I am running out of gas.",
           "I need to charge my car.",
           "I am looking for a church."]
    utterances = []
    for u in qus:
        utterances.append([Utterance(question=u)])

    context = {
                        "location" : {
                            "position" : "Amathountos Avenue 502, 4520, Limassol, Cyprus",
                            "data" : "2025-03-19T0",
                            "time" : "09:00:00",
                        }
            }
    res = IPA_LOS.simulate(utterances,
                    ["utterance"], 
                    "", 
                    sim_time=-1,
                    time_step=-1, 
                    do_visualize=False,
                    temperature = 0,
                    context = context)