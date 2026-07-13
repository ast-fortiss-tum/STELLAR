from llm.llms import LLMType
from typing import List
from opensbt.simulation.simulator import Simulator
from llm.model.qa_simout import QASimulationOutput
from llm.model.models import Utterance
from llm.llms import pass_llm
from llm.config import LLM_IPA
import logging as log
from llm.prompts import SYSTEM_PROMPT
import traceback

class IPA(Simulator):
    memory: List = []
    ipa_name = "generic"

    @staticmethod
    def simulate(list_individuals: List[List[Utterance]], 
                 variable_names: List[str], 
                 scenario_path: str, 
                 sim_time: float, 
                 time_step: float = 10,
                 do_visualize: bool = False,
                 temperature: float = 0,
                 context: object = None,
                 llm_type = LLMType(LLM_IPA)) -> List[QASimulationOutput]:

        results = []
        
        # log.info("[IPA] context", context)
        log.info(f"[IPA] list_individuals: {list_individuals}")
        
        for utterance in list_individuals:  
            utterance = utterance[0]

            def check_utterance_in_mem(utterance):
                for u in IPA.memory:
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
                            llm_type=llm_type,
                            temperature=temperature,
                            context=context,
                            system_message=SYSTEM_PROMPT
                        )
                        if response is not None:
                            utterance.answer = response
                            utterance.raw_output = {
                                "system_output": response
                            }
                            break  # Exit loop after successful response
                    except Exception as e:
                        traceback.print_exc()
                        print(f"[IPA] Attempt {attempt + 1} failed with error: {e} using llm: {LLMType(LLM_IPA)}")
                    attempt += 1
                IPA.memory.append(utterance)
            else:
                print("Utterance already in memory")
                utterance.answer = memory_utterance.answer

            result = QASimulationOutput(
                utterance=utterance,
                model=LLMType(LLM_IPA),
                ipa=IPA.ipa_name,
            )
            results.append(result)

        return results
    