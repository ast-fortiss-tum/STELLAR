from typing import List, Optional
from llm.model.models import Utterance
from llm.llms import pass_llm

from llm.model.qa_simout import QASimulationOutput


class IOSimulator:
    def __init__(self, llm_type: Optional[str] = None):
        self.llm_type = llm_type

    def simulate(
        self,
        list_individuals: List[List[Utterance]],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float = 10,
        do_visualize: bool = False,
        temperature: float = 0,
        context: object = None,
    ) -> List[QASimulationOutput]:
        result = []
        for ul in list_individuals:
            utterance = ul[0]
            response = pass_llm(
                utterance.question,
                temperature=temperature,
                system_message="",
                llm_type=self.llm_type,
            )
            utterance.answer = response
            result.append(
                QASimulationOutput(utterance, self.llm_type, response=response)
            )
        return result
