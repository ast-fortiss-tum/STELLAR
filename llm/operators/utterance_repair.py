from .base_classes import UtteranceRepairBase
from llm.llms import LLMType
from llm.config import LLM_SAMPLING
from llm.model.qa_problem import QAProblem
from llm.model.models import Utterance


class UtteranceRepairQuestionGenerator(UtteranceRepairBase):
    def __init__(
        self,
        llm_type=LLMType(LLM_SAMPLING),
    ):
        super().__init__()
        self.llm_type = llm_type
        self.generate_question = True

    def _repair_instance(self, problem: QAProblem, instance: Utterance, **kwargs):
        if instance.question is not None:
            return instance
        return self._build_utterance(
            problem,
            instance.seed,
            instance.ordinal_vars,
            instance.categorical_vars
        )


class NoUtteranceRepair(UtteranceRepairBase):
    def _repair_instance(self, problem: QAProblem, instance: Utterance, **kwargs):
        return instance
