from .base_classes import ConversationRepairBase
from llm.llms import LLMType
from llm.config import LLM_SAMPLING
from llm.model.qa_problem import QAProblem
from llm.model.models import Conversation


class ConversationRepairConversationGenerator(ConversationRepairBase):
    def __init__(
        self,
        llm_type=LLMType(LLM_SAMPLING),
    ):
        super().__init__()
        self.llm_type = llm_type
        self.generate_conversation = True

    def _repair_instance(self, problem: QAProblem, instance: Conversation, **kwargs):
        if instance.turns:
            return instance
        return self._build_conversation(
            problem,
            instance.ordinal_vars,
            instance.categorical_vars,
            instance.continuous_vars,
        )


class NoConversationRepair(ConversationRepairBase):
    def _repair_instance(self, problem: QAProblem, instance: Conversation, **kwargs):
        return instance