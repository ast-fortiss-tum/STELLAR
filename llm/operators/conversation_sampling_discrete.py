from typing import List

from llm.llms import LLMType
from llm.model.models import Conversation
from llm.model.qa_problem import QAProblem
from llm.config import LLM_SAMPLING
from .base_classes import ConversationSamplingBase


class ConversationSamplingDiscrete(ConversationSamplingBase):
    def __init__(
            self,
            variable_length=False,
            llm_type=LLMType(LLM_SAMPLING),
            generate_conversation: bool = False,
    ):
        super().__init__()
        self.variable_length = variable_length
        self.llm_type = llm_type
        self.generate_conversation = generate_conversation

        print("CSD generation_conversation:", generate_conversation)

    def _sample_instances(
            self, problem: QAProblem, n_samples: int, **kwargs
    ) -> List[Conversation]:
        result = []
        # if problem.seed_sampler is not None:
        #     seeds = problem.seed_sampler.sample_seeds(n_samples)
        #     print("seeds:", seeds)
        # else:
        #     seeds = [None for _ in range(n_samples)]

        for _ in range(n_samples):
            vars = problem.feature_handler.sample_feature_scores()
            result.append(self._build_conversation(
                problem=problem,
                ordinal_vars=vars.ordinal,
                categorical_vars=vars.categorical,
                continuous_vars=vars.continuous,
            ))
        return result