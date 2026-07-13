from llm.model.models import Utterance
from llm.utils import embeddings_local, embeddings_openai
from llm.utils.math import euclid_distance, mae, mse

from .base_classes import UtteranceDuplicateEliminationBase


class UtteranceDuplicateEliminationDiscrete(UtteranceDuplicateEliminationBase):
    def _instances_equal(self, a: Utterance, b: Utterance) -> bool:
        return euclid_distance(
            a.ordinal_vars, b.ordinal_vars
        ) < 0.1 and embeddings_openai.is_equal(a.question, b.question, threshold=0.9)

class UtteranceDuplicateEliminationLocalDiscrete(UtteranceDuplicateEliminationBase):
    def _instances_equal(self, a: Utterance, b: Utterance) -> bool:
        return euclid_distance(
            a.ordinal_vars, b.ordinal_vars
        ) < 0.1 and embeddings_local.is_equal(a.question, b.question, threshold=0.9)
    
# with los

class UtteranceDuplicateEliminationDiscreteWithContent(UtteranceDuplicateEliminationBase):
    def _instances_equal(self, a: Utterance, b: Utterance) -> bool:
         return (
            mse(a.ordinal_vars, b.ordinal_vars) <= 0.1 and
            a.categorical_vars == b.categorical_vars and
            embeddings_openai.is_equal(a.question, b.question, threshold=0.9)
        )

class UtteranceDuplicateEliminationLocalDiscreteWithContent(UtteranceDuplicateEliminationBase):
    def _instances_equal(self, a: Utterance, b: Utterance) -> bool:
          return (
            mse(a.ordinal_vars, b.ordinal_vars) <= 0.1 and
            a.categorical_vars == b.categorical_vars and
            embeddings_local.is_equal(a.question, b.question, threshold=0.9)
        )
    

class UtteranceDuplicateEliminationDiscreteVars(UtteranceDuplicateEliminationBase):
    def _instances_equal(self, a: Utterance, b: Utterance) -> bool:
        return (
        mse(a.ordinal_vars, b.ordinal_vars) <= 0.1 and
        a.categorical_vars == b.categorical_vars
    )
