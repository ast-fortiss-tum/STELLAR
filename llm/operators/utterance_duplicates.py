from llm.model.models import Utterance
from llm.utils import embeddings_local, embeddings_openai

from .base_classes import UtteranceDuplicateEliminationBase
from llm.eval.utterances_distance import UtterancesDistance


class UtteranceDuplicateElimination(UtteranceDuplicateEliminationBase):
    def _instances_equal(self, a: Utterance, b: Utterance) -> bool:
        return embeddings_openai.is_equal(
            a.question,
            b.question,
            threshold=0.9,
        )
    

class UtteranceDuplicateEliminationDistance(UtteranceDuplicateEliminationBase):
    def _instances_equal(self, a: Utterance, b: Utterance) -> bool:

        if (a.question is None) or (b.question is None):
            print("AASASASA", a, b)
            return False
        dist = UtterancesDistance.calculate(a, b, use_local_embeddings=True)
        return dist.embeddings_distance < 0.05



class UtteranceDuplicateEliminationLocal(UtteranceDuplicateEliminationBase):
    def _instances_equal(self, a: Utterance, b: Utterance) -> bool:
        return embeddings_local.is_equal(
            a.question,
            b.question,
            threshold=0.9,
        )


class UtteranceDuplicateEliminationMock(UtteranceDuplicateEliminationBase):
    def _instances_equal(self, a: Utterance, b: Utterance):
        return False
