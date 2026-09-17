from llm.model.models import Conversation
from llm.utils import embeddings_local
from .base_classes import ConversationDuplicateEliminationBase


ORDINAL_VARS_THRESHOLD = 0.05
CONTINUOUS_VARS_THRESHOLD = 0.05
UTTERANCE_SIMILARITY_THRESHOLD = 0.9


class ConversationDuplicateEliminationVars(ConversationDuplicateEliminationBase):
    """
        Conversations are considered duplicates if:
        - ordinal_vars differ by less than ORDINAL_VARS_THRESHOLD for all corresponding variables
        - categorical_vars are exactly the same
        - continuous_vars differ by less than CONTINUOUS_VARS_THRESHOLD for all corresponding variables
    """
    def _instances_equal(self, a: Conversation, b: Conversation) -> bool:
        if a.categorical_vars != b.categorical_vars:
            return False

        if len(a.ordinal_vars) != len(b.ordinal_vars):
            return False

        if any(
            abs(ov_a - ov_b) >= ORDINAL_VARS_THRESHOLD
            for ov_a, ov_b in zip(a.ordinal_vars, b.ordinal_vars)
        ):
            return False

        if len(a.continuous_vars) != len(b.continuous_vars):
            return False

        if any(
            abs(cv_a - cv_b) >= CONTINUOUS_VARS_THRESHOLD
            for cv_a, cv_b in zip(a.continuous_vars, b.continuous_vars)
        ):
            return False

        return True

# NOTE: it requires non-empty turns field in Conversation, otherwise it will consider all conversations as duplicates
class ConversationDuplicateEliminationLocal(ConversationDuplicateEliminationBase):
    """
    Determines if two conversations are duplicates by comparing their utterances sequentially.
    
    Two conversations are considered equal if:
    1. They have the same number of utterances.
    2. Each corresponding utterance pair (question and answer) is semantically similar
       above a defined threshold.
    """
    def _instances_equal(self, a: Conversation, b: Conversation) -> bool:
        # if the number of turns is different, they can't be duplicates
        if len(a) != len(b):
            return False

        # empty conversations are equal
        if len(a) == 0:
            return True

        for utt_a, utt_b in zip(a.turns, b.turns):
            # compare the questions
            if utt_a.question and utt_b.question:
                if not embeddings_local.is_equal(utt_a.question, utt_b.question, threshold=UTTERANCE_SIMILARITY_THRESHOLD):
                    return False
            elif utt_a.question != utt_b.question:  # one has a question and the other doesn't
                return False

            # compare the answers
            if utt_a.answer and utt_b.answer:
                if not embeddings_local.is_equal(utt_a.answer, utt_b.answer, threshold=UTTERANCE_SIMILARITY_THRESHOLD):
                    return False
            elif utt_a.answer != utt_b.answer:  # one has an answer and the other doesn't
                return False
        
        return True