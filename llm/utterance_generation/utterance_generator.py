from abc import abstractmethod, ABC
from typing import List, Optional, Dict, Any, Tuple

from llm.model.models import Utterance, ContentInput
from llm.features import FeatureHandler

class UtteranceGenerator(ABC):
    def __init__(self,
                 feature_handler: Optional[FeatureHandler] = None):
        self.feature_handler = feature_handler

    def apply_constraints(
        self,
        content_input: ContentInput
    ) -> ContentInput:
        pass

    @abstractmethod
    def generate_utterance(
        self,
        seed: Optional[str],
        ordinal_vars: List[float],
        categorical_vars: List[int],
        llm_type: str,
    ) -> Utterance:
        pass
