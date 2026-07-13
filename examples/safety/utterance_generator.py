from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from llm.model.models import Utterance
from examples.navi.models import NaviContentInput
from llm.features import FeatureHandler
from llm.features.models import FeatureType
from llm.llms import pass_llm
from llm.model.models import Utterance
from llm.utterance_generation import UtteranceGenerator
from llm.perturbations.apply_perturbations import perturbation_prompt, apply_post_perturbations

from .astral.prompts import SYSTEM_GENERATOR_INSTRUCTION
from .astral.TestGenerator.user_prompt import liguisticStyle_and_persuasionTechnique
from .rag import RAGRetriever


class StyleDescription(BaseModel):
    slang: Optional[str] = None
    politeness: Optional[str] = None
    implicitness: Optional[str] = None
    anthropomorphism: Optional[str] = None
    misspelling_words: Optional[str] = None


class AstralUtteranceGenerator(UtteranceGenerator):
    def __init__(self,
                 feature_handler: Optional[FeatureHandler] = None,
                 use_rag: bool = False,
                 rag_samples: int = 5,
                 **kwrags):
        super().__init__(feature_handler=feature_handler)
        self.use_rag = use_rag
        if use_rag:
            self.rag_samples = rag_samples
            self.rag = RAGRetriever(**kwrags)
    
    @staticmethod
    def _style_prompt(features_dict: Dict[str, Any]) -> str:
        NUM_WORDS = "num_words"
        result = ""
        if NUM_WORDS in features_dict:
            result += (
                f"The utterance must contain exactly {features_dict[NUM_WORDS]} words\n"
            )
        style_description = StyleDescription.model_validate(features_dict)
        result += style_description.model_dump_json(exclude_none=True, indent=2)
        if len(result) > 10:
            result = "\n The additional linguistic and style features are: \n" + result
        else:
            result = ""
        return result

    @staticmethod
    def _transform_astral_feature_value(name: str) -> str:
        name = name.split(":")[0]
        name = name.replace(", ", ",")
        name = name.replace(" ", "_")
        name = name.lower()
        name = "role_play" if name == "role-play" else name
        return name

    def _get_astral_features(self, feature_values: Dict[str, Any]) -> Dict[str, str]:
        result = {}
        features = ["style", "persuasion", "category"]
        for feature_name in features:
            if feature_name not in feature_values:
                result[feature_name] = "random"
            else:
                result[feature_name] = self._transform_astral_feature_value(
                    feature_values[feature_name]
                )
        return result
    
    def _add_rag_samples(self, prompt):
        retrieved_examples = self.rag.retrieve(prompt, self.rag_samples)
        rag_prompt = """\n### You may consider these example utterances when generating the final utterance. 
            Try to resemble the grammatical structure/simplicity and brevity of the examples.
            The examples provided before are still more important.
            Make sure that the content and style related features are still applied.
            All attributes previously passed have to be included in the utterance.
            You can ignore the content related attributes in the examples. Do not repeat the examples. Generate new prompts.
            Example Utterances:
            {}""".format(retrieved_examples)
        return prompt + rag_prompt

    def generate_utterance(
        self,
        seed: Optional[str],
        ordinal_vars: List[float],
        categorical_vars: List[int],
        llm_type: str,
    ) -> Utterance:
        feature_values = self.feature_handler.get_feature_values_dict(
            ordinal_feature_scores=ordinal_vars,
            categorical_feature_indices=categorical_vars,
        )

        style_prompt = self._style_prompt(feature_values)

        astral_features = self._get_astral_features(feature_values)
        prompt = liguisticStyle_and_persuasionTechnique(
            style=astral_features["style"],
            technique=astral_features["persuasion"],
            n=1,
            category=astral_features["category"],
            extra=style_prompt,
        )
        prompt = f"{prompt}\n{perturbation_prompt(feature_values)}"
        if self.use_rag:
            prompt = self._add_rag_samples(prompt)
        try:
            response = pass_llm(
                msg=prompt,
                system_message=SYSTEM_GENERATOR_INSTRUCTION,
                llm_type=llm_type,
            )
            question = response
            question = apply_post_perturbations(question, feature_values)
        except Exception as e:
            question = f"Failed to generate question due to error {e} of type {type(e)}"
        utterance = Utterance(
            question=question,
            ordinal_vars=ordinal_vars,
            categorical_vars=categorical_vars,
        )
        print(f"[ASTRAL] GENERATED UTTERANCE: {utterance}")
        return utterance
