from typing import List, Optional, Dict, Any

from tarot.misc.models import NLUContentInput
from llm.model.models import Utterance
from llm.features import FeatureHandler
from llm.llms import pass_llm, LLMType
from llm.utils.seed import set_seed
from llm.utterance_generation.utterance_generator import (
    UtteranceGenerator,
)
from tarot.misc.prompts import PROMPT_GENERATOR
from tarot.misc.prompts import (
    content_prompt,
    perturbation_prompt,
    style_prompt,
    utterance_prompt,
)


class NLUUtteranceGenerator(UtteranceGenerator):

    call_counter = 0

    def __init__(
        self,
        feature_handler: Optional[FeatureHandler] = None
    ):
        super().__init__(feature_handler=feature_handler)

    def build_style_prompt(
        self,
        feature_values: Dict[str, Any],
    ) -> str:
        return style_prompt(feature_values)

    def build_content_prompt(
        self,
        content_input: NLUContentInput,
    ) -> str:
        return content_prompt(content_input)

    def build_perturbation_prompt(
        self,
        feature_values: Dict[str, Any],
    ) -> str:
        return perturbation_prompt(feature_values)

    def build_utterance_prompt(
        self,
        style_prompt_text: str,
        content_prompt_text: str,
        perturbation_prompt_text: str,
    ) -> str:
        return utterance_prompt(
            style_prompt_text,
            content_prompt_text,
            perturbation_prompt_text,
        )

    def build_system_prompt(self) -> str:
        return PROMPT_GENERATOR

    def _apply_post_perturbations(
        self,
        text: str,
        feature_values: Dict[str, Any],
    ) -> str:

        from llm.perturbations.word_perturbations import (
            WORD_PERTURBATIONS,
        )
        from llm.perturbations.char_perturbations import (
            CHAR_PERTURBATIONS,
        )

        mapping = {
            "word_perturbation": WORD_PERTURBATIONS,
            "char_perturbation": CHAR_PERTURBATIONS,
        }

        for key, funcs in mapping.items():

            p = feature_values.get(key)

            if p in funcs:

                old = text

                text = funcs[p](text)

                if text == "":
                    text = old

        return text

    def generate_utterance(
        self,
        seed: Optional[str],
        ordinal_vars: List[float],
        categorical_vars: List[int],
        llm_type: str,
    ) -> Utterance:

        feature_values = (
            self.feature_handler.get_feature_values_dict(
                ordinal_feature_scores=ordinal_vars,
                categorical_feature_indices=categorical_vars,
            )
        )

        content_input = NLUContentInput.model_validate(
            feature_values
        )

        style_prompt_text = self.build_style_prompt(
            feature_values
        )

        content_prompt_text = self.build_content_prompt(
            content_input
        )

        perturbation_prompt_text = self.build_perturbation_prompt(
            feature_values
        )

        prompt = self.build_utterance_prompt(
            style_prompt_text,
            content_prompt_text,
            perturbation_prompt_text,
        )

        response = "Can not generate utterance."

        for _ in range(5):

            try:

                result = pass_llm(
                    msg=prompt,
                    system_message=self.build_system_prompt(),
                    llm_type=llm_type,
                    temperature=0.6,
                )

                if result is not None:

                    response = str(result).strip()

                    if response:
                        break

            except Exception:
                pass

        try:

            text = self._apply_post_perturbations(
                response,
                feature_values,
            )

        except Exception:

            text = response

        self.call_counter += 1

        return Utterance(
            question=text,
            seed=seed,
            ordinal_vars=ordinal_vars,
            categorical_vars=categorical_vars,
            content_input=content_input,
        )


if __name__ == "__main__":

    fhandler = FeatureHandler.from_json(
        "tarot/configs/features_nlu.json"
    )

    set_seed(42)

    gen = NLUUtteranceGenerator(fhandler)

    for _ in range(20):

        ord_vars, cat_vars, _ = (
            fhandler.sample_feature_scores()
        )

        u = gen.generate_utterance(
            seed=None,
            ordinal_vars=ord_vars[1],
            categorical_vars=cat_vars[1],
            llm_type=LLMType.GPT_4O_MINI,
        )

        print("Utterance:", u.question)