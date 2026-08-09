from __future__ import annotations
import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# If notebook is in LUNAR-DEV/custom/, this resolves to LUNAR-DEV/
repo_root = Path.cwd().resolve().parent

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
# ---- runtime setup (jupyter + env) ----
load_dotenv(find_dotenv(), override=False)

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm.features import FeatureHandler
from llm.llms import LLMType, pass_llm
from llm.model.models import Utterance
from llm.perturbations.apply_perturbations import apply_post_perturbations
from llm.utterance_generation.utterance_generator import UtteranceGenerator

from custom.custom_models import CustomContentInput, CustomStyleDescription

class CustomUtteranceGenerator(UtteranceGenerator):
    call_counter = 0
    
    # holds information of all available intents, with descriptions and examples
    INTENT_REFERENCE_PATH = repo_root / "custom" / "configs" / "intent_reference.json"

    def __init__(self, feature_handler: Optional[FeatureHandler] = None):
        super().__init__(feature_handler=feature_handler)
        self.intent_reference = self._load_intent_reference()

    @classmethod
    def _load_intent_reference(cls) -> Dict[str, Any]:
        if cls.INTENT_REFERENCE_PATH.exists():
            try:
                return json.loads(cls.INTENT_REFERENCE_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        # fallback when no descriptions file exists
        return {
            "intents": {
                "INTENT_Climate": {
                    "description": "User asks to change cabin climate settings.",
                    "samples": [
                        "Turn on the AC.",
                        "Set temperature to 20 degrees.",
                        "It is too warm in here.",
                    ],
                },
                "INTENT_Navigation": {
                    "description": "User asks for route guidance or destination help.",
                    "samples": [
                        "Navigate to the station.",
                        "Show directions to the airport.",
                        "I need help getting somewhere.",
                    ],
                },
            },
            "slots": {},
        }

    @staticmethod
    def _has_llm_credentials() -> bool:
        env_names = (
            "OPENAI_API_KEY"
        )
        return any(os.getenv(name) for name in env_names)

    @staticmethod
    def _coerce_llm_type(llm_type: str | LLMType) -> LLMType:
        return llm_type if isinstance(llm_type, LLMType) else LLMType(llm_type)

    def _style_prompt(self, feature_values: Dict[str, Any]) -> str:
        style = CustomStyleDescription.model_validate(feature_values)
        result = style.model_dump_json(
            exclude={"word_perturbation", "char_perturbation"},
            indent=2,
        )
        if "num_words" in feature_values:
            result = f'{{"num_words": {feature_values["num_words"]}}}\n' + result
        return "Style features:\n" + result

    def _content_prompt(self, content_input: CustomContentInput) -> str:
        intents = self.intent_reference.get("intents", {})
        info = intents.get(content_input.intent, {"description": "Unknown intent", "samples": []})

        samples = info.get("samples", [])[:3]
        slots = self.intent_reference.get("slots", {})
        
        # provide few shot examples for the intent
        sample_lines = "\n".join(f"- {s}" for s in samples) if samples else "- No examples"

        return f"""INTENT:
            {content_input.intent}
            
            SLOTS:
            {slots}
            
            EXAMPLES:
            {sample_lines}
            
            RULE:
            Return exactly one natural user utterance for this intent.
            Do not use placeholders.
            Keep it concise (prefer <= 15 words unless style requires longer).
            """

    @staticmethod
    def _perturbation_prompt(feature_values: Dict[str, Any]) -> str:
        word_p = feature_values.get("word_perturbation", "none")
        if word_p == "introduce_fillers_llm_combined":
            return (
                "At the end, insert 1-2 natural filler words. "
                "Return only the modified text."
            )
        if word_p == "introduce_homophones_llm_combined":
            return (
                "At the end, replace 1-2 words with valid homophones if possible. "
                "Return only the modified text."
            )
        return "No extra LLM-side perturbation instruction."

    @staticmethod
    def _apply_final_style_rules(question: str, style: CustomStyleDescription) -> str:
        question = question.strip(" \t\n.,!?")
        if style.slang == "slangy" and not question.lower().startswith("hey"):
            question = f"Hey, {question}"
        if style.politeness == "polite" and not question.lower().endswith("please"):
            question = f"{question}, please"
        return question[:1].upper() + question[1:] if question else question

    @staticmethod
    def _fallback_question(content_input: CustomContentInput, style: CustomStyleDescription) -> str:
        """ Customize based on your use case. """

        fallback = {
            "INTENT_Climate": {
                "explicit": ["turn on the AC", "set the temperature to 20 degrees", "start the fan", "turn on the heating"],
                "implicit": ["it is too warm in here", "the cabin feels cold"],
            },
            "INTENT_Navigation": {
                "explicit": ["navigate to the station", "show directions to the airport", "find a route home", "open the map"],
                "implicit": ["I need help getting somewhere", "help me reach my destination"],
            },
        }
        text = random.choice(fallback[content_input.intent][style.implicitness])
        if style.verbosity == "medium":
            text = f"{text} right now"
        elif style.verbosity == "long":
            text = f"{text} when you have a moment"
        return text

    def build_prompt(self, content_input: CustomContentInput, feature_values: Dict[str, Any]) -> str:
        """ Customize based on your use case. """
        
        style_block = self._style_prompt(feature_values)
        content_block = self._content_prompt(content_input)
        perturb_block = self._perturbation_prompt(feature_values)

        return f"""You generate human-like in-car assistant utterances.

                {style_block}
                
                {content_block}
                
                Perturbations:
                {perturb_block}
                
                Guidelines:
                - Preserve intent semantics.
                - Apply style naturally.
                - Return only one utterance.
                """

    def generate_utterance(
        self,
        seed: Optional[str],
        ordinal_vars: List[float],
        categorical_vars: List[int],
        llm_type: str | LLMType,
    ) -> Utterance:
        feature_values = self.feature_handler.get_feature_values_dict(
            ordinal_feature_scores=ordinal_vars,
            categorical_feature_indices=categorical_vars,
        )

        content_input = CustomContentInput(intent=feature_values["intent"])
        style = CustomStyleDescription.model_validate(feature_values)
        selected_llm = self._coerce_llm_type(llm_type)

        text = ""
        if selected_llm != LLMType.MOCK and self._has_llm_credentials():
            prompt = self.build_prompt(content_input, feature_values)
            for _ in range(5):
                try:
                    result = pass_llm(
                        msg=prompt,
                        system_message="You are a concise utterance generator.",
                        llm_type=selected_llm,
                        temperature=0.6,
                    )
                    if result:
                        text = str(result).strip()
                        if text:
                            break
                except Exception:
                    continue

        if not text:
            text = self._fallback_question(content_input, style)
            text = self._apply_final_style_rules(text, style)

        text = apply_post_perturbations(text, feature_values)
        self.call_counter += 1

        return Utterance(
            question=text,
            seed=seed,
            ordinal_vars=ordinal_vars,
            categorical_vars=categorical_vars,
            content_input=content_input,
        )


if __name__ == "__main__":
    features_path = repo_root / "custom" / "configs" / "features.json"
    handler = FeatureHandler.from_json(str(features_path))
    gen = CustomUtteranceGenerator(handler)
    sample = handler.sample_feature_scores()

    u_mock = gen.generate_utterance(None, sample.ordinal, sample.categorical, LLMType.MOCK)
    print("MOCK:")
    print(u_mock.model_dump_json(indent=2))

    u_llm = gen.generate_utterance(None, sample.ordinal, sample.categorical, LLMType.GPT_4O_MINI)
    print("LLM:")
    print(u_llm.model_dump_json(indent=2))