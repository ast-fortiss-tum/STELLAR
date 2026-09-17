import random
from typing import List, Optional, Dict, Any, Tuple, Union

from pydantic import BaseModel
from llm.model.models import Utterance
from examples.car_control.models_new import CCContentInput, StyleDescription
from llm.features import FeatureHandler
from llm.llms import pass_llm
from llm.utils.seed import set_seed
from llm.utterance_generation.utterance_generator import UtteranceGenerator
#from examples.navi.prompts import NAVI_QUESTION_PROMPT, PROMPT_GENERATOR
from examples.car_control.prompts import CC_QUESTION_PROMPT, PROMPT_GENERATOR
from llm.features.models import FeatureType

from llm.llms import LLMType
import json

class CCUtteranceGenerator(UtteranceGenerator):
    call_counter = 0
    def __init__(self,
                 feature_handler: Optional[FeatureHandler] = None,
                 apply_constrains_to_vars: bool = True,
                 use_rag: bool = False):
        super().__init__(feature_handler=feature_handler)
        self.apply_constrains_to_vars = apply_constrains_to_vars
        self.use_rag = use_rag

        self.max_changes=3
        #if self.use_rag:
        #    from examples.navi import rag as rag
        #    self.rag_retriever = rag.RAGRetriever()

    def apply_constraints_style(self, content_input: CCContentInput, style_input: StyleDescription) -> StyleDescription:
        #if content_input.category == "hospital" and style_input.implicitness in ["implicit", "slightly implicit"]:
        #    style_input.implicitness = "not implicit"

        #if content_input.category == "car_repair" and style_input.implicitness in ["implicit"]:
        #    style_input.implicitness = "not implicit"

        return style_input
    
    # NOTE: make sure category names are not in plural form
    
    
    
    def apply_constraints(self, content_input: CCContentInput) -> CCContentInput:
        """
        Apply constraints to CCContentInput.
        """

        ci = content_input.model_copy(deep=True)
        setattr(ci, "system2", None)

        while ci.system == ci.system2:
            setattr(ci, "system2", random.choice(["windows", "fog_lights", "ambient_lights", "head_lights", "fan", "reading_lights", "climate", "seat_heating", None]))

        allowed_fields_by_system = {
            "windows": {
                "position",
                "window_state_target",
                "window_state_initial",
                "word_perturbation",
                "politeness",
                "anthropomorphism",
                "slang",
                "implicitness",
                "choice",
                "add_preferences",
                "ask",
                "change_of_mind",
                "confirmation",
                "reject",
                "reject_clarify",
                "repeat",
                "stop"
            },
            "fog_lights": {
                "fog_light_position",
                "onoff_state_target",
                "onoff_state_initial",
                "word_perturbation",
                "politeness",
                "anthropomorphism",
                "slang",
                "implicitness",
                "choice",
                "add_preferences",
                "ask",
                "change_of_mind",
                "confirmation",
                "reject",
                "reject_clarify",
                "repeat",
                "stop"
            },
            "ambient_lights": {
                "onoff_state_target",
                "onoff_state_initial",
                "word_perturbation",
                "politeness",
                "anthropomorphism",
                "slang",
                "implicitness",
                "choice",
                "add_preferences",
                "ask",
                "change_of_mind",
                "confirmation",
                "reject",
                "reject_clarify",
                "repeat",
                "stop"
            },
            "head_lights": {
                "onoff_state_target",
                "onoff_state_initial",
                "head_lights_mode_target",
                "head_lights_mode_initial",
                "word_perturbation",
                "politeness",
                "anthropomorphism",
                "slang",
                "implicitness",
                "choice",
                "add_preferences",
                "ask",
                "change_of_mind",
                "confirmation",
                "reject",
                "reject_clarify",
                "repeat",
                "stop"
            },
            "fan": {
                "onoff_state_target",
                "onoff_state_initial",
                "word_perturbation",
                "politeness",
                "anthropomorphism",
                "slang",
                "implicitness",
                "choice",
                "add_preferences",
                "ask",
                "change_of_mind",
                "confirmation",
                "reject",
                "reject_clarify",
                "repeat",
                "stop"
            },
            "reading_lights": {
                "position",
                "onoff_state_target",
                "onoff_state_initial",
                "word_perturbation",
                "politeness",
                "anthropomorphism",
                "slang",
                "implicitness",
                "choice",
                "add_preferences",
                "ask",
                "change_of_mind",
                "confirmation",
                "reject",
                "reject_clarify",
                "repeat",
                "stop"
            },
            "climate": {
                "onoff_state_target",
                "onoff_state_initial",
                "climate_temperature_value_target",
                "climate_temperature_value_initial",
                "word_perturbation",
                "politeness",
                "anthropomorphism",
                "slang",
                "implicitness",
                "choice",
                "add_preferences",
                "ask",
                "change_of_mind",
                "confirmation",
                "reject",
                "reject_clarify",
                "repeat",
                "stop"
            },
            "seat_heating": {
                "onoff_state_target",
                "onoff_state_initial",
                "seat_heating_level_target",
                "seat_heating_level_initial",
                "seat_position",
                "word_perturbation",
                "politeness",
                "anthropomorphism",
                "slang",
                "implicitness",
                "choice",
                "add_preferences",
                "ask",
                "change_of_mind",
                "confirmation",
                "reject",
                "reject_clarify",
                "repeat",
                "stop"
            },
        }

        all_fields = {
            "position",
            "seat_position",
            "fog_light_position",
            "window_state_target",
            "window_state_initial",
            "onoff_state_target",
            "onoff_state_initial",
            "head_lights_mode_target",
            "head_lights_mode_initial",
            "climate_temperature_value_target",
            "climate_temperature_value_initial",
            "seat_heating_level_target",
            "seat_heating_level_initial",
        }

        system = ci.system

        if system in allowed_fields_by_system:
            allowed_fields = allowed_fields_by_system[system]
        else:
            allowed_fields = set()

        fields_to_clean = all_fields - allowed_fields
        for field in fields_to_clean:
            setattr(ci, field, None)

        
        if ci.onoff_state_target == "off" and ci.system != "windows":

            target_fields = [
                "window_state_target",
                "onoff_state_target",
                "head_lights_mode_target",
                "climate_temperature_value_target",
                "seat_heating_level_target",
            ]

            for field in target_fields:
                if field != "onoff_state_target":  # dejamos este porque define la acción
                    setattr(ci, field, None)
        
        allowed_fields_by_system = {
            "windows": {
                "position2",
                "window_state_target2",
                "window_state_initial2",
            },
            "fog_lights": {
                "fog_light_position2",
                "onoff_state_target2",
                "onoff_state_initial2",
            },
            "ambient_lights": {
                "onoff_state_target2",
                "onoff_state_initial2",
            },
            "head_lights": {
                "onoff_state_target2",
                "onoff_state_initial2",
                "head_lights_mode_target2",
                "head_lights_mode_initial2",
            },
            "fan": {
                "onoff_state_target2",
                "onoff_state_initial2",
            },
            "reading_lights": {
                "position2",
                "onoff_state_target2",
                "onoff_state_initial2",
            },
            "climate": {
                "onoff_state_target2",
                "onoff_state_initial2",
                "climate_temperature_value_target2",
                "climate_temperature_value_initial2",
            },
            "seat_heating": {
                "onoff_state_target2",
                "onoff_state_initial2",
                "seat_heating_level_target2",
                "seat_heating_level_initial2",
                "seat_position2",
            },
        }

        all_fields = {
            "position2",
            "seat_position2",
            "fog_light_position2",
            "window_state_target2",
            "window_state_initial2",
            "onoff_state_target2",
            "onoff_state_initial2",
            "head_lights_mode_target2",
            "head_lights_mode_initial2",
            "climate_temperature_value_target2",
            "climate_temperature_value_initial2",
            "seat_heating_level_target2",
            "seat_heating_level_initial2",
        }

        system = ci.system2

        if system in allowed_fields_by_system:
            allowed_fields = allowed_fields_by_system[system]
        else:
            allowed_fields = set()

        fields_to_clean = all_fields - allowed_fields
        for field in fields_to_clean:
            setattr(ci, field, None)

        
        if ci.onoff_state_target2 == "off" and ci.system2 != "windows":

            target_fields = [
                "window_state_target2",
                "onoff_state_target2",
                "head_lights_mode_target2",
                "climate_temperature_value_target2",
                "seat_heating_level_target2",
            ]

            for field in target_fields:
                if field != "onoff_state_target2":
                    setattr(ci, field, None)


        return ci



        

    def _style_prompt(self,
                      features_dict: Dict[str, Any]) -> str:
        NUM_WORDS = "num_words"
        result = ""
        if NUM_WORDS in features_dict:
            result += f"The utterance must contain exactly {features_dict[NUM_WORDS]} words\n"
        style_description = StyleDescription.model_validate(features_dict)
        result += style_description.model_dump_json(exclude_none=True, indent=2)
        return "The linguistic and style features are: \n" + result
    
    def _content_prompt(
        self,
        content_input: CCContentInput
    ) -> str:
        content_attributes = list(content_input.model_dump(exclude_none=True).keys())
        content_attributes = [s for s in content_attributes if "initial" not in s]
        dict_content = content_input.model_dump_json(indent=2, exclude_none=True)
        
        data = json.loads(dict_content)
        cleaned_data = {
            key: value
            for key, value in data.items()
            if "initial" not in key.lower()
        }
        dict_content = json.dumps(cleaned_data, indent=2)
        
        content_prompt = (
            f"The content related features are: {dict_content} \n"
            "Use all the attributes in the final output. "
            f"Each attribute from {content_attributes} must appear once, either explicitly (e.g., “credit card”) or implicitly (e.g., “card only”)."
        )
        return content_prompt

    def _perturbation_prompt(self,
        features_dict: Dict[str, Any]) -> str:
        result = ""
        if "word_perturbation" in features_dict:
            if features_dict["word_perturbation"] == "introduce_fillers_llm":
                result = """Apply also at the very end the following perturbation:

                        Insert 1-2 natural filler words into the text to make it sound more conversational and natural. 
                        Return ONLY the modified text with fillers inserted.

                        Use common filler words like: "uh", "um", "like", "you know", "I mean", "well", "so", "actually", "basically", or others if you think they are relevant.

                        IMPORTANT:
                        - Insert fillers at natural pause points (not in the middle of phrases)
                        - Keep the original meaning and flow
                        - Use fillers that fit the conversational tone
                        - Don't overuse fillers - 1-2 insertions maximum
                        - Maintain original punctuation and capitalization

                                        Examples:
                        Input: "I think we should go to the park tomorrow."
                        Output: "I think, um, we should go to the park tomorrow."

                        Input: "This problem seems harder than I expected."
                        Output: "This problem, you know, seems harder than I expected."

                        Input: "Well, the results show a clear improvement."
                        Output: "Well, the results actually show a clear improvement."

                        Input: "She said she would arrive by 5 PM."
                        Output: "She said she would, like, arrive by 5 PM."
                        """
            elif features_dict["word_perturbation"] == "introduce_homophones_static":
                result = """At the very end, apply the following perturbation:

                            Replace at least one and at most two words in the text with valid homophones (words that sound the same but are spelled differently).  
                            Return ONLY the modified text with the substitutions applied.

                            Requirements:
                            - Use only real, valid homophones (not invented words).
                            - Preserve the original capitalization and punctuation.
                            - If no suitable homophones are available, return the text unchanged.

                            Examples:
                            - "write" → "wright"
                            - "two" → "to"
                            - "hear" → "here"
                            - "flower" → "flour"
                            - "knight" → "night"
                            - "sea" → "see"
                            - "whole" → "hole"
                            - "pair" → "pear"
                            - "meet" → "meat"
                            - "male" → "mail"
                            - "peace" → "piece"
                            """
        return result


    #def _rag_prompt(
    #    self,
    #    feature_values: dict,
    #    top_k_retrieved: int
    #) -> str:
    #    query = " ".join(f"{k}: {v}," for k, v in ((k, v) for k, v in feature_values.items() if v is not None))
    #    
    #    retrieved_examples = self.rag_retriever.retrieve(query, top_k_retrieved)

    #    rag_prompt = """"""
    #    if len(retrieved_examples) > 0:
    #        rag_prompt = """### Consider these example utterances when generating the final utterance. 
    #        Try to resemble the grammatical structure/simplicity and brevity of the examples.
    #        Make sure that the content and style related features are still applied.
    #        All content-related attributes previously passed have to be included in the utterance.
    #        You can ignore the content related attributes in the examples.

    #        Example Utterances:
    #        {}""".format(retrieved_examples)
        
    #    return rag_prompt
    
    def _seed_prompt(self,
                     seed: Optional[str], content_prompt: str) -> str:
        if seed is None:
            return ""
        
        seed_prompt = (f"Consider the original request '{seed}' for the generation.\n"
                       "Use synonyms for verbs, names and adverbs in the original request"
                       "that are suitable to fulfill the style requirements.\n"
                       "Consider the grammatical structure of the seed utterance."
                       "Consider the verbosity."
                       "Consider still the style-related features given in the previous style feature input."
                        "Make sure every attribute from the content attributed in some way is used in the output")
        if len(content_prompt) > 0:
            seed_prompt += "Change words to fulfill content requirements.\n"
        
        return seed_prompt

    def _apply_post_perturbations(self, question, feature_values: Dict[str, Any]) -> str:
        """Apply word or character perturbations to utterance."""
        from llm.perturbations.word_perturbations import WORD_PERTURBATIONS
        from llm.perturbations.char_perturbations import CHAR_PERTURBATIONS

        perturbation_mapping = {
            "word_perturbation": WORD_PERTURBATIONS,
            "char_perturbation": CHAR_PERTURBATIONS,
        }
        
        for key, perturbations in perturbation_mapping.items():
            perturbation = feature_values.get(key)
            if perturbation is not None and perturbation in perturbations:
                question_before = question
                question = perturbations[perturbation](question)
                if question == "":
                    # undo peturbation
                    question = question_before
        return question
    
    def _get_content_input(self, feature_values: Dict[str, Any]) -> CCContentInput:
        return CCContentInput.model_validate(feature_values)
    
    def _get_style_input(self, feature_values: Dict[str, Any]) -> StyleDescription:
        return StyleDescription.model_validate(feature_values)
    
    def _update_features_from_content_input(
            self,
            ordinal_vars: List[float],
            categorical_vars: List[int],
            content_input: object,
    ) -> Tuple[List[float], List[int]]:
        for i, feature in enumerate(self.feature_handler.ordinal_features.values()):
            if not hasattr(content_input, feature.name):
                continue
            new_value = getattr(content_input, feature.name, None)
            new_var = self.feature_handler.get_var_from_feature_value(
                feature,
                new_value,
                feature_type=FeatureType.ORDINAL
            )
            if new_var is not None:
                ordinal_vars[i] = new_var
        for i, feature in enumerate(self.feature_handler.categorical_features.values()):
            if not hasattr(content_input, feature.name):
                continue
            new_value = getattr(content_input, feature.name, None)
            new_var = self.feature_handler.get_var_from_feature_value(
                feature,
                new_value,
                feature_type=FeatureType.CATEGORICAL
            )
            if new_var is not None:
                categorical_vars[i] = new_var
        return ordinal_vars, categorical_vars

    def generate_utterance(
        self,
        seed: Optional[str],
        ordinal_vars: List[float],
        categorical_vars: List[int],
        llm_type: str,
        top_k_retrieved=5,
        content_input_override: Optional[CCContentInput] = None,
    ) -> Utterance:
        feature_values = self.feature_handler.get_feature_values_dict(
            ordinal_feature_scores=ordinal_vars,
            categorical_feature_indices=categorical_vars,
        )

        # allow caller to control which content features are active
        if content_input_override is not None:
            content_input = content_input_override
        else:
            content_input = self._get_content_input(feature_values)

        content_input = self.apply_constraints(content_input)

        style_input = self._get_style_input(feature_values)
        style_input = self.apply_constraints_style(content_input, style_input)

        if self.apply_constrains_to_vars:
            ordinal_vars, categorical_vars = self._update_features_from_content_input(
                ordinal_vars,
                categorical_vars,
                content_input,
            )
            ordinal_vars, categorical_vars = self._update_features_from_content_input(
                ordinal_vars,
                categorical_vars,
                style_input,
            )

        style_prompt = self._style_prompt(feature_values)
        content_prompt = self._content_prompt(content_input)
        seed_prompt = self._seed_prompt(seed, content_prompt)

        # post-only perturbations: do NOT instruct the LLM via prompt
        perturbation_prompt = self._perturbation_prompt(feature_values)

        #if self.use_rag:
        #    rag_prompt = self._rag_prompt(feature_values, top_k_retrieved)
        #else:
        #    rag_prompt = ""
        rag_prompt = ""

        prompt = CC_QUESTION_PROMPT.format(
            style_prompt=style_prompt,
            content_prompt=content_prompt,
            seed_prompt=seed_prompt,
            perturbation_prompt=perturbation_prompt,
            rag_prompt=rag_prompt
        )
        # print("""navi utternace rag prompt: {}""".format(prompt))
        # input()
        try:
            response = pass_llm(
                msg=prompt.replace("_", " "), 
                system_message=PROMPT_GENERATOR, 
                llm_type=llm_type,
                temperature=0.2
            )
            question = response 
            question = self._apply_post_perturbations(question, feature_values)
    
            self.call_counter = self.call_counter + 1
            print(f"{self.call_counter} generate_utterance calls")

        except Exception as e:
            print(f"[CCUtteranceGenerator] Failed to generate question: {e}")
            if content_input.system == "windows":
                question = f'{str(content_input.window_state).replace("_", " ")} the windows.'
            else:
                question = f'Set {str(content_input.system).replace("_", " ")} to {str(content_input.onoff_state_target).replace("_", " ")}'
            #question = None  # Let caller handle fallback

        return Utterance(
            question = question,
            seed = seed,
            ordinal_vars = ordinal_vars,
            categorical_vars = categorical_vars,
            content_input = content_input
        )

if __name__ == "__main__":
    fhandler = FeatureHandler.from_json("configs/features_simple_judge_cc.json")
    set_seed(100)
    gen = CCUtteranceGenerator(fhandler, use_rag = True)
    for i in range(5):
        sample_ord, sample_cat, continuous_cat = fhandler.sample_feature_scores()
        utter = gen.generate_utterance(seed=None,
                                    ordinal_vars=sample_ord[1],
                                    categorical_vars=sample_cat[1],
                                    llm_type=LLMType.GPT_4O_MINI
                        )
        print(fhandler.map_categorical_indices_to_labels(sample_cat[1]))
        print(fhandler.map_numerical_scores_to_labels(sample_ord[1]))
        print("Question:")
        print(utter.question)
        print("\n")



