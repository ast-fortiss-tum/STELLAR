import json
import random
import traceback
from abc import abstractmethod
from typing import List, Tuple, Dict, Set, Optional, Any, Type

import numpy as np

from opensbt.simulation.simulator import Simulator
from llm.model.qa_simout import QASimulationOutput
from llm.model.mt_simout import MultiTurnSimulationOutput
from llm.model.models import Utterance, Conversation, Turn, ContentInput
from llm.model.conversation_intents import (
    classify_system_intent,
    UserIntent,
    SYSTEM_TO_USER,
    OPTIMIZABLE_USER_INTENTS,
    select_intent_by_priority,
    PRE_CONFIRMATION_INTENTS,
)
from llm.features import FeatureHandler
from llm.features.models import CombinedFeaturesInstance
from llm.llms import LLMType, pass_llm
import llm.config as _llm_config

# Intents that should NOT include content_requirements in their prompts
INTENTS_WITHOUT_CONTENT_REQUIREMENTS = {"ask", "choice", "confirmation", "reject", "stop"}

# Intents that should include ONLY the newly introduced feature
INTENTS_WITH_NEW_FEATURE_ONLY = {"add_preferences", "reject_clarify"}

# Intent that should include ONLY the changed features after resampling
INTENTS_WITH_CHANGED_FEATURES_ONLY = {"change_of_mind"}

# Intent that repeats the previous turn's content features
INTENTS_WITH_REPEAT = {"repeat"}


class IPABase(Simulator):
    """
    Abstract base class for all IPA simulators.

    Subclasses must set the following class attributes:
        - FOLLOW_UP_PROMPTS: dict mapping intent -> prompt template
        - ANCHOR_KEYS: list of keys always included in content (e.g. ["category"] or ["system", "system2"])
        - content_input_class: the Pydantic model class for content input

    And override the abstract methods for domain-specific behaviour.
    """
    ipa_name: str = "base"
    global_user_counter: int = 0

    # --- Subclass must set these ---
    FOLLOW_UP_PROMPTS: Dict[str, str] = {}
    ANCHOR_KEYS: List[str] = []
    content_input_class: Type[ContentInput] = ContentInput

    SYSTEM_FALLBACK_RESPONSE = "Unfortunately, something failed."

    # ------------------------------------------------------------------ #
    #  Abstract methods — must be overridden by every concrete subclass   #
    # ------------------------------------------------------------------ #

    @staticmethod
    @abstractmethod
    def simulate(
        list_individuals: List[List[Utterance]],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float = 10,
        do_visualize: bool = False,
        temperature: float = 0,
        context: object = None,
        max_retries: int = 3,
        **kwargs,
    ) -> List[QASimulationOutput]:
        pass

    @staticmethod
    @abstractmethod
    def simulate_turn(
        user_text: str,
        user_intent: str,
        user_id: str,
        current_content_input: Optional[ContentInput],
        history: List[str],
        max_retries: int = 3,
        llm_type: Optional[str] = None,
        **kwargs,
    ) -> Turn:
        pass

    @staticmethod
    @abstractmethod
    def simulate_conversation(
        list_individuals: List[List[Conversation]],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float = 10,
        do_visualize: bool = False,
        temperature: float = 0,
        context: object = None,
        config_path: str = "configs/features_navi_mt.json",
        max_retries: int = 3,
        min_turns: int = 2,
        max_turns: int = 5,
        max_repeats: int = 2,
        **kwargs,
    ) -> List[MultiTurnSimulationOutput]:
        pass

    # ------------------------------------------------------------------ #
    #  Abstract hooks — domain-specific, overridden by Navi / CC          #
    # ------------------------------------------------------------------ #

    @staticmethod
    @abstractmethod
    def resample_content(
        feature_handler: FeatureHandler,
        original_categorical_vars: List[int],
        original_ordinal_vars: List[float],
        previous_content_input: Optional[ContentInput] = None,
        max_attempts: int = 20,
    ) -> Tuple[List[int], List[float]]:
        pass

    @staticmethod
    @abstractmethod
    def initialize_content_features(
        initial_content_input: ContentInput,
        sample: bool = True,
    ) -> Tuple[Set[str], Set[str], ContentInput]:
        pass

    @staticmethod
    @abstractmethod
    def _build_fallback_user_utterance(
        current_content_input: Optional[ContentInput],
        new_feature_this_turn: Optional[str] = None,
        changed_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        pass

    @staticmethod
    @abstractmethod
    def _build_turn_content_input(
        current_content_input: ContentInput,
        used_content_features: Set[str],
        new_feature_this_turn: Optional[str],
        user_intent: str,
        content_input_turn1: Optional[ContentInput] = None,
    ) -> ContentInput:
        pass

    @staticmethod
    @abstractmethod
    def handle_change_of_mind(
        feature_handler: FeatureHandler,
        utterance_gen: Any,
        categorical_vars: List[int],
        ordinal_vars: List[float],
        previous_content_input: ContentInput,
        max_resample_attempts: int = 10,
    ) -> Tuple[ContentInput, Dict[str, Any], Set[str], Set[str], Dict[str, Any]]:
        pass

    @staticmethod
    @abstractmethod
    def handle_add_preferences(
        user_intent: str,
        current_content_input: ContentInput,
        all_content_features: Set[str],
        used_content_features: Set[str],
        feature_handler: FeatureHandler,
    ) -> Tuple[ContentInput, Set[str], Optional[str]]:
        """
        Handle add_preferences / reject_clarify intent.

        Returns:
            current_content_input: (possibly modified) content input
            used_content_features: updated set
            new_feature_this_turn: name of newly added feature, or None
        """
        pass

    # ------------------------------------------------------------------ #
    #  Shared concrete methods                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_intent_priorities(
        continuous_vars: List[float],
    ) -> Dict[str, float]:
        if not continuous_vars:
            return {}
        return dict(zip(OPTIMIZABLE_USER_INTENTS, continuous_vars))

    @staticmethod
    def get_num_turns(
        categorical_vars: List[int],
        feature_handler: FeatureHandler,
    ) -> int:
        try:
            num_turns_feature = feature_handler.categorical_features["num_turns"]
            num_turns_idx = categorical_vars[-1]
            return num_turns_feature.values[num_turns_idx]
        except Exception:
            return 1

    @staticmethod
    def determine_next_user_intent(
        turn_idx: int,
        min_turns: int,
        max_turns: int,
        processed_turns: List[Turn],
        conversation: Conversation,
        intent_priorities: Dict[str, float],
        unused_content_features: Set[str],
        llm_type: LLMType,
        max_retries: int = 3,
        allow_repeat_intent: bool = True,
        pre_confirmation_intents_used: Optional[Set[str]] = None,
        confirmed: bool = False,
        add_preferences_used: bool = False,
        **kwargs,
    ) -> Tuple[str, str]:
        if not processed_turns:
            return UserIntent.STOP.value, "misc"

        conversation.turns = processed_turns

        try:
            sys_intent = classify_system_intent(conversation, llm_type)
        except Exception as e:
            print(f"[IPABase] classify_system_intent Exception: {e}")
            sys_intent = "misc"

        processed_turns[-1].answer_intent_classified = sys_intent
        prev_user_intent = processed_turns[-1].question_intent

        if sys_intent not in SYSTEM_TO_USER:
            print(f"[IPABase] System intent '{sys_intent}' not in mapping, defaulting to 'misc'")
            sys_intent = "misc"

        possible_user_intents = SYSTEM_TO_USER.get(
            sys_intent, SYSTEM_TO_USER["misc"]
        )

        allow_stop = (turn_idx + 1) >= min_turns

        user_intent = select_intent_by_priority(
            possible_intents=possible_user_intents,
            intent_priorities=intent_priorities,
            prev_user_intent=prev_user_intent,
            allow_repeat=(sys_intent in ["reject", "reject_and_followup"]),
            unused_content_features=unused_content_features,
            allow_stop=allow_stop,
            allow_repeat_intent=allow_repeat_intent,
            pre_confirmation_intents_used=pre_confirmation_intents_used,
            confirmed=confirmed,
            add_preferences_used=add_preferences_used,
            system_intent_classified=sys_intent,
        )

        return user_intent, sys_intent

    @staticmethod
    def handle_repeat(
        previous_turn: Turn,
        used_content_features: Set[str],
    ) -> Tuple[Optional[ContentInput], Set[str]]:
        if previous_turn.content_input is not None:
            repeat_content_input = previous_turn.content_input.model_copy()
        else:
            repeat_content_input = None
        return repeat_content_input, used_content_features

    @classmethod
    def build_content_requirements_for_intent(
        cls,
        user_intent: str,
        current_content_input: ContentInput,
        used_content_features: Set[str],
        new_feature_this_turn: Optional[str] = None,
        changed_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        anchor_keys = cls.ANCHOR_KEYS

        if user_intent in INTENTS_WITHOUT_CONTENT_REQUIREMENTS:
            return "No specific content requirements for this turn."

        if user_intent in INTENTS_WITH_NEW_FEATURE_ONLY:
            if new_feature_this_turn is not None:
                val = getattr(current_content_input, new_feature_this_turn, None)
                if val is not None:
                    lines = []
                    for ak in anchor_keys:
                        ak_val = getattr(current_content_input, ak, None)
                        if ak_val is not None:
                            lines.append(f"- {ak}: {ak_val}")
                    lines.append(f"- {new_feature_this_turn}: {val}")
                    lines.append("- You MUST reflect this attribute in your turn.")
                    return "\n".join(lines)
            return "No specific content requirements for this turn."

        if user_intent in INTENTS_WITH_CHANGED_FEATURES_ONLY:
            if changed_features:
                lines = []
                for k, v in changed_features.items():
                    if v is not None:
                        lines.append(f"- {k}: {v}")
                    else:
                        lines.append(f"- {k}: removed/no longer required")
                if lines:
                    lines.append("- You MUST reflect the changed attributes in your turn.")
                    return "\n".join(lines)
            return "No specific content requirements for this turn."

        if user_intent in INTENTS_WITH_REPEAT:
            d = current_content_input.model_dump(exclude_none=True) if current_content_input else {}
            relevant = {k: v for k, v in d.items() if k in anchor_keys or k in used_content_features}
            if relevant:
                lines = [f"- {k}: {v}" for k, v in relevant.items()]
                lines.append("- You are REPEATING your previous request. Rephrase it but keep the same requirements.")
                return "\n".join(lines)
            return "Repeat your previous request with the same requirements."

        # Default: show all used features up to this point
        d = current_content_input.model_dump(exclude_none=True)
        relevant = {k: v for k, v in d.items() if k in anchor_keys or k in used_content_features}
        if relevant:
            lines = [f"- {k}: {v}" for k, v in relevant.items()]
            lines.append("- You MUST reflect every listed attribute in your turn.")
            return "\n".join(lines)
        return "No specific content requirements for this turn."

    @classmethod
    def generate_follow_up_utterance(
        cls,
        user_intent: str,
        feature_values: Dict[str, Any],
        current_content_input: ContentInput,
        history: List[str],
        utterance_gen: Any,
        llm_type: LLMType,
        used_content_features: Set[str],
        new_feature_this_turn: Optional[str] = None,
        changed_features: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> str:
        prompts = cls.FOLLOW_UP_PROMPTS
        prompt_template = prompts.get(
            user_intent,
            prompts.get("repeat", prompts.get("ask", "")),
        )

        style_prompt_str = utterance_gen._style_prompt(feature_values)

        content_req_str = cls.build_content_requirements_for_intent(
            user_intent=user_intent,
            current_content_input=current_content_input,
            used_content_features=used_content_features,
            new_feature_this_turn=new_feature_this_turn,
            changed_features=changed_features,
        )

        dialogue_history_str = "\n".join(history)

        full_prompt = prompt_template.format(
            history=dialogue_history_str,
            style_prompt=style_prompt_str,
            content_requirements=content_req_str,
        )

        user_text = None
        for attempt in range(max_retries):
            try:
                user_text = pass_llm(full_prompt, llm_type=llm_type, temperature=0.7)
                if user_text is not None and user_text.strip() != "":
                    break
                else:
                    user_text = None
                    print(f"[IPABase] generate_follow_up_utterance attempt {attempt + 1} returned empty string")
            except Exception:
                print(f"[IPABase] generate_follow_up_utterance attempt {attempt + 1} failed:")
                traceback.print_exc()

        if user_text is None or user_text.strip() == "":
            user_text = cls._build_fallback_user_utterance(
                current_content_input=current_content_input,
                new_feature_this_turn=new_feature_this_turn,
                changed_features=changed_features,
            )
            print(f"[IPABase] Using fallback user utterance: {user_text}")

        return utterance_gen._apply_post_perturbations(user_text, feature_values)

    @classmethod
    def generate_first_utterance(
        cls,
        conversation: Conversation,
        utterance_gen: Any,
        content_input_turn1: ContentInput,
        llm_type: LLMType,
        max_retries: int = 3,
    ) -> Utterance:
        last_exception = None
        for attempt in range(max_retries):
            try:
                utterance = utterance_gen.generate_utterance(
                    seed=conversation.seed,
                    ordinal_vars=conversation.ordinal_vars,
                    categorical_vars=conversation.categorical_vars,
                    llm_type=llm_type,
                    content_input_override=content_input_turn1,
                )
                if utterance.question is not None and utterance.question.strip() != "":
                    return utterance
                else:
                    print(f"[IPABase] generate_first_utterance attempt {attempt + 1} returned empty question")
            except Exception as e:
                print(f"[IPABase] generate_first_utterance attempt {attempt + 1} failed: {e}")
                traceback.print_exc()
                last_exception = e

        fallback_question = cls._build_fallback_user_utterance(
            current_content_input=content_input_turn1,
        )
        print(f"[IPABase] Using fallback first utterance: {fallback_question}")
        return Utterance(
            question=fallback_question,
            seed=conversation.seed,
            ordinal_vars=conversation.ordinal_vars,
            categorical_vars=conversation.categorical_vars,
            content_input=content_input_turn1,
        )

    @classmethod
    def prepare_conversation_state(
        cls,
        conversation: Conversation,
        feature_handler: FeatureHandler,
        utterance_gen: Any,
        min_turns: int = 2,
        max_turns: int = 5,
    ) -> Dict[str, Any]:
        intent_priorities = cls.parse_intent_priorities(
            conversation.continuous_vars
        )
        feature_values = feature_handler.get_feature_values_dict(
            ordinal_feature_scores=conversation.ordinal_vars,
            categorical_feature_indices=conversation.categorical_vars,
        )
        initial_content_input = cls.content_input_class.model_validate(feature_values)
        initial_content_input = utterance_gen.apply_constraints(initial_content_input)

        all_cf, used_cf, content_input_turn1 = cls.initialize_content_features(
            initial_content_input, sample=(max_turns > 1)
        )

        return {
            "min_turns": min_turns,
            "max_turns": max_turns,
            "intent_priorities": intent_priorities,
            "feature_values": feature_values,
            "initial_content_input": initial_content_input,
            "all_content_features": all_cf,
            "used_content_features": used_cf,
            "content_input_turn1": content_input_turn1,
        }

    @classmethod
    def run_conversation_loop(
        cls,
        conversation: Conversation,
        feature_handler: FeatureHandler,
        utterance_gen: Any,
        llm_ipa: LLMType,
        llm_classifier: LLMType,
        context: object = None,
        max_retries: int = 3,
        min_turns: int = 2,
        max_turns: int = 5,
        max_repeats: int = 2,
        llm_type_utterance_gen: Optional[LLMType] = None,
        **turn_kwargs,
    ) -> Conversation:
        change_of_mind_intents = False
        if llm_type_utterance_gen is None:
            llm_type_utterance_gen = llm_classifier

        state = cls.prepare_conversation_state(
            conversation, feature_handler, utterance_gen,
            min_turns=min_turns, max_turns=max_turns,
        )

        min_turns = state["min_turns"]
        max_turns = state["max_turns"]
        intent_priorities = state["intent_priorities"]
        feature_values = state["feature_values"]
        initial_content_input = state["initial_content_input"]
        all_content_features = state["all_content_features"]
        used_content_features = state["used_content_features"]
        content_input_turn1 = state["content_input_turn1"]

        utterance_obj = cls.generate_first_utterance(
            conversation, utterance_gen, content_input_turn1, llm_type_utterance_gen
        )

        user_text = utterance_obj.question
        user_intent = UserIntent.START.value
        current_content_input = initial_content_input

        processed_turns: List[Turn] = []
        history: List[str] = []

        new_feature_this_turn: Optional[str] = None
        changed_features: Optional[Dict[str, Any]] = None

        repeat_count: int = 0
        pre_confirmation_intents_used: Set[str] = set()
        confirmed: bool = False
        add_preferences_used: bool = False

        for turn_idx in range(max_turns):
            if turn_idx > 0:
                unused_content_features = all_content_features - used_content_features
                allow_repeat_intent = repeat_count < max_repeats

                user_intent, sys_intent = cls.determine_next_user_intent(
                    turn_idx=turn_idx,
                    min_turns=min_turns,
                    max_turns=max_turns,
                    processed_turns=processed_turns,
                    conversation=conversation,
                    intent_priorities=intent_priorities,
                    unused_content_features=unused_content_features,
                    llm_type=llm_classifier,
                    max_retries=max_retries,
                    allow_repeat_intent=allow_repeat_intent,
                    pre_confirmation_intents_used=pre_confirmation_intents_used,
                    confirmed=confirmed,
                    add_preferences_used=add_preferences_used,
                )

                if user_intent == UserIntent.CHANGE_OF_MIND.value and change_of_mind_intents:
                    user_intent = UserIntent.REPEAT.value

                new_feature_this_turn = None
                changed_features = None

                if user_intent == UserIntent.CONFIRMATION.value:
                    confirmed = True

                if user_intent in PRE_CONFIRMATION_INTENTS:
                    pre_confirmation_intents_used.add(user_intent)

                if user_intent == UserIntent.ADD_PREFERENCES.value:
                    add_preferences_used = True

                # handle repeat
                if user_intent == UserIntent.REPEAT.value:
                    repeat_count += 1
                    previous_turn = processed_turns[-1]
                    repeat_content_input, used_content_features = cls.handle_repeat(
                        previous_turn=previous_turn,
                        used_content_features=used_content_features,
                    )
                    if repeat_content_input is not None:
                        current_content_input = repeat_content_input

                # handle change of mind
                elif user_intent == UserIntent.CHANGE_OF_MIND.value:
                    change_of_mind_intents = True
                    (
                        current_content_input,
                        feature_values,
                        all_content_features,
                        used_content_features,
                        changed_features,
                    ) = cls.handle_change_of_mind(
                        feature_handler=feature_handler,
                        utterance_gen=utterance_gen,
                        categorical_vars=conversation.categorical_vars,
                        ordinal_vars=conversation.ordinal_vars,
                        previous_content_input=current_content_input,
                    )

                # handle add_preferences / reject_clarify
                elif user_intent in [
                    UserIntent.ADD_PREFERENCES.value,
                    UserIntent.REJECT_CLARIFY.value,
                ]:
                    current_content_input, used_content_features, new_feature_this_turn = cls.handle_add_preferences(
                        user_intent=user_intent,
                        current_content_input=current_content_input,
                        all_content_features=all_content_features,
                        used_content_features=used_content_features,
                        feature_handler=feature_handler,
                    )

                user_text = cls.generate_follow_up_utterance(
                    user_intent=user_intent,
                    feature_values=feature_values,
                    current_content_input=current_content_input,
                    history=history,
                    utterance_gen=utterance_gen,
                    llm_type=llm_type_utterance_gen,
                    used_content_features=used_content_features,
                    new_feature_this_turn=new_feature_this_turn,
                    changed_features=changed_features,
                    max_retries=max_retries,
                )

            turn_content_input = cls._build_turn_content_input(
                current_content_input=current_content_input,
                used_content_features=used_content_features,
                new_feature_this_turn=new_feature_this_turn,
                user_intent=user_intent,
                content_input_turn1=content_input_turn1 if turn_idx == 0 else None,
            )

            turn = None
            for attempt in range(max_retries):
                try:
                    turn = cls.simulate_turn(
                        user_text=user_text,
                        user_intent=user_intent,
                        user_id=str(conversation.assigned_user_id),
                        current_content_input=turn_content_input,
                        history=history,
                        max_retries=max_retries,
                        context=context,
                        conversation=conversation,
                        llm_type=llm_ipa,
                        **turn_kwargs,
                    )
                    if turn is not None and turn.answer is not None and turn.answer.strip() != "":
                        break
                    else:
                        print(f"[IPABase] simulate_turn attempt {attempt + 1} returned empty system response")
                        turn = None
                except Exception as e:
                    print(f"[IPABase] simulate_turn attempt {attempt + 1} failed: {e}")
                    traceback.print_exc()
                    turn = None

            if turn is None or turn.answer is None or turn.answer.strip() == "":
                if turn is None:
                    turn = Turn(
                        question=user_text,
                        answer=cls.SYSTEM_FALLBACK_RESPONSE,
                        question_intent=user_intent,
                        content_input=turn_content_input.model_copy() if turn_content_input else None,
                        content_output_list=[],
                        poi_exists=False,
                    )
                else:
                    turn.answer = cls.SYSTEM_FALLBACK_RESPONSE
                history.append(f"User: {user_text}")
                history.append(f"System: {turn.answer}")
                print(f"[IPABase] Using fallback system response for turn {turn_idx}")

            processed_turns.append(turn)

            if user_intent == UserIntent.STOP.value:
                break

        conversation.turns = processed_turns
        conversation.content_input_used = used_content_features

        return conversation
