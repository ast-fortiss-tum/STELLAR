import random
from enum import Enum
from typing import TYPE_CHECKING, Optional, List, Dict

from llm.llms import pass_llm, LLMType
from llm.prompts import CLASSIFY_SYSTEM_INTENT_PROMPT


class UserIntent(Enum):
    START = "start"
    CHOICE = "choice"
    ADD_PREFERENCES = "add_preferences"
    ASK = "ask"
    CHANGE_OF_MIND = "change_of_mind"
    CONFIRMATION = "confirmation"
    REJECT = "reject"
    REJECT_CLARIFY = "reject_clarify"
    REPEAT = "repeat"
    STOP = "stop"


class OptimizableUserIntent(Enum):
    CHOICE = "choice"
    ADD_PREFERENCES = "add_preferences"
    ASK = "ask"
    CHANGE_OF_MIND = "change_of_mind"
    CONFIRMATION = "confirmation"
    REJECT = "reject"
    REJECT_CLARIFY = "reject_clarify"
    REPEAT = "repeat"
    STOP = "stop"


class SystemIntent(Enum):
    INFORM = "inform"
    INFORM_AND_FOLLOWUP = "inform_and_followup"
    # INFORM_REPEAT = "inform_repeat"
    # INFORM_REPEAT_AND_FOLLOWUP = "inform_repeat_and_followup"
    CHOICE = "choice"
    CHOICE_AND_FOLLOWUP = "choice_and_followup"
    CONFIRMATION = "confirmation"
    CONFIRMATION_AND_FOLLOWUP = "confirmation_and_followup"
    CLARIFY = "clarify"
    REJECT = "reject"
    REJECT_AND_FOLLOWUP = "reject_and_followup"
    FAILURE = "failure"
    MISC = "misc"


USER_INTENTS = [intent.value for intent in UserIntent]
SYSTEM_INTENTS = [intent.value for intent in SystemIntent]

# Intents that are part of the search space (have continuous priorities)
OPTIMIZABLE_USER_INTENTS = [intent.value for intent in OptimizableUserIntent]

# user-intent ordering and ID mapping
# IDs are 0-based indices into USER_INTENTS
USER_INTENT_TO_ID = {intent: idx for idx, intent in enumerate(USER_INTENTS)}
ID_TO_USER_INTENT = {idx: intent for intent, idx in USER_INTENT_TO_ID.items()}

# mapping from system's classified intents to possible user intents
SYSTEM_TO_USER = {
    SystemIntent.INFORM.value: [
        UserIntent.ADD_PREFERENCES.value,
        UserIntent.ASK.value,
        UserIntent.CHANGE_OF_MIND.value,
        UserIntent.REJECT.value,
        UserIntent.REJECT_CLARIFY.value,
        UserIntent.STOP.value,
    ],

    SystemIntent.INFORM_AND_FOLLOWUP.value: [
        UserIntent.CHANGE_OF_MIND.value,
        UserIntent.ADD_PREFERENCES.value,
        UserIntent.CONFIRMATION.value,
        UserIntent.REJECT.value,
        UserIntent.REJECT_CLARIFY.value,
        UserIntent.ASK.value,
        UserIntent.STOP.value,
    ],

    # SystemIntent.INFORM_REPEAT.value: [
    #     UserIntent.ADD_PREFERENCES.value,
    #     UserIntent.CHANGE_OF_MIND.value,
    #     UserIntent.REJECT.value,
    #     UserIntent.REJECT_CLARIFY.value,
    #     UserIntent.ASK.value,
    # ],

    # SystemIntent.INFORM_REPEAT_AND_FOLLOWUP.value: [
    #     UserIntent.ADD_PREFERENCES.value,
    #     UserIntent.CHANGE_OF_MIND.value,
    #     UserIntent.REJECT.value,
    #     UserIntent.REJECT_CLARIFY.value,
    #     UserIntent.CONFIRMATION.value,
    # ],

    SystemIntent.CHOICE.value: [
        UserIntent.CHOICE.value,
        UserIntent.ASK.value,
        UserIntent.ADD_PREFERENCES.value,
        UserIntent.CHANGE_OF_MIND.value,
        UserIntent.REJECT.value,
        UserIntent.REJECT_CLARIFY.value,
        UserIntent.STOP.value,
    ],

    SystemIntent.CHOICE_AND_FOLLOWUP.value: [
        UserIntent.CHOICE.value,
        UserIntent.ADD_PREFERENCES.value,
        UserIntent.CHANGE_OF_MIND.value,
        UserIntent.REJECT.value,
        UserIntent.REJECT_CLARIFY.value,
        UserIntent.STOP.value,
    ],

    SystemIntent.CONFIRMATION.value: [
        UserIntent.CHANGE_OF_MIND.value,
        UserIntent.ASK.value,
        UserIntent.ADD_PREFERENCES.value,
        UserIntent.STOP.value,
    ],

    SystemIntent.CONFIRMATION_AND_FOLLOWUP.value: [
        UserIntent.CHANGE_OF_MIND.value,
        UserIntent.ADD_PREFERENCES.value,
        UserIntent.CONFIRMATION.value,
        UserIntent.STOP.value,
    ],

    SystemIntent.CLARIFY.value: [
        UserIntent.ADD_PREFERENCES.value,
        UserIntent.ASK.value,
        UserIntent.STOP.value,
    ],

    SystemIntent.REJECT.value: [
        UserIntent.CHANGE_OF_MIND.value,
        UserIntent.ASK.value,
        UserIntent.STOP.value,
    ],

    SystemIntent.REJECT_AND_FOLLOWUP.value: [
        UserIntent.CHANGE_OF_MIND.value,
        UserIntent.ASK.value,
        UserIntent.CONFIRMATION.value,
        UserIntent.STOP.value,
    ],

    SystemIntent.FAILURE.value: [
        UserIntent.CHANGE_OF_MIND.value,
        UserIntent.REPEAT.value,
        UserIntent.STOP.value,
    ],

    SystemIntent.MISC.value: [
        UserIntent.CHANGE_OF_MIND.value,
        UserIntent.ASK.value,
        UserIntent.REJECT.value,
        UserIntent.REJECT_CLARIFY.value,
        UserIntent.STOP.value,
    ],
}


def user_intent_to_id(intent: str) -> int:
    if intent not in USER_INTENT_TO_ID:
        raise ValueError(f"Unknown user intent: '{intent}'. Known intents: {USER_INTENTS}")
    
    return USER_INTENT_TO_ID[intent]


def id_to_user_intent(idx: int) -> str:
    if idx not in ID_TO_USER_INTENT:
        raise ValueError(f"Unknown user intent id: {idx}. Valid ids: 0..{len(USER_INTENTS)-1}")
    
    return ID_TO_USER_INTENT[idx]


def classify_system_intent(conversation, llm_type) -> str:
    """
    Classify the system's last response intent using an LLM.
    """
    # get the last turns's answer
    if not conversation.turns or not conversation.turns[-1].answer:
        raise ValueError("Cannot classify system intent: no system answer in the last utterance")
    
    last_answer = conversation.turns[-1].answer
    possible_intents = list(SYSTEM_TO_USER.keys())

    prompt = CLASSIFY_SYSTEM_INTENT_PROMPT.format(
        system_answer=last_answer,
        possible_intents=', '.join(possible_intents)
    )

    result = pass_llm(
        prompt,
        temperature=0.0,
        max_tokens=10*1000,
        llm_type=llm_type,
    ).strip().lower()

    if result in possible_intents:
        return result
    
    raise ValueError(f"LLM returned an invalid system intent: '{result}'. Valid intents are: {possible_intents}")


# The set of intents that must be used (2 distinct) before confirmation is allowed,
# and that are blocked after confirmation is used.
PRE_CONFIRMATION_INTENTS = {
    UserIntent.CHANGE_OF_MIND.value,
    UserIntent.REJECT.value,
    UserIntent.REJECT_CLARIFY.value,
    UserIntent.ADD_PREFERENCES.value,
}


def select_intent_by_priority(
    possible_intents: List[str],
    intent_priorities: Dict[str, float],
    prev_user_intent: Optional[str] = None,
    allow_repeat: bool = False,
    unused_content_features: Optional[set] = None,
    allow_stop: bool = True,
    allow_repeat_intent: bool = True,
    pre_confirmation_intents_used: Optional[set] = None,
    confirmed: bool = False,
    **kwargs,
) -> str:
    """
    Select an intent from possible_intents using priority-weighted random selection.
    
    Args:
        possible_intents: List of valid intents to choose from
        intent_priorities: Dict mapping intent names to their priority values [0.001, 1.0]
        prev_user_intent: Previous user intent (to potentially exclude)
        allow_repeat: Whether the same intent can be selected again
        unused_content_features: Set of content features not yet used in conversation.
                                 If empty, add_preferences and reject_clarify are excluded.
        allow_stop: Whether the stop intent is allowed (False when min_turns not yet reached)
        allow_repeat_intent: Whether the 'repeat' intent is allowed (False when max_repeats reached)
        pre_confirmation_intents_used: Set of distinct intents from PRE_CONFIRMATION_INTENTS
                                        that have been used so far. Confirmation requires >= 2.
        confirmed: Whether confirmation has already been used. If True,
                   all PRE_CONFIRMATION_INTENTS are blocked.
    
    Returns:
        Selected intent string
    """
    if pre_confirmation_intents_used is None:
        pre_confirmation_intents_used = set()

    # Filter out 'start' (only valid for first turn)
    candidates = [i for i in possible_intents if i != UserIntent.START.value]
    
    # Filter out 'stop' if not yet allowed (min_turns not reached)
    if not allow_stop:
        candidates = [i for i in candidates if i != UserIntent.STOP.value]
    
    # Filter out 'repeat' if max_repeats reached
    if not allow_repeat_intent:
        candidates = [i for i in candidates if i != UserIntent.REPEAT.value]
    
    # Rule 2: After confirmation, block all pre-confirmation intents
    if confirmed:
        candidates = [i for i in candidates if i not in PRE_CONFIRMATION_INTENTS]
    
    # Rule 1: Confirmation requires at least 2 distinct pre-confirmation intents used
    if len(pre_confirmation_intents_used) < 2:
        candidates = [i for i in candidates if i != UserIntent.CONFIRMATION.value]
    
    # Also block confirmation if already confirmed (can't confirm twice)
    if confirmed:
        candidates = [i for i in candidates if i != UserIntent.CONFIRMATION.value]
    
    if not allow_repeat and prev_user_intent and prev_user_intent in candidates:
        candidates = [i for i in candidates if i != prev_user_intent]
    
    # Filter out add_preferences and reject_clarify if no unused content features remain
    if unused_content_features is not None and len(unused_content_features) == 0:
        feature_requiring_intents = {
            UserIntent.ADD_PREFERENCES.value,
            UserIntent.REJECT_CLARIFY.value
        }
        candidates = [i for i in candidates if i not in feature_requiring_intents]
    
    # If no candidates left, fall back
    if not candidates:
        candidates = [i for i in possible_intents if i != UserIntent.START.value]
        if not allow_stop:
            candidates = [i for i in candidates if i != UserIntent.STOP.value]
        if not allow_repeat_intent:
            candidates = [i for i in candidates if i != UserIntent.REPEAT.value]
        if confirmed:
            candidates = [i for i in candidates if i not in PRE_CONFIRMATION_INTENTS]
            candidates = [i for i in candidates if i != UserIntent.CONFIRMATION.value]
        if len(pre_confirmation_intents_used) < 2:
            candidates = [i for i in candidates if i != UserIntent.CONFIRMATION.value]
        if unused_content_features is not None and len(unused_content_features) == 0:
            feature_requiring_intents = {
                UserIntent.ADD_PREFERENCES.value,
                UserIntent.REJECT_CLARIFY.value
            }
            candidates = [i for i in candidates if i not in feature_requiring_intents]
    
    if not candidates:
        return UserIntent.STOP.value  # terminal fallback
    
    # Build weights from priorities
    weights = []
    for intent in candidates:
        if intent in intent_priorities:
            weights.append(intent_priorities[intent])
        else:
            weights.append(0.5)
    
    # Normalize weights to sum to 1.0
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    else:
        weights = [1.0 / len(candidates)] * len(candidates)
    
    # Weighted random selection
    return random.choices(candidates, weights=weights, k=1)[0]


def get_followup_user_intent_with_priorities(
    system_intent_classified: str,
    intent_priorities: Dict[str, float],
    prev_user_intent: Optional[str] = None,
    unused_content_features: Optional[set] = None,
    allow_repeat_intent: bool = True,
    pre_confirmation_intents_used: Optional[set] = None,
    confirmed: bool = False,
) -> str:
    """
    Get the next user intent based on the classified system intent and intent priorities.
    """
    if not system_intent_classified or system_intent_classified not in SYSTEM_TO_USER:
        raise ValueError(
            f"Invalid or unclassified system intent: '{system_intent_classified}'. "
            "Cannot determine follow-up user intent."
        )

    possible_intents = list(SYSTEM_TO_USER[system_intent_classified])
    
    allow_repeat = system_intent_classified in {
        SystemIntent.REJECT.value, 
        SystemIntent.REJECT_AND_FOLLOWUP.value
    }
    
    return select_intent_by_priority(
        possible_intents=possible_intents,
        intent_priorities=intent_priorities,
        prev_user_intent=prev_user_intent,
        allow_repeat=allow_repeat,
        unused_content_features=unused_content_features,
        allow_repeat_intent=allow_repeat_intent,
        pre_confirmation_intents_used=pre_confirmation_intents_used,
        confirmed=confirmed,
    )


# Keep original function for backward compatibility
def get_followup_user_intent(system_intent_classified: str, prev_user_intent: Optional[str] = None) -> str:
    """
    Get the next user intent based on the classified system intent.
    Legacy function - uses uniform random selection.
    """
    if not system_intent_classified or system_intent_classified not in SYSTEM_TO_USER:
        raise ValueError(
            f"Invalid or unclassified system intent: '{system_intent_classified}'. "
            "Cannot determine follow-up user intent."
        )

    possible_intents = list(SYSTEM_TO_USER[system_intent_classified])

    allow_repeat = system_intent_classified in {
        SystemIntent.REJECT.value, 
        SystemIntent.REJECT_AND_FOLLOWUP.value
    }
    if not allow_repeat and prev_user_intent:
        possible_intents = [i for i in possible_intents if i != prev_user_intent]

    # fallback: if we filtered everything out, allow repeat rather than crash
    if not possible_intents and system_intent_classified in SYSTEM_TO_USER:
        possible_intents = list(SYSTEM_TO_USER[system_intent_classified])

    return random.choice(possible_intents)