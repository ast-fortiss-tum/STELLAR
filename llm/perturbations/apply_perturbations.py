from typing import Dict, Any

from .word_perturbations import WORD_PERTURBATIONS
from .char_perturbations import CHAR_PERTURBATIONS


def perturbation_prompt(features_dict: Dict[str, Any]) -> str:
    result = ""
    if "word_perturbation" in features_dict:
        if features_dict["word_perturbation"] == "introduce_fillers_llm_combined":
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

                    Input: "I don’t know if this approach will work."
                    Output: "I don’t know, I mean, if this approach will work."
                    """
        elif features_dict["word_perturbation"] == "introduce_homophones_llm_combined":
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


def apply_post_perturbations(question, feature_values: Dict[str, Any]) -> str:
    """Apply word or character perturbations to utterance."""

    perturbation_mapping = {
        "word_perturbation": WORD_PERTURBATIONS,
        "char_perturbation": CHAR_PERTURBATIONS,
    }
    
    for key, perturbations in perturbation_mapping.items():
        perturbation = feature_values.get(key)
        if perturbation is not None and perturbation in perturbations:
            question = perturbations[perturbation](question)
    return question
