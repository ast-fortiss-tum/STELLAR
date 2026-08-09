from typing import Any, Dict

from examples.navi.models import StyleDescription

from tarot.misc.models import NLUContentInput
from jupyter.nlu_bot.intents import INTENTS, INTENT_SET


PROMPT_GENERATOR = """ You are an intelligent user request generator to test an in car assistant."""

FALLBACK_INTENT = "INTENT_Unknown"


def style_prompt(features_dict: Dict[str, Any]) -> str:
	num_words = "num_words"

	result = ""

	if num_words in features_dict:
		result += (
			f"The utterance must contain exactly "
			f"{features_dict[num_words]} words\n"
		)

	style_description = StyleDescription.model_validate(features_dict)

	result += style_description.model_dump_json(
		exclude_none=True,
		indent=2,
	)

	return "The linguistic and style features are:\n" + result


def content_prompt(content_input: NLUContentInput) -> str:
	intent_name = (
		content_input.intent
		if content_input.intent in INTENT_SET
		else FALLBACK_INTENT
	)
	allowed_intents = "\n".join(f"- {intent}" for intent in INTENTS)

	return f"""
				INTENT:
				{intent_name}

				ALLOWED_INTENTS:
				{allowed_intents}

				RULE:
				Return one natural utterance expressing the intent.
				Use only the selected intent above as semantic target.

				The utterance should be shorter than 15 words.
				Prefer shorter utterances.

				Do not include variables or placeholders.

				The language must be: ENGLISH
				"""


def perturbation_prompt(feature_values: Dict[str, Any]) -> str:
	result = ""

	if "word_perturbation" in feature_values:
		if feature_values["word_perturbation"] == "introduce_fillers_llm_combined":
			result = """
					Apply also at the very end the following perturbation:

					Insert 1-2 natural filler words into the text
					to make it sound more conversational and natural.

					Return ONLY the modified text
					with fillers inserted.

					Use common English filler words like:
					"um", "uh", "well", "like", "you know",
					"actually", "I mean", "sort of", "kind of",
					or others if you think they are relevant.

					IMPORTANT:
					- Insert fillers at natural pause points
					(not in the middle of phrases)
					- Keep the original meaning and flow
					- Use fillers that fit the conversational tone
					- Don't overuse fillers
					- 1-2 insertions maximum
					- Maintain original punctuation and capitalization

					Examples:
					Input:
					"I think we should go to the park tomorrow."

					Output:
					"I think, um, we should go to the park tomorrow."

					Input:
					"The problem seems harder than expected."

					Output:
					"The problem seems, you know,
					harder than expected."

					Input:
					"The results show a clear improvement."

					Output:
					"The results actually
					show a clear improvement."

					Input:
					"She said she will arrive at five."

					Output:
					"She said she will arrive, well,
					at five."

					Input:
					"I do not know if this approach will work."

					Output:
					"I do not know, I mean,
					if this approach will work."
					"""

		elif feature_values["word_perturbation"] == "introduce_homophones_llm_combined":
			result = """
					At the very end, apply the following perturbation:

					Replace at least one and at most two words
					in the text with valid English homophones
					(words that sound the same
					but are spelled differently).

					Return ONLY the modified text
					with the substitutions applied.

					Requirements:
					- Use only real, valid homophones
					- Preserve original capitalization
					and punctuation
					- If no suitable homophones are available,
					return the text unchanged

					Examples:
					- "to" -> "too"
					- "there" -> "their"
					- "right" -> "write"
					- "break" -> "brake"
					- "know" -> "no"
					- "pair" -> "pear"
					"""

	return result


def utterance_prompt(
	style_text: str,
	content_text: str,
	perturbation_text: str,
) -> str:
	return f"""You are an intelligent, human-like utterance generator and know how people talk. Your task is to generate natural utterances considering the style attributes, content, and perturbation features defined below.
				
				Style:
				{style_text}

				Content:
				{content_text}

				Perturbations:
				{perturbation_text}

				Guidelines:
				- Styles:
					Slang (Slangy):
					Use German slang or colloquial expressions.

					Examples:
					- "Where can I grab some food?"
					- "Take me to the nearest spot."
					- "I need a place to chill."


					Implicit (Implicit):
					Ask indirectly without naming the venue explicitly.

					Examples:
					- "Where can I still get something warm?"
					- "My car is making strange noises."
					- "I need somewhere to sleep."


					Politeness (Rude):
					Sound unfriendly, impatient, or insulting.

					Examples:
					- "Hurry up already."
					- "Where is that damn place?"
					- "Start driving now."


					Anthropomorphism (very directive):
					Make the utterance very short and directive.

					Examples:
					- "To the station."
					- "Nearest gas station."
					- "Hospital now."
                    
				Return only one utterance.
				"""
