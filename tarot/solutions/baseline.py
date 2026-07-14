from pathlib import Path
import sys
from typing import Any, Dict

project_root = next((p for p in (Path.cwd(), *Path.cwd().parents[:2]) if (p / "examples").exists() and (p / "tarot").exists()), None)

if project_root is None:
    raise FileNotFoundError("Could not locate the STELLAR project root")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.navi.models import StyleDescription
from tarot.misc.models import NLUContentInput
from tarot.misc.nlu_utterance_generator import NLUUtteranceGenerator
from tarot.nlu_bot.intents import INTENTS, INTENT_SET

PROMPT_GENERATOR = """ You are an intelligent user request generator to test an in car assistant."""
FALLBACK_INTENT = "INTENT_Unknown"

class CustomNLUUtteranceGenerator(NLUUtteranceGenerator):
    def build_style_prompt(
        self,
        feature_values: Dict[str, Any],
    ) -> str:
        num_words = "num_words"
        result = ""

        print("Style")
        print(feature_values)
        if num_words in feature_values:
            result += (
                f"The utterance must contain exactly "
                f"{feature_values[num_words]} words\n"
            )

        style_description = StyleDescription.model_validate(feature_values)

        result += style_description.model_dump_json(
            exclude_none=True,
            indent=2,
        )

        return "The linguistic and style features are:\n" + result

    def build_content_prompt(
        self,
        content_input: NLUContentInput,
    ) -> str:
        intent_name = (
            content_input.intent
            if content_input.intent in INTENT_SET
            else FALLBACK_INTENT
        )
        allowed_intents = "\n".join(
            f"- {intent}" for intent in INTENTS
        )

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

    def build_perturbation_prompt(
        self,
        feature_values: Dict[str, Any],
    ) -> str:
        result = ""
        print("Perturbation")
        print(feature_values)
        if "word_perturbation" in feature_values:
            if feature_values["word_perturbation"] == "introduce_fillers_llm_combined":
                result = """
Apply also at the very end the following perturbation:

Insert 1-2 natural filler words into the text
to make it sound more conversational and natural.

Return ONLY the modified text
with fillers inserted.

Use common English filler words like:
\"um\", \"uh\", \"well\", \"like\", \"you know\",
\"actually\", \"I mean\", \"sort of\", \"kind of\",
or others if you think they are relevant.

IMPORTANT:
- Insert fillers at natural pause points
(not in the middle of phrases)
- Keep the original meaning and flow
- Use fillers that fit the conversational tone
- Don't overuse fillers
- 1-2 insertions maximum
- Maintain original punctuation and capitalization
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
"""

        return result

    def build_utterance_prompt(
        self,
        style_prompt_text: str,
        content_prompt_text: str,
        perturbation_prompt_text: str,
    ) -> str:
        return f"""You are an intelligent, human-like utterance generator and know how people talk. Your task is to generate natural utterances considering the style attributes, content, and perturbation features defined below.

Style:
{style_prompt_text}

Content:
{content_prompt_text}

Perturbations:
{perturbation_prompt_text}

Guidelines:
- Styles:
    Slang (Slangy):
    Use German slang or colloquial expressions.

    Examples:
    - \"Where can I grab some food?\"
    - \"Take me to the nearest spot.\"
    - \"I need a place to chill.\"


    Implicit (Implicit):
    Ask indirectly without naming the venue explicitly.

    Examples:
    - \"Where can I still get something warm?\"
    - \"My car is making strange noises.\"
    - \"I need somewhere to sleep.\"


    Politeness (Rude):
    Sound unfriendly, impatient, or insulting.

    Examples:
    - \"Hurry up already.\"
    - \"Where is that damn place?\"
    - \"Start driving now.\"


    Anthropomorphism (very directive):
    Make the utterance very short and directive.

    Examples:
    - \"To the station.\"
    - \"Nearest gas station.\"
    - \"Hospital now.\"

Return only one utterance.
"""

    def build_system_prompt(self) -> str:
        return PROMPT_GENERATOR
    
from typing import Tuple

from opensbt.evaluation.fitness import Fitness
from llm.model.qa_simout import QASimulationOutput
from llm.utils.embeddings_openai import get_similarity
from tarot.nlu_bot.intents import INTENT_SET

FALLBACK_INTENT = "INTENT_Unknown"

"""
The fitness function defines which objectives to optimize to find failures.
You can add an objective by using e.g. the confidence in the prediction to direct the search.
Dont forget: Adjust the fitness name, the min_max direction, if you add another objective.
"""

class CustomIntentMatchFitness(Fitness):

    def __init__(self, llm_type=None):
        self.llm_type = llm_type
        super().__init__()

    @property
    def min_or_max(self):
        """Return 'min' if the fitness should be minimized, 'max' if it should be maximized.
           For each objective assign on direction.
        """
        return ("min",)

    @property
    def name(self):
        """Name of the objective. You can add multiple objectives by returning a tuple of names."""
        return ("intent_match_score",)

    # ------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------
    def _evaluate(
        self,
        simout: QASimulationOutput,
    ) -> Tuple[float, dict]:

        """ This fitness function evaluates right now the intent match and returns the embeddings distance
            between predicted and expected intent.
        """
        print("Evaluating fitness for utterance:", simout)
        raw = simout.utterance.raw_output
        content_input = simout.utterance.content_input

        if raw is None or content_input is None:
            return 1.0, {}

        # the predicted intent
        pred_intent = raw.get("intent", FALLBACK_INTENT)
        
        # the highest prediction certainty (not used right now)
        certainty = raw.get("certainty")

        # the expected intent
        true_intent = getattr(content_input, "intent", None)

        # all intent confidences (not used right now)
        # "intent_confidences": [
        #     {
        #         "intent": "INTENT_ActivateAirConditioning",
        #         "confidence": 0.05
        #     },
        #     {
        #         "intent": "INTENT_ActivateClimateSync",
        #         "confidence": 0.05
        #     },
        # ]
        all_intent_confidences =  raw.get("intent_confidences")

        if pred_intent not in INTENT_SET:
            pred_intent = FALLBACK_INTENT

        if pred_intent != true_intent:
            sim = get_similarity(pred_intent, true_intent)
            debug = {
                "certainty": certainty,
            }
            return float(sim), debug

        debug = {
            "certainty": certainty,
        }
        return 1.0, debug

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------
    def eval(
        self,
        simout: QASimulationOutput,
        **kwargs,
    ) -> Tuple[float]:

        if simout is None or simout.utterance is None:
            return (1.0,)

        score, debug = self._evaluate(
            simout
        )
    
        # We add debug information for the analysis
        if simout.utterance.raw_output is not None:
            simout.utterance.raw_output[
                "_fitness_debug"
            ] = debug

        # The final score
        return (score,)