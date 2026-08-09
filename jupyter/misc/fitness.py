from typing import Tuple

from opensbt.evaluation.fitness import Fitness
from llm.model.qa_simout import QASimulationOutput
from llm.utils.embeddings_openai import get_similarity
from jupyter.nlu_bot.intents import INTENT_SET

FALLBACK_INTENT = "INTENT_Unknown"

"""
The fitness function defines which objectives to optimize to find failures.
You can add an objective by using e.g. the confidence in the prediction to direct the search.
Dont forget: Adjust the fitness name, the min_max direction, if you add another objective.
"""

class IntentMatchFitness(Fitness):

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

        raw = simout.utterance.raw_output
        content_input = simout.utterance.content_input

        if raw is None or content_input is None:
            return 1.0, {}

        # the predicted intent
        pred_intent = raw.get("intent", FALLBACK_INTENT)
        
        # the prediction certainty (not used right now)
        certainty = raw.get("certainty")

        # the expected intent
        true_intent = getattr(content_input, "intent", None)

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