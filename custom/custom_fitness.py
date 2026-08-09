from typing import Tuple

from opensbt.evaluation.fitness import Fitness
from llm.model.qa_simout import QASimulationOutput


"""This Fitness function uses the probability of the expected intent to be minimized to find failures.
"""
class CustomFitness(Fitness):

    @property
    def min_or_max(self):
        """Return 'min' if the fitness should be minimized, 'max' if it should be maximized.
           For each objective assign one direction.
        """
        return ("min",)

    @property
    def name(self):
        """Name of the objective. You can add multiple objectives by returning a tuple of names."""
        return ("probability_expected_intent",)

    def eval(self, simout: QASimulationOutput, **kwargs) -> Tuple[float]:
        """ This fitness function received the execution output object and
            return the intent probability of the expected intent.
        """
        if simout is None or simout.utterance is None:
            return (1.0,)

        # retrieve raw output
        raw_output = simout.utterance.raw_output or {}

        # retrieve content input
        content_input = simout.utterance.content_input

        expected_intent = getattr( content_input, "intent", None)
        probabilities = raw_output.get("probabilities", {})
        
        if expected_intent not in probabilities:
            return (1.0,)

        # assign score (take already provided probability)
        score = float(probabilities[expected_intent])
        
        raw_output["_fitness_debug"] = {
            "expected_intent": expected_intent,
            "predicted_intent": raw_output.get("intent"),
            "probabilities": probabilities,
        }

        return (score,)