"""The system under test."""
from __future__ import annotations
import re
from typing import List

from opensbt.simulation.simulator import Simulator
from llm.model.qa_simout import QASimulationOutput
from llm.model.models import Utterance
from custom.custom_models import CustomContentInput

from custom.custom_models import CustomOutputModel

class CustomSUT(Simulator):
    """Classifies utterances' intents by keywords, with an intentional navigation bias."""

    ipa_name = "custom_keyword_intent_classifier"

    @staticmethod
    def _predict(text_input: str) -> CustomOutputModel:
        """This method is implementation specific. Here: Returns probabilities from keyword counts to mock NLU.
           Implement this function to pass the text_input to your SUT.
        """

        CLIMATE_KEYWORDS = {"ac", "air", "conditioning", "climate", "fan", "heat", "heating", "cold", "temperature"}
        NAVIGATION_KEYWORDS = {"navigate", "navigation", "route", "directions", "destination", "map", "drive"}

        tokens = re.findall(r"[a-z]+", text_input.lower())
        climate_hits = sum(token in CLIMATE_KEYWORDS for token in tokens)
        navigation_hits = sum(token in NAVIGATION_KEYWORDS for token in tokens)
        total_hits = climate_hits + navigation_hits

        if total_hits == 0:
            probabilities = {"INTENT_Climate": 0.5, "INTENT_Navigation": 0.5}
        else:
            probabilities = {
                "INTENT_Climate": climate_hits / total_hits,
                "INTENT_Navigation": navigation_hits / total_hits,
            }

        intent = max(probabilities, key=probabilities.get)  # This will now be "INTENT_Climate" or "INTENT_Navigation"
        return CustomOutputModel(
            intent=intent,
            probabilities=probabilities,
            score=probabilities[intent],
        )

    @staticmethod
    def simulate(
        list_individuals: List[List[Utterance]],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float = 10,
        do_visualize: bool = False,
        temperature: float = 0,
        context: object = None,
        llm_type=None,
        **kwargs,
    ) -> List[QASimulationOutput]:
        results: List[QASimulationOutput] = []
        """ Implement this function to run the execution for a given list of utterances and return the results. """

        for utterance_group in list_individuals:
            utterance = utterance_group[0]

            # Run the classifier; modify this call or underlying function based on your use case
            output = CustomSUT._predict(utterance.question)

            # The framework collects the `answer` and the `raw_output`.
            utterance.answer = output.intent
            utterance.raw_output = output.model_dump()

            # For each utterance a QASimulationOutput instance is generated
            results.append(
                QASimulationOutput(
                    utterance=utterance,
                    model="None",
                    ipa=CustomSUT.ipa_name,
                )
            )

        return results

if __name__ == "__main__":
    import json
    utterance = Utterance(
        question="Turn on the AC please",
        seed=None,
        ordinal_vars=[],
        categorical_vars=[],
        content_input=CustomContentInput(intent="INTENT_Climate"),
    )

    print("Utterance:")
    print(utterance.model_dump_json(indent=2))

    results = CustomSUT.simulate(
        list_individuals=[[utterance]],   # one individual with one utterance
        variable_names=["utterance"],
        scenario_path=".",
        sim_time=1.0,
    )
    print("Response:")
    print(json.dumps(results[0].utterance.raw_output, indent=2))