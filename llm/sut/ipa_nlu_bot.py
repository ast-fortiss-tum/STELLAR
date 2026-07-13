from typing import List, Tuple
import json
import logging as log
import traceback
import re

from opensbt.simulation.simulator import Simulator

from llm.model.qa_simout import QASimulationOutput
from llm.model.models import Utterance
from llm.llms import pass_llm, LLMType
from llm.config import LLM_IPA
from tarot.nlu_bot.intents import INTENTS, INTENT_SET

FALLBACK_INTENT = "INTENT_Unknown"


class IPA_NLU_BOT(Simulator):
    memory: List[Utterance] = []
    ipa_name = "nlu_bot"

    @staticmethod
    def _build_system_prompt() -> str:
        intents_text = "\n".join(INTENTS)
        return (
            "You are an intent classifier for in-car assistant utterances. "
            "Map the user utterance to exactly one intent from the allowed list. "
            "If no intent fits, return INTENT_Unknown. Also estimate how certain you are on a scale from 0.0 to 1.0.\n\n"
            "Allowed intents:\n"
            f"{intents_text}\n\n"
            "Output rules (strict):\n"
            '1) Return exactly one JSON object.\n'
            '2) The JSON object must contain exactly these keys: "intent", "certainty".\n'
            '3) "intent" must be one allowed intent or INTENT_Unknown.\n'
            '4) "certainty" must be a number between 0.0 and 1.0.\n'
            '5) No markdown, no code fences, no explanation, no extra text.\n'
            '6) Example: {"intent": "INTENT_Gen_Hello", "certainty": 0.82}'
        )

    @staticmethod
    def _normalize_intent(raw_output: str) -> str:
        if raw_output is None:
            return FALLBACK_INTENT

        text = str(raw_output).strip()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            candidate = payload.get("intent")
            if candidate in INTENT_SET:
                return candidate

        if text in INTENT_SET:
            return text

        match = re.search(r"INTENT_[A-Za-z0-9_]+", text)
        if match:
            candidate = match.group(0)
            if candidate in INTENT_SET:
                return candidate

        return FALLBACK_INTENT

    @staticmethod
    def _normalize_certainty(raw_output: str, intent: str) -> float:
        if raw_output is None:
            return 0.0

        text = str(raw_output).strip()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            try:
                certainty = float(payload.get("certainty"))
                return max(0.0, min(1.0, certainty))
            except (TypeError, ValueError):
                pass

        match = re.search(r"CERTAINTY\s*:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        if match:
            try:
                certainty = float(match.group(1))
                return max(0.0, min(1.0, certainty))
            except ValueError:
                pass

        number_matches = re.findall(r"([0-9]*\.?[0-9]+)", text)
        for candidate in number_matches:
            try:
                certainty = float(candidate)
            except ValueError:
                continue
            if 0.0 <= certainty <= 1.0:
                return certainty

        return 0.0 if intent == FALLBACK_INTENT else 0.5

    @staticmethod
    def _parse_prediction(raw_output: str) -> Tuple[str, float]:
        intent = IPA_NLU_BOT._normalize_intent(raw_output)
        certainty = IPA_NLU_BOT._normalize_certainty(raw_output, intent)
        return intent, certainty

    @staticmethod
    def _build_raw_output(raw_output: str, intent: str, certainty: float):
        return {
            "llm_output": raw_output,
            "intent": intent,
            "certainty": certainty,
        }

    @staticmethod
    def _ensure_prediction_metadata(utterance: Utterance):
        if isinstance(utterance.raw_output, dict):
            if "certainty" in utterance.raw_output:
                utterance.raw_output.pop("confidence", None)
                return

            raw_text = utterance.raw_output.get("llm_output")
            intent = utterance.raw_output.get("intent") or utterance.answer or FALLBACK_INTENT
        else:
            raw_text = utterance.raw_output
            intent = utterance.answer or FALLBACK_INTENT

        _, certainty = IPA_NLU_BOT._parse_prediction(raw_text)
        utterance.raw_output = IPA_NLU_BOT._build_raw_output(raw_text, intent, certainty)

    @staticmethod
    def _check_utterance_in_mem(utterance: Utterance):
        for cached in IPA_NLU_BOT.memory:
            if cached.question == utterance.question:
                return True, cached
        return False, utterance

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
        llm_type=LLMType(LLM_IPA),
    ) -> List[QASimulationOutput]:

        results = []
        log.info(f"[IPA_NLU_BOT] list_individuals: {list_individuals}")

        for utterance_group in list_individuals:
            utterance = utterance_group[0]

            in_memory, memory_utterance = IPA_NLU_BOT._check_utterance_in_mem(utterance)

            if in_memory:
                log.info(
                    f"[IPA_NLU_BOT] Utterance already in memory: {utterance.question}"
                )
                utterance.answer = memory_utterance.answer
                utterance.raw_output = memory_utterance.raw_output
                IPA_NLU_BOT._ensure_prediction_metadata(utterance)
            else:
                max_attempts = 5
                attempt = 0

                while attempt < max_attempts:
                    try:
                        llm_output = pass_llm(
                            msg=utterance.question,
                            llm_type=llm_type,
                            temperature=temperature,
                            context=context,
                            system_message=IPA_NLU_BOT._build_system_prompt(),
                        )

                        intent, certainty = IPA_NLU_BOT._parse_prediction(llm_output)

                        utterance.answer = intent
                        utterance.raw_output = IPA_NLU_BOT._build_raw_output(
                            llm_output, intent, certainty
                        )

                        log.info(
                            f"[IPA_NLU_BOT] Success: {utterance.question} -> {intent} ({certainty:.2f})"
                        )
                        break

                    except Exception as exc:
                        traceback.print_exc()
                        log.error(
                            f"[IPA_NLU_BOT] Attempt {attempt + 1} failed for utterance: {utterance.question}"
                        )
                        log.error(str(exc))
                    attempt += 1

                if utterance.answer is None:
                    utterance.answer = FALLBACK_INTENT
                    utterance.raw_output = {
                        "error": "Inference failed after retries",
                        "question": utterance.question,
                        "intent": FALLBACK_INTENT,
                        "certainty": 0.0,
                    }

                IPA_NLU_BOT.memory.append(utterance)

            model_name = llm_type.value if isinstance(llm_type, LLMType) else str(llm_type)
            result = QASimulationOutput(
                utterance=utterance,
                model=model_name,
                ipa=IPA_NLU_BOT.ipa_name,
            )
            results.append(result)

        return results
