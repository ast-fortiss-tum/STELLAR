from typing import Dict, List, Tuple
import json
import logging as log
import traceback
import re
from json_repair import repair_json

from opensbt.simulation.simulator import Simulator

from llm.model.qa_simout import QASimulationOutput
from llm.model.models import Utterance
from llm.llms import pass_llm, LLMType
from llm.config import LLM_IPA
from jupyter.nlu_bot.intents import INTENTS, INTENT_SET

FALLBACK_INTENT = "INTENT_Unknown"


class IPA_NLU_BOT(Simulator):
    memory: List[Utterance] = []
    ipa_name = "nlu_bot"
    TOP_K_INTENTS = 3

    @staticmethod
    def _build_system_prompt() -> str:
        intents_text = IPA_NLU_BOT._build_indexed_intents_text()
        few_shot_json = IPA_NLU_BOT._build_few_shot_example_json()
        return (
            "Role: You are an intent classifier for in-car assistant utterances.\n\n"
            "Task:\n"
            "For the given utterance, provide confidence scores for every allowed intent listed below.\n\n"
            "Allowed intents (index: name):\n"
            f"{intents_text}\n\n"
            "Output format (strict):\n"
            '1) Return exactly one JSON object.\n'
            '2) The JSON object must contain exactly one key: "intent_confidences".\n'
            '3) "intent_confidences" must be a list with exactly one object per allowed intent (no missing, no extra).\n'
            '4) Each list object must contain exactly these keys: "intent_index", "confidence".\n'
            f'5) "intent_index" must be an integer in [0, {len(INTENTS) - 1}].\n'
            '6) "intent_index" values must be unique.\n'
            '7) "confidence" must be a number in [0.0, 1.0].\n'
            '8) Confidence values are independent and do not need to sum to 1.0.\n'
            '9) Do not output markdown, code fences, explanations, or any extra text.\n\n'
            '10) INTENT_Unknown is a special intent that should be used when the utterance does not match any of the other intents.\n\n'
            "Few-shot example:\n"
            'Utterance: "increase the fan speed please"\n'
            f"Output: {few_shot_json}"
        )

    @staticmethod
    def _build_indexed_intents_text() -> str:
        return "\n".join(f"{idx}: {intent_name}" for idx, intent_name in enumerate(INTENTS))

    @staticmethod
    def _build_few_shot_example_json() -> str:
        example_intents: List[Dict[str, float]] = []
        for intent_index, intent_name in enumerate(INTENTS):
            confidence = 0.02
            if intent_name == "INTENT_Climate_IncreaseFanSpeed":
                confidence = 0.22
            elif intent_name == "INTENT_Climate_SetFanSpeed":
                confidence = 0.14
            elif intent_name == "INTENT_Climate_ActivateFan":
                confidence = 0.10
            example_intents.append({"intent_index": intent_index, "confidence": confidence})

        return json.dumps({"intent_confidences": example_intents}, separators=(",", ":"))

    @staticmethod
    def _intent_name_from_index(intent_index) -> str:
        try:
            idx = int(intent_index)
        except (TypeError, ValueError):
            return ""

        if 0 <= idx < len(INTENTS):
            return INTENTS[idx]

        return ""

    @staticmethod
    def _load_json_payload(text: str):
        if text is None:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            repaired = repair_json(text)
            return json.loads(repaired)
        except Exception:
            return None

    @staticmethod
    def _validate_intent_confidences(intent_confidences: List[Dict[str, float]]) -> Tuple[bool, str]:
        if not isinstance(intent_confidences, list):
            return False, "intent_confidences must be a list"

        if len(intent_confidences) != len(INTENTS):
            return False, f"intent_confidences must contain exactly {len(INTENTS)} items"

        seen_intents = set()
        for index, item in enumerate(intent_confidences):
            if not isinstance(item, dict):
                return False, f"item {index + 1} is not an object"

            intent_name = item.get("intent")
            if intent_name not in INTENT_SET:
                return False, f"item {index + 1} has invalid intent"
            if intent_name in seen_intents:
                return False, "intent names must be unique"
            seen_intents.add(intent_name)

            try:
                confidence = float(item.get("confidence"))
            except (TypeError, ValueError):
                return False, f"item {index + 1} confidence is not numeric"

            if not (0.0 <= confidence <= 1.0):
                return False, f"item {index + 1} confidence is out of range"

        if seen_intents != INTENT_SET:
            return False, "intent_confidences must include every allowed intent exactly once"

        return True, "ok"

    @staticmethod
    def _extract_intent_confidences(raw_output: str) -> List[Dict[str, float]]:
        if raw_output is None:
            return []

        payload = IPA_NLU_BOT._load_json_payload(str(raw_output).strip())
        if payload is None:
            return []

        if not isinstance(payload, dict):
            return []

        payload_candidates = payload.get("intent_confidences")
        normalized: List[Dict[str, float]] = []

        def append_candidate(intent_name, confidence):
            if intent_name not in INTENT_SET:
                return
            normalized.append({"intent": intent_name, "confidence": confidence})

        if isinstance(payload_candidates, list):
            for candidate in payload_candidates:
                if not isinstance(candidate, dict):
                    continue
                if "intent_index" in candidate:
                    intent_name = IPA_NLU_BOT._intent_name_from_index(candidate.get("intent_index"))
                    append_candidate(intent_name, candidate.get("confidence"))
                    continue

                # Backward compatibility with older "intent" outputs.
                append_candidate(candidate.get("intent"), candidate.get("confidence"))

            return normalized

        if isinstance(payload_candidates, dict):
            # Support map-style output keyed either by index or by intent name.
            for key, confidence in payload_candidates.items():
                intent_name = ""
                if isinstance(key, str) and key in INTENT_SET:
                    intent_name = key
                else:
                    intent_name = IPA_NLU_BOT._intent_name_from_index(key)
                append_candidate(intent_name, confidence)

            return normalized

        return []

    @staticmethod
    def _coerce_confidence(value, default: float) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = default
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _top_k_from_intent_confidences(intent_confidences: List[Dict[str, float]], k: int = None) -> List[Dict[str, float]]:
        if k is None:
            k = IPA_NLU_BOT.TOP_K_INTENTS
        sorted_candidates = sorted(
            intent_confidences,
            key=lambda item: IPA_NLU_BOT._coerce_confidence(item.get("confidence", 0.0), 0.0),
            reverse=True,
        )
        return sorted_candidates[:k]

    @staticmethod
    def _uniform_intent_confidences() -> List[Dict[str, float]]:
        share = 1.0 / len(INTENTS)
        return [{"intent": intent_name, "confidence": share} for intent_name in INTENTS]

    @staticmethod
    def _normalize_intent(raw_output: str) -> str:
        if raw_output is None:
            return FALLBACK_INTENT

        text = str(raw_output).strip()
        payload = IPA_NLU_BOT._load_json_payload(text)

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
        payload = IPA_NLU_BOT._load_json_payload(text)

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
    def _fallback_intent_confidences(primary_intent: str, primary_confidence: float) -> List[Dict[str, float]]:
        if primary_intent not in INTENT_SET:
            return IPA_NLU_BOT._uniform_intent_confidences()

        target_intent = primary_intent if primary_intent in INTENT_SET else INTENTS[0]
        target_confidence = IPA_NLU_BOT._coerce_confidence(primary_confidence, 1.0)
        residual = max(0.0, 1.0 - target_confidence)
        spread = residual / max(1, len(INTENTS) - 1)

        intent_confidences: List[Dict[str, float]] = []
        for intent_name in INTENTS:
            if intent_name == target_intent:
                confidence = target_confidence
            else:
                confidence = spread
            intent_confidences.append({"intent": intent_name, "confidence": confidence})
        return intent_confidences

    @staticmethod
    def _fallback_top_intents(primary_intent: str, primary_confidence: float) -> List[Dict[str, float]]:
        return IPA_NLU_BOT._top_k_from_intent_confidences(
            IPA_NLU_BOT._fallback_intent_confidences(primary_intent, primary_confidence)
        )

    @staticmethod
    def _normalize_intent_confidences(raw_output: str) -> List[Dict[str, float]]:
        if raw_output is None:
            return IPA_NLU_BOT._uniform_intent_confidences()

        candidates = IPA_NLU_BOT._extract_intent_confidences(raw_output)

        confidence_by_intent: Dict[str, float] = {intent_name: 0.0 for intent_name in INTENTS}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            intent_name = candidate.get("intent")
            if intent_name not in INTENT_SET:
                continue
            confidence_by_intent[intent_name] = IPA_NLU_BOT._coerce_confidence(
                candidate.get("confidence"),
                confidence_by_intent[intent_name],
            )

        if any(value > 0.0 for value in confidence_by_intent.values()):
            return [
                {"intent": intent_name, "confidence": confidence_by_intent[intent_name]}
                for intent_name in INTENTS
            ]

        primary_intent = IPA_NLU_BOT._normalize_intent(raw_output)
        primary_confidence = IPA_NLU_BOT._normalize_certainty(raw_output, primary_intent)
        return IPA_NLU_BOT._fallback_intent_confidences(primary_intent, primary_confidence)

    @staticmethod
    def _parse_prediction(raw_output: str) -> Tuple[str, float, List[Dict[str, float]], List[Dict[str, float]]]:
        intent_confidences = IPA_NLU_BOT._normalize_intent_confidences(raw_output)
        top_intents = IPA_NLU_BOT._top_k_from_intent_confidences(intent_confidences)
        if top_intents:
            intent = top_intents[0]["intent"]
            certainty = top_intents[0]["confidence"]
        else:
            intent = FALLBACK_INTENT
            certainty = 0.0
            intent_confidences = IPA_NLU_BOT._fallback_intent_confidences(intent, 1.0)
            top_intents = IPA_NLU_BOT._top_k_from_intent_confidences(intent_confidences)
        return intent, certainty, top_intents, intent_confidences

    @staticmethod
    def _build_raw_output(
        raw_output: str,
        intent: str,
        certainty: float,
        top_intents: List[Dict[str, float]],
        intent_confidences: List[Dict[str, float]],
    ):
        return {
            "llm_output": raw_output,
            "intent": intent,
            "certainty": certainty,
            "top_intents": top_intents,
            "intent_confidences": intent_confidences,
        }

    @staticmethod
    def _ensure_prediction_metadata(utterance: Utterance):
        if isinstance(utterance.raw_output, dict):
            if (
                "certainty" in utterance.raw_output
                and "top_intents" in utterance.raw_output
                and "intent_confidences" in utterance.raw_output
            ):
                utterance.raw_output.pop("confidence", None)
                return

            raw_text = utterance.raw_output.get("llm_output")
            intent = utterance.raw_output.get("intent") or utterance.answer or FALLBACK_INTENT
        else:
            raw_text = utterance.raw_output
            intent = utterance.answer or FALLBACK_INTENT

        _, certainty, top_intents, intent_confidences = IPA_NLU_BOT._parse_prediction(raw_text)
        if top_intents:
            intent = top_intents[0]["intent"]
        utterance.answer = intent
        utterance.raw_output = IPA_NLU_BOT._build_raw_output(
            raw_text,
            intent,
            certainty,
            top_intents,
            intent_confidences,
        )

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

                        log.info(
                            f"[IPA_NLU_BOT] Attempt {attempt + 1}/{max_attempts} raw LLM output: {llm_output}"
                        )
                        
                        raw_intent_confidences = IPA_NLU_BOT._extract_intent_confidences(llm_output)
                        is_valid, invalid_reason = IPA_NLU_BOT._validate_intent_confidences(raw_intent_confidences)
                        if not is_valid:
                            log.warning(
                                f"[IPA_NLU_BOT] Validation failed before parse: {invalid_reason}. "
                                f"Raw output: {llm_output}"
                            )
                            raise ValueError(
                                f"Invalid LLM prediction format: {invalid_reason}. Raw output: {llm_output}"
                            )
                        if invalid_reason != "ok":
                            log.info(f"[IPA_NLU_BOT] Validation note: {invalid_reason}")

                        intent, certainty, top_intents, intent_confidences = IPA_NLU_BOT._parse_prediction(llm_output)

                        utterance.answer = intent
                        utterance.raw_output = IPA_NLU_BOT._build_raw_output(
                            llm_output, intent, certainty, top_intents, intent_confidences
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
                    fallback_confidences = IPA_NLU_BOT._uniform_intent_confidences()
                    utterance.answer = FALLBACK_INTENT
                    utterance.raw_output = {
                        "error": "Inference failed after retries",
                        "question": utterance.question,
                        "intent": FALLBACK_INTENT,
                        "certainty": 0.0,
                        "top_intents": IPA_NLU_BOT._top_k_from_intent_confidences(fallback_confidences),
                        "intent_confidences": fallback_confidences,
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
