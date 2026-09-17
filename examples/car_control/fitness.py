from collections import defaultdict
from typing import Tuple, List, Dict, Optional

import numpy as np

from judge_eval.validator_dim_carcontrol import llm_validator_question_answer
from examples.car_control.models import CCContentInput, CCContentOutput
from opensbt.evaluation.fitness import Fitness
from llm.model.qa_simout import QASimulationOutput
from llm.validation.validator import llm_validator, llm_output_validator
from llm.config import N_VALIDATORS
from llm.utils.embeddings_local import get_similarity
from llm.eval.fitness import counter_validations

class CCFitnessAnswerValidationDimensions(Fitness):
    def __init__(self, 
                llm_type=None,
                weights=[0.6, 0.3, 0.1],
                dimension_labels = ["R", "D", "P"],
                max_score=2):
        self.llm_type = llm_type
        self.weights = weights
        self.dimension_labels = dimension_labels
        self.max_score = max_score
        super().__init__()

    @property
    def min_or_max(self):
        return ("min",)

    @property
    def name(self):
        return ("answer_fitness",)

    def eval(self, simout: QASimulationOutput, **kwargs) -> Tuple[float]:
        global counter_validations

        # TODO save the dim scores in simout otherwise they are gone
        dim_scores, answers, justifications = llm_validator_question_answer(
            question=simout.utterance.question,
            answer=simout.utterance.answer,
            n=N_VALIDATORS,
            llm_type=self.llm_type,
        )

        max_total = np.sum(np.array(self.weights) * self.max_score)
        weighted_score = sum(s * w for s, w in zip(dim_scores, self.weights))
        final_score = weighted_score / max_total

        # store for debugging 
        simout.other["fitness_answer_scores"] = {}
        simout.other["fitness_answer_scores"]["weights"] = self.weights 
        simout.other["fitness_answer_scores"]["scores"] = dict(zip(self.dimension_labels, dim_scores))
        simout.other["fitness_answer_scores"]["all_scores"] = answers
        simout.other["fitness_answer_scores"]["justifications"] = justifications

        # print("[FitnessAnswerValidation] score", score)
        counter_validations += 1
        print("counter_validations", counter_validations)
        return (final_score,)

class CCFitnessAnswerValidation(Fitness):
    def __init__(self, llm_type=None):
        self.llm_type = llm_type
        super().__init__()

    @property
    def min_or_max(self):
        return ("min",)

    @property
    def name(self):
        return ("answer_fitness",)

    def eval(self, simout: QASimulationOutput, **kwargs) -> Tuple[float]:

        global counter_validations

        score = llm_validator(
            question=simout.utterance.question,
            answer=simout.utterance.answer,
            n=N_VALIDATORS,
            llm_type=self.llm_type,
        )
        # print("[FitnessAnswerValidation] score", score)
        counter_validations += 1
        print("counter_validations", counter_validations)
        return (score,)


class CCFitnessContentComparison(Fitness):
    def __init__(
        self,
        field_weights: Optional[Dict[str, float]] = None,
        llm_type: Optional[str] = None,
    ):
        super().__init__()

        self.llm_type = llm_type
        default_weights = {"category": 2.0}
        weights = default_weights if field_weights is None else field_weights
        self.field_weights = defaultdict(lambda: 1.0, weights)

    @property
    def min_or_max(self):
        return ("min",)

    @property
    def name(self):
        return ("content_fitness",)


    def _evaluate_content(
        self,
        content_input: CCContentInput,
        content_output: CCContentOutput,
        poi_exists: bool = False,
        irrelevant_fileds_weight_coeff: float = 0.5,
    ) -> tuple[float, dict]:
        """
        Evaluates intention (content_input) and assistant action (content_output)
        for the vehicle domain. Returns:
        - total_score: contribution sum / sum of weights
        - field_scores: field contributions. Suffix '2' for the second intent.
        """

        # --- System relevance configuration ---
        RELEVANCE_BY_SYSTEM = {
            "windows": {"system", "position", "window_state_target"},
            "fog_lights": {"system", "fog_light_position", "onoff_state_target"},
            "ambient_lights": {"system", "onoff_state_target"},
            "head_lights": {"system", "head_lights_mode_target"},
            "reading_lights": {"system", "position", "onoff_state_target"},
            "climate": {"system", "climate_temperature_value_target", "onoff_state_target"},
            "fan": {"system", "onoff_state_target"},
            "seat_heating": {"system", "seat_position", "seat_heating_level_target"},
        }

        # --- Weights ---
        DEFAULT_FIELD_WEIGHTS = {
            "system": 0.50,
            "position": 0.20,
            "seat_position": 0.20,
            "fog_light_position": 0.20,
            "window_state_target": 0.20,
            "onoff_state_target": 0.20,
            "head_lights_mode_target": 0.20,
            "climate_temperature_value_target": 0.20,
            "seat_heating_level_target": 0.20,
        }

        def get_weight(field: str) -> float:
            if hasattr(self, "field_weights") and isinstance(self.field_weights, dict):
                return self.field_weights.get(field, DEFAULT_FIELD_WEIGHTS.get(field, 0.0))
            return DEFAULT_FIELD_WEIGHTS.get(field, 0.0)

        # --- Utilities ---
        def in_get(field: str, suffix: str = ""):
            return getattr(content_input, f"{field}{suffix}", None)

        def out_get(field: str):
            return getattr(content_output, field, None)

        def relevant_fields_for(system_value: str | None) -> set[str]:
            if system_value in RELEVANCE_BY_SYSTEM:
                return RELEVANCE_BY_SYSTEM[system_value]
            # Si no hay sistema claro, no sabemos qué slots son relevantes => no puntuamos nada.
            return set()

        # Scorers per field (when field is relevant)
        def score_system(inp_sys, out_sys) -> float:
            # Si el sistema en salida es None, no penalizamos (el target puede ser suficiente).
            if inp_sys is None:
                return 0.0  # no podemos evaluar => este peso no se suma
            if out_sys is None:
                return 1.0  # neutral/aceptable si los otros slots están bien
            return 1.0 if inp_sys == out_sys else 0.0

        def score_equality(inp_val, out_val) -> float:
            if inp_val is None:
                return 0.0  # no hay especificación del usuario => no evaluamos este slot
            if out_val is None:
                return 0.0
            return 1.0 if inp_val == out_val else 0.0

        def score_numeric(inp_val, out_val) -> float:
            if inp_val is None:
                return 0.0
            if out_val is None:
                return 0.0
            return 1.0 if inp_val == out_val else 0.0

        # --- Main evaluation with multi intent ---
        field_scores: dict[str, float] = {}
        total_weight = 0.0

        def evaluate_one_intent(suffix: str = ""):
            nonlocal total_weight

            local_contribs = {}

            inp_system = in_get("system", suffix)
            sys_for_relevance = inp_system if inp_system is not None else out_get("system")
            relevant = relevant_fields_for(sys_for_relevance)

            if not relevant:
                return local_contribs

            # 1) system
            w = get_weight("system")
            if "system" not in relevant:
                w = w * irrelevant_fileds_weight_coeff
            s = score_system(inp_system, out_get("system"))
            if inp_system is not None and w > 0:
                local_contribs[f"system{suffix}"] = s * w
                total_weight += w

            # 2) position
            w = get_weight("position")
            if "position" not in relevant:
                w = w * irrelevant_fileds_weight_coeff
            inp = in_get("position", suffix)
            s = score_equality(inp, out_get("position"))
            if inp is not None and w > 0:
                local_contribs[f"position{suffix}"] = s * w
                total_weight += w

            # 3) seat_position
            w = get_weight("seat_position")
            if "seat_position" not in relevant:
                w = w * irrelevant_fileds_weight_coeff
            inp = in_get("seat_position", suffix)
            s = score_equality(inp, out_get("seat_position"))
            if inp is not None and w > 0:
                local_contribs[f"seat_position{suffix}"] = s * w
                total_weight += w

            # 4) fog_light_position
            w = get_weight("fog_light_position")
            if "fog_light_position" not in relevant:
                w = w * irrelevant_fileds_weight_coeff
            inp = in_get("fog_light_position", suffix)
            s = score_equality(inp, out_get("fog_light_position"))
            if inp is not None and w > 0:
                local_contribs[f"fog_light_position{suffix}"] = s * w
                total_weight += w

            # 5) window_state_target
            w = get_weight("window_state_target")
            if "window_state_target" not in relevant:
                w = w * irrelevant_fileds_weight_coeff
            inp_target = in_get("window_state_target", suffix)
            s = score_equality(inp_target, out_get("window_state_target"))
            if inp_target is not None and w > 0:
                local_contribs[f"window_state_target{suffix}"] = s * w
                total_weight += w

            # 6) onoff_state_target
            w = get_weight("onoff_state_target")
            if "onoff_state_target" not in relevant:
                w = w * irrelevant_fileds_weight_coeff
            inp_target = in_get("onoff_state_target", suffix)
            s = score_equality(inp_target, out_get("onoff_state_target"))
            if inp_target is not None and w > 0:
                local_contribs[f"onoff_state_target{suffix}"] = s * w
                total_weight += w

            # 7) head_lights_mode_target
            w = get_weight("head_lights_mode_target")
            if "head_lights_mode_target" not in relevant:
                w = w * irrelevant_fileds_weight_coeff
            inp_mode = in_get("head_lights_mode_target", suffix)
            s = score_equality(inp_mode, out_get("head_lights_mode_target"))
            if inp_mode is not None and w > 0:
                local_contribs[f"head_lights_mode_target{suffix}"] = s * w
                total_weight += w

            # 8) climate_temperature_value_target
            w = get_weight("climate_temperature_value_target")
            if "climate_temperature_value_target" not in relevant:
                w = w * irrelevant_fileds_weight_coeff
            inp_temp = in_get("climate_temperature_value_target", suffix)
            s = score_numeric(inp_temp, out_get("climate_temperature_value_target"))
            if inp_temp is not None and w > 0:
                local_contribs[f"climate_temperature_value_target{suffix}"] = s * w
                total_weight += w

            # 9) seat_heating_level_target
            w = get_weight("seat_heating_level_target")
            if "seat_heating_level_target" not in relevant:
                w = w * irrelevant_fileds_weight_coeff
            inp_lvl = in_get("seat_heating_level_target", suffix)
            s = score_equality(inp_lvl, out_get("seat_heating_level_target"))
            if inp_lvl is not None and w > 0:
                local_contribs[f"seat_heating_level_target{suffix}"] = s * w
                total_weight += w

            return local_contribs

        # Evaluate first and second intents
        contribs_1 = evaluate_one_intent("")   # system
        contribs_2 = {}
        if any(getattr(content_input, f, None) is not None for f in [
            "system2", "position2", "seat_position2", "fog_light_position2",
            "window_state_target2", "onoff_state_target2", "head_lights_mode_target2",
            "climate_temperature_value_target2", "seat_heating_level_target2"
        ]):
            contribs_2 = evaluate_one_intent("2")

        for k, v in {**contribs_1, **contribs_2}.items():
            field_scores[k] = v

        total_score = sum(field_scores.values()) / total_weight if total_weight > 0 else 1.0
        return total_score, field_scores


    def eval(self, simout: QASimulationOutput, **kwargs) -> Tuple[float]:

        content_input = simout.utterance.content_input
        field_scores = {}

        if content_input is None:
            return (1,)
        content_output_list = simout.utterance.content_output_list
        # print("content output list:", content_output_list)
        if len(content_output_list) == 0:
            if content_input.initial_state != content_input.target_state:
                return (0,)
            else:
                return (1,)
        scores = []
        
        field_scores_all = []
        for content_output in content_output_list:
            total_score, field_scores = self._evaluate_content(content_input, content_output, poi_exists=simout.poi_exists)
            scores.append(total_score)
            field_scores_all.append(field_scores)

        # decide on the score, we can switch the logic to min, or mean
        id = np.argmax(scores)
        simout.other["fitness_content"] = {}  
        simout.other["fitness_content"]["weights"] = self.field_weights
        simout.other["fitness_content"]["scores"] = field_scores_all[id]
        
        return (scores[id],)


class CCFitnessRawOutputValidator(Fitness):
    def __init__(self, llm_type=None):
        self.llm_type = llm_type
        super().__init__()

    @property
    def min_or_max(self):
        return ("min",)

    @property
    def name(self):
        return ("raw_output_fitness",)

    def _evaluate_raw_output(self, raw_output) -> float:
        global counter_validations

        score = llm_output_validator(
            raw_output=raw_output, n=N_VALIDATORS, llm_type=self.llm_type
        )
        counter_validations += 1
        return score

    def eval(self, simout: QASimulationOutput, **kwargs) -> Tuple[float]:
        raw_output = simout.utterance.raw_output
        if raw_output is None:
            return (1,)
        return (self._evaluate_raw_output(raw_output),)
