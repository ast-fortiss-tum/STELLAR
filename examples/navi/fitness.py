from collections import defaultdict
from typing import Tuple, Dict, Optional

import numpy as np

from judge_eval.validator_dim import llm_validator_question_answer
from examples.navi.models import NaviContentInput, NaviContentOutput
from opensbt.evaluation.fitness import Fitness
from llm.model.qa_simout import QASimulationOutput
from llm.config import N_VALIDATORS
from llm.utils.embeddings_local import get_similarity
from llm.eval.fitness import counter_validations

class NaviFitnessAnswerValidationDimensions(Fitness):
    def __init__(self, 
                llm_type=None,
                weights=[1, 0, 0.5],
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

class NaviFitnessContentComparison(Fitness):
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

    @staticmethod
    def _is_correct_payment_method(
        required_payment_method: str, found_payment_method: str
    ) -> bool:
        keywords_dict = (
            {
                "CASH": ["cash"],
                "DEBIT": ["debit"],
                "CREDIT_CARD_MASTER": ["master"],
                "CREDIT_CARD_VISA": ["visa"],
                "CONTACTLESS": ["contactless"],
                "MOBILE_PAYMENT": ["mobile", "phone"],
            },
        )
        if required_payment_method not in keywords_dict:  # unsupported for evaluation
            return True
        for keyword in keywords_dict[required_payment_method]:
            if keyword in found_payment_method.lower():
                return True
        return False

    def _evaluate_content(
        self,
        content_input: NaviContentInput,
        content_output: NaviContentOutput,
        poi_exists: bool = False
    ) -> dict:
        """
        Evaluate the content and return per-field contribution scores.
        
        Returns a dictionary with:
            - keys: input field names that contributed to the score
            - values: score contributed by that field (weighted)
            - 'total_score': normalized total score (sum of contributions / total weights)
        """
        field_scores = {}
        total_weight = 0.0

        # Define scoring functions for each field
        def score_category(inp, out):
            if not out.categories and not poi_exists:
                return 1.0
            elif out.categories:
                return max((1 + get_similarity(inp.category, c)) / 2 for c in out.categories)
            return 0.0

        def score_business_hours(inp, out):
            return 1.0 if out.business_hours_status is None or \
                        out.business_hours_status.lower() == inp.business_hours_status.lower() else 0.0

        def score_payment_method(inp, out):
            if out.payment_methods is None:
                return 1.0
            return 1.0 if any(self._is_correct_payment_method(inp.payment_method, pm) for pm in out.payment_methods) else 0.0

        def score_fuel_type(inp, out):
            if out.fuel_types is None:
                return 1.0
            return 1.0 if inp.fuel_type in out.fuel_types else 0.0

        def score_fuel_price(inp, out):
            if out.fuel_prices is None:
                return 1.0
            return 1.0 if inp.fuel_price >= out.fuel_prices.get(inp.fuel_type, float("inf")) else 0.0

        def score_food_type(inp, out):
            print("out:",out)
            if not out.food_types:
                return 1.0
            return max((1 + get_similarity(inp.food_type, ft)) / 2 for ft in out.food_types)

        def score_parking(inp, out):
            return 1.0 if inp.parking == "available" else 0.0

        def score_price_range(inp, out):
            if out.price_range is None:
                return 1.0
            return 1.0 if inp.price_range.lower() in out.price_range.lower() else 0.0

        def score_rating(inp, out):
            if out.rating is None:
                return 1.0
            score_rating = 1.0 if inp.rating <= out.rating else (inp.rating - out.rating)/1.5
            return score_rating
        
        # Map fields to their scoring functions
        scoring_rules = {
            "category": score_category,
            "business_hours_status": score_business_hours,
            "payment_method": score_payment_method,
            "fuel_type": score_fuel_type,
            "fuel_price": score_fuel_price,
            "food_type": score_food_type,
            "parking": score_parking,
            "price_range": score_price_range,
            "rating": score_rating
        }

        for field_name, scorer in scoring_rules.items():
            if getattr(content_input, field_name, None) is not None:
                weight = self.field_weights.get(field_name, 0)
                total_weight += weight
                contribution = scorer(content_input, content_output) * weight
                field_scores[field_name] = contribution

        total_score = sum(field_scores.values()) / total_weight if total_weight else 1.0

        return total_score, field_scores

    def eval(self, simout: QASimulationOutput, **kwargs) -> Tuple[float]:

        content_input = simout.utterance.content_input
        field_scores = {}

        if content_input is None:
            return (1,)
        content_output_list = simout.utterance.content_output_list
        # print("content output list:", content_output_list)
        if len(content_output_list) == 0:
            if simout.poi_exists:
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