from collections import defaultdict
from typing import Tuple, List, Dict, Optional

import numpy as np

from examples.car_control.fitness import CCFitnessContentComparison
from llm.config import N_VALIDATORS
from judge_eval.validator_dim_carcontrol import llm_validator_conversation
from opensbt.evaluation.fitness import Fitness
from llm.model.mt_simout import MultiTurnSimulationOutput
from llm.eval.fitness import counter_validations


class CCFitnessConversationValidationDimensions(Fitness):
    def __init__(self, 
                llm_type=None,
                weights=[0.5, 0.5],
                dimension_labels = ["C", "R"],  # clarity & request-orientedness by default - see llm\judge_eval_mt\prompts.py
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
        return ("dimensions_fitness",)

    def eval(self, simout: MultiTurnSimulationOutput, **kwargs) -> Tuple[float]:
        global counter_validations

        dim_scores, answers, justifications = llm_validator_conversation(
            conversation=simout.conversation,
            n=N_VALIDATORS,
            llm_type=self.llm_type,
        )

        max_total = np.sum(np.array(self.weights) * self.max_score)
        weighted_score = sum(s * w for s, w in zip(dim_scores, self.weights))
        final_score = weighted_score / max_total

        # store for debugging 
        simout.other["fitness_conversation_scores"] = {}
        simout.other["fitness_conversation_scores"]["weights"] = self.weights 
        simout.other["fitness_conversation_scores"]["scores"] = dict(zip(self.dimension_labels, dim_scores))
        simout.other["fitness_conversation_scores"]["all_scores"] = answers
        simout.other["fitness_conversation_scores"]["justifications"] = justifications

        counter_validations += 1
        print("counter_validations", counter_validations)

        return (final_score,)

# TODO: move inefficient system intents to params
class CCFitnessConversationEfficiency(Fitness):
    def __init__(self):
        super().__init__()
        # TODO: does it make sense to add other intents with follow-up? maybe weight them differently?
        self.inefficient_system_intents = {"reject", "reject_and_followup", "failure"}

    @property
    def min_or_max(self):
        return ("min",)

    @property
    def name(self):
        return ("efficiency_fitness",)

    def eval(self, simout: MultiTurnSimulationOutput, **kwargs) -> Tuple[float]:
        conversation = simout.conversation
        total_turns = len(conversation)
        
        inefficient_turns = 0
        for turn in conversation.turns:
            # check if likely system failure or rejection
            if turn.answer_intent_classified in self.inefficient_system_intents:
                inefficient_turns += 1

        score = 1.0 - (inefficient_turns / total_turns) if total_turns > 0 else 1.0

        if not hasattr(simout, 'other') or simout.other is None:
            simout.other = {}
        simout.other["efficiency"] = {
            "total_turns": total_turns,
            "inefficient_turns": inefficient_turns,
            "score": score
        }

        return (score,)

# TODO: move evaluate intents to params
class CCFitnessConversationEffectiveness(Fitness):
    def __init__(self, field_weights: Optional[dict] = None):
        super().__init__()
        self.content_comparator = CCFitnessContentComparison(field_weights=field_weights)

    @property
    def min_or_max(self):
        return ("min",)

    @property
    def name(self):
        return ("effectiveness_fitness",)
    
    def eval(self, simout: MultiTurnSimulationOutput, **kwargs) -> Tuple[float]:
        conversation = simout.conversation
        per_turn_scores = []
        per_turn_details = []
        relevant_scores = []  # relates to target_intents

        print("Running eval")

        for i, turn in enumerate(conversation.turns):
            content_input = turn.content_input
            #print("#####################")
            #print(content_input)
            #print("#####################")
            content_output_list = turn.content_output_list
            poi_exists = turn.poi_exists

            # prefer stored flag; fallback to constant-based computation
            user_intent_influences_fit = turn.user_intent_influences_fit

            if content_input is None:
                turn_score = 1.0
                turn_detail = {"turn": i, "score": turn_score, "reason": "no_content_input"}

            elif not user_intent_influences_fit:
                turn_score = 1.0
                turn_detail = {"turn": i, "score": turn_score, "reason": "user_intent_not_influencing_fitness"}

            elif not content_output_list:
                if poi_exists:
                    turn_score = 0.0
                    turn_detail = {"turn": i, "score": turn_score, "reason": "poi_exists_but_no_output"}
                else:
                    turn_score = 1.0
                    turn_detail = {"turn": i, "score": turn_score, "reason": "no_poi_no_output"}
            else:
                scores = []
                field_scores_all = []
                for content_output in content_output_list:
                    score, field_scores = self.content_comparator._evaluate_content(
                        content_input, content_output, poi_exists=poi_exists
                    )
                    #print("")
                    #print("###############################################")
                    #print(f"Content input: {content_input}")
                    #print(f"Content output: {content_output}")
                    #print(f"Score: {score}")
                    #print(f"Field_scores: {field_scores}")
                    #print("###############################################")
                    #print("")
                    scores.append(score)
                    field_scores_all.append(field_scores)

                if scores:
                    best_idx = int(np.argmax(scores))
                    turn_score = scores[best_idx]
                    turn_detail = {
                        "turn": i,
                        "score": turn_score,
                        "field_scores": field_scores_all[best_idx],
                        "reason": "evaluated"
                    }
                else:
                    turn_score = 0.0
                    turn_detail = {"turn": i, "score": turn_score, "reason": "no_scores"}

            # keep details for all turns
            per_turn_details.append(turn_detail)

            # include in score aggregation only if intent influences fitness
            if user_intent_influences_fit:
                per_turn_scores.append(turn_score)
                relevant_scores.append(turn_score)

        avg_effectiveness = sum(relevant_scores) / len(relevant_scores) if relevant_scores else 1.0

        if not hasattr(simout, 'other') or simout.other is None:
            simout.other = {}

        simout.other["effectiveness"] = {
            "per_turn_scores": per_turn_scores,
            "relevant_scores": relevant_scores, 
            "per_turn_details": per_turn_details,
            "average_score": avg_effectiveness
        }
        
        return (avg_effectiveness,)