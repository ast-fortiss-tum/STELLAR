from collections import defaultdict
from typing import Tuple, List, Dict, Optional

import numpy as np

from llm.adapter.embeddings_openai_adapter import get_similarity_individual
from examples.navi.models import NaviContentInput, NaviContentOutput
from llm.validation.validator_los import llm_validator_los, llm_validator_response_los
from opensbt.evaluation.fitness import Fitness
from llm.model.qa_simout import QASimulationOutput
from llm.validation.validator_category import llm_validator_category
from llm.config import N_VALIDATORS
from llm.utils.nlp import compute_word_length
from llm.utils.embeddings_local import get_similarity
from llm.llms import LLMType
from llm.adapter.embeddings_local_adapter import get_disimilarity_individual

counter_validations = 0
    
    
class FitnessAnswerValidationCategory(Fitness):

    @property
    def min_or_max(self):
        return "min",

    @property
    def name(self):
        return "dimension_1",

    def __init__(self, expected_category: str) -> None:
        super().__init__()
        self.expected_category = expected_category

    def eval(self, 
             simout: QASimulationOutput, 
             **kwargs) -> Tuple[float]:
        
        global counter_validations

        scores = llm_validator_category(answer=simout.utterance.answer,
                              n = N_VALIDATORS)
        score = scores[self.expected_category]
        # print("[FitnessAnswerValidation] score", score)
        counter_validations += 1
        print("counter_validations", counter_validations)
        return (score,)
    
class FitnessDiverse(Fitness):
    def __init__(self, diversify=False) -> None:
        super().__init__()

    @property
    def min_or_max(self):
        return "max",
    
    @property
    def name(self):
        return "distance",

    def eval(self, simout: QASimulationOutput, **kwargs) -> Tuple[float]:  
        distance_archive = 0
        if "algorithm" in kwargs:
            algorithm = kwargs["algorithm"]
            
            if not hasattr(algorithm, 'archive_novelty'):
                distance_archive = 0
                print("no archive novelty")
            else:
                _, distance_archive = algorithm.archive_novelty.closest_individual_from_vars(
                                                kwargs["individual"], 
                                                dist_fnc = get_disimilarity_individual)
                print("Distance archive:", distance_archive)
        f_vector = (distance_archive,)
        return f_vector
    
class FitnessNumberOfWords(Fitness):
    @property
    def min_or_max(self):
        return "max",
    
    @property
    def name(self):
        return "number_of_words",
    
    def eval(self, simout: QASimulationOutput, **kwargs):
        return compute_word_length(simout.utterance.answer),

    
class FitnessMerged(Fitness):
    def __init__(self, fitnesses: List[Fitness]) -> None:
        super().__init__()
        self.fitnesses = fitnesses

    @property
    def min_or_max(self):
        res = ()
        for fitness in self.fitnesses:
            res += fitness.min_or_max
        return res
    
    @property
    def name(self):
        res = ()
        for fitness in self.fitnesses:
            res += fitness.name
        return res

    def eval(self, simout: QASimulationOutput, **kwargs) -> Tuple[float]:  
        res = ()
        for fitness in self.fitnesses:
            res += fitness.eval(simout, **kwargs)
        return res
    
  
class FitnessAnswerValidationLOS(Fitness):

    @property
    def min_or_max(self):
        return "min", "min"

    @property
    def name(self):
        return "los_similarity", "response_los_similarity"

    def eval(self, 
             simout: QASimulationOutput, 
             **kwargs) -> Tuple[float]:
        
        global counter_validations

        score_los = llm_validator_los(question=simout.utterance.question,
                              answer=simout.utterance.answer,
                              los_in=simout.utterance.question_los,
                              los_out=simout.utterance.los,
                              n = N_VALIDATORS)
        
        score_response_los = llm_validator_response_los(los_out=simout.utterance.los,
                                                        question=simout.utterance.question)
        
        print("score los vs. response:", score_response_los)
        # print("[FitnessAnswerValidation] score", score)
        counter_validations += 1
        print("counter_validations", counter_validations)
        return (score_los, score_response_los)
    
class FitnessAnswerValidationDiverseLOS(Fitness):
    def __init__(self, diversify=False) -> None:
        super().__init__()

    @property
    def min_or_max(self):
        return "min", "min", "max"
    
    @property
    def name(self):
        return "Score_LOS", "Score_LOS_Response", "Distance_Archive"

    def eval(self, simout: QASimulationOutput, **kwargs) -> Tuple[float]:  
        distance_archive = 0
        F = FitnessAnswerValidationLOS()
        scores = F.eval(simout=simout, kwargs=kwargs)

        if "algorithm" in kwargs:
            algorithm = kwargs["algorithm"]
            
            if not hasattr(algorithm, 'archive_novelty'):
                distance_archive = 0
                print("no archive novelty")
            else:
                _, distance_archive = algorithm.archive_novelty.closest_individual_from_vars(
                                                kwargs["individual"], 
                                                dist_fnc = get_similarity_individual)
        f_vector = (scores[0], scores[1], distance_archive)
        return f_vector