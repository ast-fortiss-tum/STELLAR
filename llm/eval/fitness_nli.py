from typing import Tuple
from opensbt.evaluation.fitness import Fitness
from llm.model.qa_simout import QASimulationOutput
from llm.validation.validator_nli import llm_validator
import logging as log
from llm.adapter.embeddings_openai_adapter import get_similarity_individual

counter_validations = 0

class FitnessAnswerValidationNLI(Fitness):

    @property
    def min_or_max(self):
        return "min",

    @property
    def name(self):
        return "score",

    def eval(self, 
             simout: QASimulationOutput, 
             **kwargs) -> Tuple[float]:
        
        global counter_validations

        score = llm_validator(question=simout.utterance.question,
                              answer=simout.utterance.answer)
        
        # print("[FitnessAnswerValidation] score", score)
        counter_validations += 1
        log.info(f"counter_validations: {counter_validations}")
        return (score,)

class FitnessAnswerValidationDiverseNLI(Fitness):

    @property
    def min_or_max(self):
        return "min","max"
    
    @property
    def name(self):
        return "Score","Distance Archive"
    
    def eval(self, simout: QASimulationOutput, **kwargs) -> Tuple[float]:
        global counter_validations
 
        distance_archive = 0
        F = FitnessAnswerValidationNLI()
        score = F.eval(simout=simout, kwargs=kwargs)[0]

        if "algorithm" in kwargs:
            algorithm = kwargs["algorithm"]
            
            if not hasattr(algorithm, 'archive_novelty'):
                distance_archive = 0
                print("no archive novelty")
            else:
                _, distance_archive = algorithm.archive_novelty.closest_individual_from_vars(
                                                kwargs["individual"], 
                                                dist_fnc = get_similarity_individual)
                # print("[FitnessAnswerValidation] score", score)
        counter_validations += 1
        log.info(f"counter_validations: {counter_validations}")
        f_vector = (score, distance_archive)
        
        return f_vector
