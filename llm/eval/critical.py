from typing import Tuple, List, Literal, Iterable

import numpy as np
from llm.utils.nlp import compute_word_length
from opensbt.evaluation.critical import Critical
from opensbt.evaluation.fitness import Fitness
from llm.model.qa_simout import QASimulationOutput

class CriticalAnswerValidation(Critical):
    def name():
        return "CriticalAnswerValidation"
    def eval(self, vector_fitness: np.ndarray, simout: QASimulationOutput) -> bool:
        return abs(vector_fitness[0]) < 0.8
    
class CriticalAnswerValidationLength(Critical):
    def name():
        return "CriticalAnswerValidationLength"
    def eval(self, vector_fitness: np.ndarray, simout: QASimulationOutput) -> bool:
        return abs(vector_fitness[0]) < 0.8 or abs(vector_fitness[2]) > 30 # score, num_words_answer
    

class CriticalByFitnessThreshold(Critical):
    def __init__(self, 
                 var_name: str = "",
                 score: float = 0.8, 
                 mode: str = "<"):
        super().__init__()

        self.mode = mode
        self.score = score
        self.var_name = var_name

    def name(self) -> str:
        return f"{self.var_name} {self.mode} {self.score}"
    
    def eval(self, vector_fitness: np.ndarray, simout: QASimulationOutput) -> bool:
        value = abs(vector_fitness[0])

        if self.mode == "<":
            return value < self.score
        elif self.mode == "<=":
            return value <= self.score
        elif self.mode == ">":
            return value > self.score
        elif self.mode == ">=":
            return value >= self.score
        elif self.mode == "==":
            return value == self.score
        elif self.mode == "!=":
            return value != self.score
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Choose from: <, <=, >, >=, ==, !=")

class CriticalAnswerLength(Critical):
    def __init__(self, limit: int = 16):
        self.limit = limit
        super().__init__()

    def name():
        return "CriticalAnswerLength"
    
    def eval(self, vector_fitness: np.ndarray, simout: QASimulationOutput):
        return compute_word_length(simout.utterance.answer) > self.limit


class CriticalMerged(Critical):
    def __init__(self,
                 fitness_names: Iterable[str],
                 criticals: List[Tuple[Critical, List[str]]],
                 mode: Literal["and", "or"] = "or"):
        """
        Apply multiple critical functions with shared or distinct fitness inputs.

        :param fitness_names: All dimension names of the fitness vector.
        :param criticals: List of tuples (critical, required_fitness_names).
        :param mode: "or" → True if any critical is triggered, "and" → all must be triggered.
        """
        if mode not in {"or", "and"}:
            raise ValueError(f"Invalid mode: {mode}. Use 'or' or 'and'.")

        self.names_dict = {name: i for i, name in enumerate(fitness_names)}
        self.criticals = criticals
        self.mode = mode

    def name(self) -> str:
        crit_names = []
        for critical, fitness_names in self.criticals:
            feat_str = ", ".join(fitness_names) if fitness_names else "no fitness names"
            crit_names.append(f"{feat_str} {critical.name()}")
        return f"CriticalMerged[{self.mode.upper()}]({'; '.join(crit_names)})"
    
    def eval(self, vector_fitness: np.ndarray, simout: QASimulationOutput) -> bool:
        results = []
        for critical, fitness_names in self.criticals:
            indices = [self.names_dict[name] for name in fitness_names]
            subvector = vector_fitness[indices]
            results.append(critical.eval(subvector, simout))

        return any(results) if self.mode == "or" else all(results)
