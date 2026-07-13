from typing import List, Tuple, Optional
import random
import multiprocessing
import signal

import numpy as np
from testflows.combinatorics import Covering

from llm.llms import LLMType
from llm.model.models import Utterance
from llm.features.models import DiscreteFeature
from llm.model.qa_problem import QAProblem
from llm.config import LLM_SAMPLING
from llm.features.models import DiscreteFeature
from .base_classes import UtteranceSamplingBase


class UtteranceSamplingDiscrete(UtteranceSamplingBase):
    def __init__(
        self,
        variable_length=False,
        llm_type=LLMType(LLM_SAMPLING),
        generate_question: bool = True,
    ):
        super().__init__()
        self.variable_length = variable_length
        self.llm_type = llm_type
        self.generate_question = generate_question

    def _sample_instances(
        self, problem: QAProblem, n_samples: int, **kwargs
    ) -> List[Utterance]:
        result = []
        if problem.seed_sampler is not None:
            seeds = problem.seed_sampler.sample_seeds(n_samples)
            print("seeds:", seeds)
        else:
            seeds = [None for _ in range(n_samples)]

        for i in range(n_samples):
            vars = problem.feature_handler.sample_feature_scores()
            result.append(self._build_utterance(
                problem=problem,
                seed=seeds[i],
                ordinal_vars=vars.ordinal,
                categorical_vars=vars.categorical,
            ))
        return result


def calculate_covering_size(params, t, result_dict):
    covering = Covering(params, strength=t)
    result_dict[t] = len(list(o for o in covering))


class UtteranceSamplingGrid(UtteranceSamplingBase):
    def __init__(
        self,
        llm_type=LLMType(LLM_SAMPLING),
        covering_search_time: int = 30,
        total_samples: Optional[int] = None,
        t: int = None
    ):
        super().__init__()
        self.llm_type = llm_type
        self.covering_search_time = covering_search_time
        self.total_samples = total_samples
        self.covering_variables = []
        self.t = t

    @staticmethod
    def _timeout(*args, **kwargs):
        raise TimeoutError

    def _get_covering(self, n_samples, features: List[DiscreteFeature]) -> List[Tuple[int]]:
        # Create parameter dictionary for features
        params = {f.name: list(range(f.num_values)) for f in features}
        
        # Binary search for the minimal strength t that covers enough samples
        max_t, min_t = len(features) + 1, 0

        m = multiprocessing.Manager()
        covering_size_dict = m.dict()

        if self.t is None:
            while max_t - min_t > 1:
                t = (max_t + min_t) // 2
                p = multiprocessing.Process(target=calculate_covering_size, args=[params, t, covering_size_dict])
                p.start()
                p.join(self.covering_search_time)
                if p.is_alive():
                    max_t = t
                    p.kill()
                    p.join()
                else:
                    if covering_size_dict[t] > n_samples:
                        max_t = t
                    else:
                        min_t = t
            t = max_t
        else:
            t = self.t
            
        covering_raw = Covering(params, strength=t)

        # Convert to list of tuples
        covering = [tuple(c[f.name] for f in features) for c in covering_raw]

        # Calculate max samples per combination
        max_samples_per_combination = n_samples // len(covering) + 1

        # Initialize combination counts
        combination_to_count = {c: max_samples_per_combination for c in covering}

        # Adjust counts by randomly removing excess samples
        total_samples = len(covering) * max_samples_per_combination
        excess_samples = total_samples - n_samples
        for c in random.sample(covering, excess_samples):
            combination_to_count[c] -= 1

        # Expand combinations into the result list
        result = [c for c, count in combination_to_count.items() for _ in range(count)]
        print(f"For {n_samples} generated a covering of size {len(result)} with t={t}")
        random.shuffle(result)
        return result
    
    def sample_covering(self, problem: QAProblem, n_samples: int, **kwargs):
        # Gather categorical and ordinal features
        categorical_features = list(problem.feature_handler.categorical_features.values())
        ordinal_features = list(problem.feature_handler.ordinal_features.values())
        features = categorical_features + ordinal_features

        # Get covering variables
        self.covering_variables = self._get_covering(n_samples, features)

    def _sample_instances(self, problem: QAProblem, n_samples: int, **kwargs) -> List[str]:
        # Gather categorical and ordinal features
        categorical_features = list(problem.feature_handler.categorical_features.values())
        ordinal_features = list(problem.feature_handler.ordinal_features.values())

        result = []
        for i in range(n_samples):
            if len(self.covering_variables) == 0:
                to_sample = self.total_samples or n_samples - i
                self.sample_covering(problem, to_sample)
            covering_variable = self.covering_variables.pop()

            # Split into categorical and ordinal variables
            categorical_vars = covering_variable[:len(categorical_features)]
            ordinal_vars = covering_variable[len(categorical_features):]

            # Normalize ordinal variables
            ordinal_vars = [(value + 0.5) / feature.num_values for value, feature in zip(ordinal_vars, ordinal_features)]

            # Generate utterance
            utterance = problem.question_generator.generate_utterance(
                seed=None,
                ordinal_vars=ordinal_vars,
                categorical_vars=categorical_vars,
                llm_type=self.llm_type
            )
            result.append(utterance)

        return result
