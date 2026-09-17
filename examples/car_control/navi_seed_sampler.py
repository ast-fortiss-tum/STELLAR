from abc import ABC, abstractmethod
import random
from typing import List
import json

with open("llm/examples/navi/data/user_data.json", "r") as f:
    seed_corpus = json.load(f)

class NaviSeedSampler(ABC):
    def sample_seeds(self, n_samples: int) -> List[str]:
        return random.sample(seed_corpus, n_samples)
