from abc import ABC, abstractmethod
from typing import List


class SeedSampler(ABC):
    @abstractmethod
    def sample_seeds(self, n_samples: int) -> List[str]:
        pass
