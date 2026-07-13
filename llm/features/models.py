import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class FeatureType(Enum):
    ORDINAL = 0
    CATEGORICAL = 1
    CONTINUOUS = 2


class Feature(BaseModel):
    name: str = ""
    feature_type: Optional[FeatureType] = Field(..., description="Type of the feature")


class DiscreteFeature(Feature):
    values: List[Any] = []
    distribution: Optional[List[float]] = None

    @property
    def num_values(self) -> int:
        return len(self.values)
    

class ContinuousFeature(Feature):
    lb: float = 0.0
    ub: float = 1.0
    

class CombinedFeaturesInstance(BaseModel):
    ordinal: Optional[List[float]] = None
    categorical: Optional[List[int]] = None
    continuouos: Optional[List[float]] = None
