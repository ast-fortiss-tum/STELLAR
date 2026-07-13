from typing import List, Dict, Any, Optional
import random
from pydantic import BaseModel
from dataclasses import dataclass, field

@dataclass
class Location:
    longitude: float
    latitude: float

    def to_dict(self):
        return {
            "lng": self.longitude,
            "lat": self.latitude
        }
    
@dataclass
class LOS:
    title: Optional[str] = None
    location: Location = field(default_factory=lambda: Location(0.0, 0.0))
    address: Optional[str] = None
    opening_times: Optional[str] = None
    types: List[str] = field(default_factory=list)
    costs: Optional[str] = None  # Example: "20-30 EUR"
    ratings: Optional[float] = None  # Expected to be in the range 1–5
    foodtypes: Optional[List[str]] = field(default_factory=list)
    payments: List[str] = field(default_factory=list)
    distance: Optional[str] = None

    def to_dict(self):
        """Convert the Utterance object into a JSON-serializable dictionary."""
        return {
                "title": self.title,
                "location": {
                    "latitude": self.location.latitude,
                    "longitude": self.location.longitude,
                } if self.location else None,
                "address": self.address,
                "opening_times": self.opening_times,
                "types": self.types,
                "costs": self.costs,
                "ratings": self.ratings,
                "foodtypes": self.foodtypes if self.foodtypes is not None else [],
                "payments": self.payments,
                "distance": self.distance
            }
    
class Coordinates(BaseModel):
    lat: float
    lng: float

class ContentInput(BaseModel):
    pass


class ContentOutput(BaseModel):
    pass

class Utterance(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    seed: Optional[str] = None
 
    ordinal_vars: List[float] = field(default_factory=list)
    categorical_vars: List[int] = field(default_factory=list)
 
    content_input: Optional[ContentInput] = None
    content_output_list: List[ContentOutput] = field(default_factory=list)
    raw_output: Any = None

    
if __name__ == "__main__":
    utter = Utterance(question="hi, how are you?")
    print(utter.model_dump())
    