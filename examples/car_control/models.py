from typing import Any, Dict, Optional, List
from enum import Enum
from pydantic import BaseModel, Field
from llm.model.models import ContentInput, ContentOutput, Coordinates


class WindowState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class LightState(str, Enum):
    OFF = "off"
    ON = "on"


class ClimateState(str, Enum):
    OFF = "off"
    ON = "on"


class SeatHeatingLevel(str, Enum):
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StyleDescription(BaseModel):
    slang: Optional[str] = None
    politeness: Optional[str] = None
    implicitness: Optional[str] = None
    anthropomorphism: Optional[str] = None
    misspelling_words: Optional[str] = None
    use_fillers: Optional[str] = None
    wrong_declination_of_verbs: Optional[str] = None


class CarState(BaseModel):
    window_front_left: Optional[WindowState] = None
    window_front_right: Optional[WindowState] = None
    window_rear_left: Optional[WindowState] = None
    window_rear_right: Optional[WindowState] = None

    fog_light: Optional[LightState] = None
    head_light: Optional[LightState] = None
    ambient_light: Optional[LightState] = None
    reading_light_front_left: Optional[LightState] = None
    reading_light_front_right: Optional[LightState] = None
    reading_light_rear_left: Optional[LightState] = None
    reading_light_rear_right: Optional[LightState] = None

    temperature: Optional[float] = None
    climate: Optional[ClimateState] = None
    fan: Optional[ClimateState] = None

    seat_heating_front_left: Optional[SeatHeatingLevel] = None
    seat_heating_front_right: Optional[SeatHeatingLevel] = None
    seat_heating_rear_left: Optional[SeatHeatingLevel] = None
    seat_heating_rear_right: Optional[SeatHeatingLevel] = None


INITIAL_STATE_STR = "initial_state_"
TARGET_STATE_STR = "target_state_"  

class CCContentOutput(ContentOutput, CarState):
    target_state: Optional[CarState] = None

class CCContentInput(ContentInput):
    initial_state: Optional[CarState] = None
    target_state: Optional[CarState] = None

    @classmethod
    def from_features_dict(cls, features_dict: Dict[str, Any]) -> "CCContentInput":
        initial_state = CarState.model_validate({k[len(INITIAL_STATE_STR):]: v for k, v in features_dict.items() if k.startswith(INITIAL_STATE_STR)})
        target_state = CarState.model_validate({k[len(TARGET_STATE_STR):]: v for k, v in features_dict.items() if k.startswith(TARGET_STATE_STR)})
        return cls(initial_state=initial_state, target_state=target_state)
    
    def as_dict(self) -> Dict[str, Any]:
        result = {}
        if self.initial_state:
            result.update({f"{INITIAL_STATE_STR}{k}": v for k, v in self.initial_state.model_dump(exclude_none=True).items()})
        if self.target_state:
            result.update({f"{TARGET_STATE_STR}{k}": v for k, v in self.target_state.model_dump(exclude_none=True).items()})
        return result

class CCUserIntent(BaseModel):
    choice: Optional[float] = Field(default=None, ge=0.001, le=1.0)
    add_preferences: Optional[float] = Field(default=None, ge=0.001, le=1.0)
    ask: Optional[float] = Field(default=None, ge=0.001, le=1.0)
    change_of_mind: Optional[float] = Field(default=None, ge=0.001, le=1.0)
    confirmation: Optional[float] = Field(default=None, ge=0.001, le=1.0)
    reject: Optional[float] = Field(default=None, ge=0.001, le=1.0)
    reject_clarify: Optional[float] = Field(default=None, ge=0.001, le=1.0)
    stop: Optional[float] = Field(None, ge=0.001, le=1.0)