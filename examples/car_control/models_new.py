from typing import Any, Dict, Optional, List, Literal
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
    use_fillers: Optional[str] = None

INITIAL_STATE_STR = "initial_state_"
TARGET_STATE_STR = "target_state_"  

class CCContentOutput(ContentOutput):

    system: Optional[
        Literal["windows", "fog_lights", "ambient_lights", "head_lights", "reading_lights", "climate", "fan", "seat_heating"]
    ] = None

    position: Optional[Literal["driver", "passenger", "back_left", "back_right"]] = None
    seat_position: Optional[Literal["driver", "front_passenger", "back_left", "back_center", "back_right"]] = None
    fog_light_position: Optional[Literal["front", "rear"]] = None
    window_state_target: Optional[Literal["open", "close"]] = None
    onoff_state_target: Optional[Literal["on", "off"]] = None
    head_lights_mode_target: Optional[Literal["parking", "low_beam", "high_beam", "auto"]] = None
    climate_temperature_value_target: Optional[Literal[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]] = None
    # climate_temperature_value_initial: Optional[Literal[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]] = None
    seat_heating_level_target: Optional[Literal[0, 1, 2, 3]] = None

    system2: Optional[
        Literal["windows", "fog_lights", "ambient_lights", "head_lights", "reading_lights", "climate", "fan", "seat_heating"]
    ] = None

    position2: Optional[Literal["driver", "passenger", "back_left", "back_right"]] = None
    seat_position2: Optional[Literal["driver", "front_passenger", "back_left", "back_center", "back_right"]] = None
    fog_light_position2: Optional[Literal["front", "rear"]] = None
    window_state_target2: Optional[Literal["open", "close"]] = None
    onoff_state_target2: Optional[Literal["on", "off"]] = None
    head_lights_mode_target2: Optional[Literal["off", "parking", "low_beam", "high_beam", "auto"]] = None
    # climate_temperature_value_target2: Optional[Literal[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]] = None
    seat_heating_level_target2: Optional[Literal[0, 1, 2, 3]] = None


class CCContentInput(ContentInput):
    word_perturbation: Optional[str]=None
    slang: Optional[str] = None
    politeness: Optional[str] = None
    implicitness: Optional[str] = None
    anthropomorphism: Optional[str] = None


    
    system: Optional[
        Literal["windows", "fog_lights", "ambient_lights", "head_lights", "reading_lights", "climate", "fan", "seat_heating"]
    ] = None

    position: Optional[Literal["driver", "passenger", "back_left", "back_right"]] = None
    seat_position: Optional[Literal["driver", "front_passenger", "back_left", "back_center", "back_right"]] = None
    fog_light_position: Optional[Literal["front", "rear"]] = None
    window_state_target: Optional[Literal["open", "close"]] = None
    window_state_initial: Optional[Literal["open", "close"]] = None
    onoff_state_target: Optional[Literal["on", "off"]] = None
    onoff_state_initial: Optional[Literal["on", "off"]] = None
    head_lights_mode_target: Optional[Literal["off", "parking", "low_beam", "high_beam", "auto"]] = None
    head_lights_mode_initial: Optional[Literal["off", "parking", "low_beam", "high_beam", "auto"]] = None
    climate_temperature_value_target: Optional[Literal[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]] = None
    climate_temperature_value_initial: Optional[Literal[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]] = None
    seat_heating_level_target: Optional[Literal[0, 1, 2, 3]] = None
    seat_heating_level_initial: Optional[Literal[0, 1, 2, 3]] = None

    system2: Optional[
        Literal["windows", "fog_lights", "ambient_lights", "head_lights", "reading_lights", "climate", "fan", "seat_heating"]
    ] = None

    position2: Optional[Literal["driver", "passenger", "back_left", "back_right"]] = None
    seat_position2: Optional[Literal["driver", "front_passenger", "back_left", "back_center", "back_right"]] = None
    fog_light_position2: Optional[Literal["front", "rear"]] = None
    window_state_target2: Optional[Literal["open", "close"]] = None
    window_state_initial2: Optional[Literal["open", "close"]] = None
    onoff_state_target2: Optional[Literal["on", "off"]] = None
    onoff_state_initial2: Optional[Literal["on", "off"]] = None
    head_lights_mode_target2: Optional[Literal["off", "parking", "low_beam", "high_beam", "auto"]] = None
    head_lights_mode_initial2: Optional[Literal["off", "parking", "low_beam", "high_beam", "auto"]] = None
    climate_temperature_value_target2: Optional[Literal[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]] = None
    climate_temperature_value_initial2: Optional[Literal[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]] = None
    seat_heating_level_target2: Optional[Literal[0, 1, 2, 3]] = None
    seat_heating_level_initial2: Optional[Literal[0, 1, 2, 3]] = None

    #CONTINUOUS FEATURES
    choice: Optional[float] = Field(default=None, ge=0.001, le=1.0)
    add_preferences: Optional[float] = Field(default=None, ge=0.001, le=1.0)
    ask: Optional[float] = Field(default=None, ge=0.001, le=1.0)
    change_of_mind: Optional[float] = Field(default=None, ge=0.001, le=1.0)
    confirmation: Optional[float] = Field(default=None, ge=0.001, le=1.0)
    reject: Optional[float] = Field(default=None, ge=0.001, le=0.3)
    reject_clarify: Optional[float] = Field(default=None, ge=0.001, le=0.5)
    repeat: Optional[float] = Field(default=None, ge=0.001, le=0.5)
    stop: Optional[float] = Field(default=None, ge=0.001, le=0.1)




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