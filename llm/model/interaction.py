from typing import Any, Dict, Optional, List
from enum import Enum
from pydantic import BaseModel, Field
from .models import Coordinates


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


class POI(BaseModel):
    title: Optional[str] = None
    categories: Optional[List[str]] = None
    address: Optional[str] = None
    location: Optional[Coordinates] = None
    business_hours_status: Optional[str] = None
    payment_methods: Optional[List[str]] = None
    rating: Optional[float] = None
    price_range: Optional[str] = None

    fuel_prices: Optional[Dict[str, float]] = None
    fuel_types: Optional[List[str]] = None
    gas_station_brand: Optional[str] = None
    restaurant_brand: Optional[str] = None
    food_types: Optional[List[str]] = None
    parking: Optional[str] = None
    charging: Optional[str] = None
    availability: Optional[str] = None
