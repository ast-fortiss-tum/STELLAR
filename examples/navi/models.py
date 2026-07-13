from typing import List, Dict, Optional

from pydantic import BaseModel
from llm.model.models import ContentInput, ContentOutput, Coordinates

class NaviContentInput(ContentInput): # TODO handle multiple values, e.g., food_type: [Chinese, Italian]
    title: Optional[str] = None
    category: Optional[str] = None
    address: Optional[str] = None
    location: Optional[Coordinates] = None
    business_hours_status: Optional[str] = None
    payment_method: Optional[str] = None
    rating: Optional[float] = None
    price_range: Optional[str] = None

    fuel_price: Optional[float] = None
    fuel_type: Optional[str] = None
    gas_station_brand: Optional[str] = None
    restaurant_brand: Optional[str] = None
    food_type: Optional[str] = None
    parking: Optional[str] = None
    charging: Optional[str] = None
    availability: Optional[str] = None

class NaviContentOutput(ContentOutput):
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
    parking: Optional[str] = None # TODO think about these features v
    charging: Optional[str] = None
    availability: Optional[str] = None
    
class StyleDescription(BaseModel):
    slang: Optional[str] = None
    politeness: Optional[str] = None
    implicitness: Optional[str] = None
    anthropomorphism: Optional[str] = None
    misspelling_words: Optional[str] = None
    use_fillers: Optional[str] = None
    wrong_declination_of_verbs: Optional[str] = None
    