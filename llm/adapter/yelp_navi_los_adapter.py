
from typing import Optional, List, Dict

from examples.navi.models import NaviContentOutput
from llm.model.models import Coordinates

def convert_yelp_navi_los_to_content_output(yelp_poi: dict) -> NaviContentOutput:
    categories = None
    if yelp_poi.get("category"):
        categories = [cat.strip() for cat in yelp_poi["category"].split(",")]

    # need to parse the food type from categories
    FOODS_ALL = [
        "german",
        "indian",
        "italian",
        "middle_eastern",
        "french",
        "chinese",
        "japanese",
        "thai",
        "mexican",
        "greek",
        "vietnamese",
        "turkish",
        "american"
    ]
    food_types = []
    for food in FOODS_ALL:
        if categories is not None and food in categories or \
            (yelp_poi.get("name") is not None and food in yelp_poi.get("name")):
            food_types.append(food)

    location = None
    if "latitude" in yelp_poi and "longitude" in yelp_poi:
        location = Coordinates(lat=yelp_poi["latitude"], lng=yelp_poi["longitude"])

    price_range = None
    if yelp_poi.get("price_level") is not None:
        # Assuming price_level maps directly to price_range as string or custom mapping here
        price_range = str(yelp_poi["price_level"])

    return NaviContentOutput(
        title=yelp_poi.get("name"),
        categories=categories,
        food_types=food_types,
        address=yelp_poi.get("address"),
        location=location,
        rating=yelp_poi.get("rating"),
        price_range=price_range
    )
