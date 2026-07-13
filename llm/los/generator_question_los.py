from llm.feature_discretization import score_to_label
from llm.model.models import LOS, Location

def generate(problem, vars):
    feature_names = problem.names_dim_utterance

    feature_to_attr = {
        "venue": "types",
        "address": "address",
        "opening_times": "opening_times",
        "costs": "costs",
        "ratings": "ratings",
        "foodtypes": "foodtypes",
        "payments": "payments",
        "max_distance": "distance",
        "types": "types",
    }

    los_data = {
        "title": "No Title",
        "address": None,
        "opening_times": "",
        "types": [],
        "costs": "N/A",
        "ratings": 0,
        "foodtypes": [],
        "payments": [],
        "distance": None,
        "location": None,  # dummy location
    }

    for feature, score in zip(feature_names, vars):
        value = score_to_label(dimension=feature, score=score)
        attr_name = feature_to_attr.get(feature)
        if attr_name is None:
            continue
        if attr_name in ["foodtypes", "payments", "types"]:
            los_data[attr_name] = value if isinstance(value, list) else [value]
        else:
            los_data[attr_name] = value

    return LOS(
        title=los_data["title"],
        address=los_data["address"],
        opening_times=los_data["opening_times"],
        types=los_data["types"],
        costs=los_data["costs"],
        ratings=los_data["ratings"],
        foodtypes=los_data["foodtypes"],
        payments=los_data["payments"],
        distance=los_data["distance"]
    )