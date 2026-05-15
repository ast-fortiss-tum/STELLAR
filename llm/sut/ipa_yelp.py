import requests
import json
import time
import sys
import re
from llm.adapter.yelp_navi_los_adapter import convert_yelp_navi_los_to_content_output
from llm.llms import LLMType
from opensbt.model_ga.individual import Individual
from typing import List
from opensbt.simulation.simulator import Simulator
from llm.model.qa_simout import QASimulationOutput
from math import sin, cos, pi, ceil
import numpy as np
from random import random
from opensbt.utils import geometric
import json
from datetime import datetime
import csv
import os
from examples.navi.models import NaviContentOutput
from llm.model.models import Utterance
from llm.llms import pass_llm
from llm.config import LLM_IPA
import logging as log
from llm.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_CONTENT_INPUT
import copy
import traceback
from json_repair import repair_json
from dataclasses import fields

def send_request_poi_exists(navi_input, user_location=(39.955431, -75.154903)):
    """
    Sends a request to the /poi_exists API using a NaviContentInput object.

    Args:
        navi_input: NaviContentInput object containing the query constraints.
        user_location: Tuple of (latitude, longitude)

    Returns:
        dict: {
            "exists": bool,
            "matching_pois": list of dicts
        }
    """
    def map_price_range(price_range):
        """
        Maps input price range into standardized symbolic levels:
        $  -> low
        $$ -> medium
        $$$ -> high
        """
        if price_range in [1, "low", "$"]:
            return "$"      # low
        elif price_range in [2, "medium", "$$"]:
            return "$$"     # medium
        elif price_range in [3, "high", "$$$"]:
            return "$$$"    # high
        else:
            return None     # unknown / unspecified

    # Map NaviContentInput fields to the /poi_exists constraint keys
    constraints = {
        "category": navi_input.category,
        "name": navi_input.title,
        "price_level": map_price_range(navi_input.price_range),
        "rating": navi_input.rating,
        "radius_km": None,  # Could be added dynamically if needed
        "open_now": None,   # Could map from business_hours_status
        "cuisine": navi_input.food_type  # Optional: could map from food_type or restaurant_brand
    }

    # Remove None values
    constraints = {k: v for k, v in constraints.items() if v is not None}

    url = "http://127.0.0.1:8000/poi_exists"
    payload = {**constraints, "user_location": list(user_location)}
    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    try:
        print("Sending payload:", payload)
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        duration = time.time() - start_time
        print(f"Request completed in {duration:.2f} seconds")

        response_data = response.json()
        # print(json.dumps(response_data, indent=2))

        return response_data

    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time
        print(f"Request failed after {duration:.2f} seconds")
        raise e

def send_request_conv_navi(query):
    """
        The response is a dict.
        Example:

        response_dict = {
            "response": "The closest match is Triangle Medical, rated 4.5/5, at 253 S 10th St. Should I navigate you there?",
            "retrieved_pois": [
                {
                "name": "Triangle Medical",
                "category": "Doctors, Health & Medical, Neurologist, Osteopathic Physicians",      
                "rating": 4.5,
                "price_level": null,
                "address": "253 S 10th St",
                "latitude": 39.9468194,
                "longitude": -75.1575343
                }
            ]
        }
    
    """
    url = "http://127.0.0.1:8000/query"

    payload = {
        "query": query,
        "user_location": [39.955431, -75.154903],
        "llm_type" : os.getenv("DEPLOYMENT_NAME")
    }

    headers = {
        "Content-Type": "application/json"
    }

    start_time = time.time()

    try:
        print("payload sent:", payload)
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        duration = time.time() - start_time
        print(json.dumps(response.json(), indent=2))
        print(f"Request completed in {duration:.2f} seconds")
        return response.json()
    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time
        print(e)
        print(f"Request failed after {duration:.2f} seconds")
        raise e

class IPA_YELP(Simulator):
    memory: List = []
    ipa_name = "yelp_based_conv_navi"

    @staticmethod
    def simulate(list_individuals: List[Utterance], 
                 variable_names: List[str], 
                 scenario_path: str, 
                 sim_time: float, 
                 time_step: float = 10,
                 do_visualize: bool = False,
                 temperature: float = 0,
                 context: object = None) -> List[QASimulationOutput]:

        results = []
        
        # log.info("[IPA] context", context)
        log.info(f"[IPA_YELP] list_individuals: {list_individuals}")
        
        for utterance in list_individuals:  
            utterance = utterance[0]

            def check_utterance_in_mem(utterance):
                for u in IPA_YELP.memory:
                    if u.question == utterance.question:
                        return True, u
                return False, utterance
            
            in_memory, memory_utterance = check_utterance_in_mem(utterance)

            response = ""
            poi_exists = False

            if not in_memory:
                max_attempts = 5
                attempt = 0
                while attempt < max_attempts:
                    try:
                        # TODO we need be able to send the llm type and other infos as well to 
                        # the yelp based SUT
                        # Example
                        res = send_request_poi_exists(navi_input=utterance.content_input)

                        print("[SUT YELP] POI exists: ", res["exists"])
                        poi_exists = res["exists"]
                        
                        response = send_request_conv_navi(query=utterance.question)
                                                
                        if response is not None:
                            utterance.answer = response["response"]
                            utterance.content_output_list = [
                                convert_yelp_navi_los_to_content_output(poi) for \
                                    poi in response["retrieved_pois"]
                            ]
                            
                            # print("IPA YELP Utterance Answer: ", utterance)
                    except Exception as e:
                        traceback.print_exc()
                        print(f"[IPA_YELP] Attempt {attempt + 1} failed with error: {e}")
                        time.sleep(2)
                    attempt += 1
                IPA_YELP.memory.append(utterance)
            else:
                print("Utterance already in memory")
                utterance.answer = memory_utterance.answer
                utterance.content_output_list = memory_utterance.content_output_list
            print(f"[IPA LOS] {utterance}")
                     
            result = QASimulationOutput(
                utterance=utterance,
                model=LLMType(LLM_IPA),
                ipa=IPA_YELP.ipa_name,
                response=response,
                poi_exists=poi_exists
            )
            results.append(result)

        return results


if __name__ == "__main__":
    qus = ["Time for burgers! Need directions, 5 stars!",
           "I need to open my windows on the left.", 
           "It is very cold.",
           "I am very hungry and need a Pizza.",
           "I am very hungry and need a French baguette.",
           "Show options for breakfasts.",
           "Where is the closest public toilet.",
           "Where is the closest Cafe.",
           "I am running out of gas.",
           "I need to charge my car.",
           "I am looking for a church."]
    utterances = []
    for u in qus:
        utterances.append([Utterance(question=u)])

    context = {
                        "location" : {
                            "position" : "Amathountos Avenue 502, 4520, Limassol, Cyprus",
                            "data" : "2025-03-19T0",
                            "time" : "09:00:00",
                        }
            }
    res = IPA_YELP.simulate(utterances,
                    ["utterance"], 
                    "", 
                    sim_time=-1,
                    time_step=-1, 
                    do_visualize=False,
                    temperature = 0,
                    context = context)