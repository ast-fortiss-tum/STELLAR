import requests
import json
import time
import traceback
import random
from typing import List, Optional

from examples.car_control.cc_utterance_generator_new import CCUtteranceGenerator
from llm.adapter.yelp_navi_los_adapter_cc import convert_yelp_navi_los_to_content_output
from llm.features.feature_handler import FeatureHandler
from llm.llms import LLMType
from llm.model.conversation_intents import UserIntent

USER_INTENTS_INFLUENCING_FITNESS = {
    UserIntent.START.value,
    UserIntent.ADD_PREFERENCES.value,
    UserIntent.CHANGE_OF_MIND.value,
    UserIntent.REJECT_CLARIFY.value,
    UserIntent.REPEAT.value,
}

from llm.model.mt_simout import MultiTurnSimulationOutput
from llm.sut.ipa_base_cc import IPABaseCC as IPABase
from llm.model.qa_simout import QASimulationOutput
from llm.model.models import Conversation, Turn, Utterance
import llm.config as _llm_config
import logging as log
from examples.car_control.models_new import CCContentInput, CCContentOutput
import os


def send_request_poi_exists(navi_input, url=None, user_location=(39.955431, -75.154903)):
    if url is None:
        url = os.environ.get("CONV_NAVI_ADDRESS", "http://127.0.0.1:8000")
    """
    Sends a request to the /poi_exists API using a NaviContentInput object.
    """
    def map_price_range(price_range):
        if price_range in [1, "low", "$"]:
            return "$"
        elif price_range in [2, "medium", "$$"]:
            return "$$"
        elif price_range in [3, "high", "$$$"]:
            return "$$$"
        else:
            return None
    #constraints = {
    #    "category": navi_input.system,
    #    "name": navi_input.title,
    #    "price_level": map_price_range(navi_input.price_range),
    #    "rating": navi_input.rating,
    #    "radius_km": None,
    #    "open_now": None,
    #    "cuisine": navi_input.food_type,
    #    "parking": navi_input.parking,
    #}
    constraints = {
        "category": None,
        "name": None,
        "price_level": None,
        "rating": None,
        "radius_km": None,
        "open_now": None,
        "cuisine": None,
        "parking": None,
    }
    constraints = {k: v for k, v in constraints.items() if v is not None}

    poi_url = f"{url}/poi_exists"
    
    if isinstance(user_location, (list, tuple)):
        loc_val = list(user_location)
    else:
        loc_val = user_location

    payload = {**constraints, "user_location": loc_val}
    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    try:
        print("Sending payload:", payload)
        response = requests.post(poi_url, json=payload, headers=headers)
        response.raise_for_status()
        duration = time.time() - start_time
        print(f"Request completed in {duration:.2f} seconds")
        return response.json()
    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time
        print(f"Request failed after {duration:.2f} seconds")
        raise e


def send_request_conv_navi(query, user_location=None, user_id=1, max_retries: int = 3,
                    llm_type = os.environ.get('LLM_IPA')):
    """
    Send a conversation-navi query with retry support.
    """
    url = os.environ.get("CONV_NAVI_ADDRESS", "http://127.0.0.1:8000") + "/query"
    
    # use provided location or fallback to default (Philadelphia)
    if user_location is None:
        loc_val = [39.955431, -75.154903]
    elif isinstance(user_location, (list, tuple)):
        loc_val = list(user_location)
    else:
        loc_val = user_location

    payload = {
        "query": query,
        "user_location": loc_val,
        "llm_type": llm_type,
        "user_id": user_id,
    }
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
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
            print(f"[send_request_conv_navi] Attempt {attempt + 1} failed after {duration:.2f}s: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return {"response": "Request failed after max retries.", "retrieved_pois": []}


class IPA_YELP(IPABase):
    memory: List = []
    ipa_name = "yelp_based_conv_navi"
    global_user_counter = 0

    @staticmethod
    def simulate(
        list_individuals: List[Utterance],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float = 10,
        do_visualize: bool = False,
        temperature: float = 0,
        context: object = None,
        max_retries: int = 5,
        **kwargs,
    ) -> List[QASimulationOutput]:

        results = []
        log.info(f"[IPA_YELP] list_individuals: {list_individuals}")

        # extract location from context
        user_location = None
        if isinstance(context, dict):
            user_location = context.get("location", {}).get("position")

        for utterance in list_individuals:
            utterance = utterance[0]

            def check_utterance_in_mem(utt):
                for u in IPA_YELP.memory:
                    if u.question == utt.question:
                        return True, u
                return False, utt

            in_memory, memory_conversation = check_utterance_in_mem(utterance)

            response = ""
            poi_exists = True

            if not in_memory:
                for attempt in range(max_retries):
                    try:
                        print("content input:", utterance.content_input)
                        # pass user_location to POI check
                        res = send_request_poi_exists(
                            navi_input=utterance.content_input,
                            user_location=user_location
                        )
                        print("[SUT YELP] POI exists: ", res["exists"])
                        poi_exists = res["exists"]

                        # pass user_location to ConvNavi query
                        response = send_request_conv_navi(
                            query=utterance.question,
                            user_location=user_location,
                            max_retries=max_retries
                        )

                        if response is not None:
                            utterance.answer = response["response"]
                            utterance.content_output_list = [
                                convert_yelp_navi_los_to_content_output(response["retrieved_pois"], utterance.content_input)
                            ]
                    except Exception as e:
                        traceback.print_exc()
                        print(f"[IPA_YELP] Attempt {attempt + 1} failed with error: {e}")
                        time.sleep(2)
                IPA_YELP.memory.append(utterance)
            else:
                print("conversation already in memory")
                utterance.answer = memory_conversation.answer
                utterance.content_output_list = memory_conversation.content_output_list

            print(f"[IPA LOS] {utterance}")

            result = QASimulationOutput(
                utterance=utterance,
                model=LLMType(_llm_config.LLM_IPA),
                ipa=IPA_YELP.ipa_name,
                response=response,
                poi_exists=poi_exists,
            )
            results.append(result)

        return results

    @staticmethod
    def simulate_turn(
        user_text: str,
        user_intent: str,
        user_id: str,
        current_content_input: Optional[CCContentInput],
        history: List[str],
        max_retries: int = 3,
        llm_type = os.environ.get('LLM_IPA'),
        **kwargs,
    ) -> Turn:
        """
        Process a single turn via the Yelp-based conv-navi service.
        """
        context = kwargs.get("context")
        user_location = None
        if isinstance(context, dict):
            user_location = context.get("location", {}).get("position")

        response_obj = send_request_conv_navi(
            user_text,
            user_location=user_location,
            user_id=user_id,
            max_retries=max_retries,
            llm_type=llm_type.value if isinstance(llm_type, LLMType) else str(llm_type)
        )

        sys_answer_text = response_obj.get("response", "System Error")

        out_list = [
            convert_yelp_navi_los_to_content_output(response_obj.get("retrieved_pois", []), current_content_input)
        ]


        poi_exists = False
        try:
            if current_content_input is not None:
                res = send_request_poi_exists(
                    navi_input=current_content_input,
                    user_location=user_location
                )
                poi_exists = bool(res.get("exists", False))
        except Exception:
            poi_exists = False

        user_intent_influences_fit = user_intent in USER_INTENTS_INFLUENCING_FITNESS

        turn = Turn(
            question=user_text,
            answer=sys_answer_text,
            question_intent=user_intent,
            content_input=current_content_input.model_copy() if current_content_input else None,
            content_output_list=out_list,
            poi_exists=poi_exists,
            user_intent_influences_fit=user_intent_influences_fit,
            raw_output=response_obj,
        )

        history.append(f"User: {user_text}")
        history.append(f"System: {sys_answer_text}")

        return turn

    @staticmethod
    def simulate_conversation(
        list_individuals: List[Conversation],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float = 10,
        do_visualize: bool = False,
        temperature: float = 0,
        context: object = None,
        config_path: str = "configs/features_cc_mt.json",
        max_retries: int = 3,
        min_turns: int = 2,
        max_turns: int = 5,
        max_repeats: int = 2,
        **kwargs,
    ) -> List[MultiTurnSimulationOutput]:

        feature_handler = FeatureHandler.from_json(config_path)
        utterance_gen = CCUtteranceGenerator(feature_handler=feature_handler)
        
        from llm.config import LLM_CLASSIFIER, LLM_GENERATOR, LLM_IPA
        llm_classifier = LLMType(LLM_CLASSIFIER)
        llm_type_utterance_gen = LLMType(LLM_GENERATOR)
        llm_ipa = LLMType(_llm_config.LLM_IPA)

        results = []

        for conversation_wrapper in list_individuals:
            conversation = conversation_wrapper[0]

            def id_from_ns(seed=0):
                ns = time.time_ns()        # current time in nanoseconds
                id_candidate = ((ns // 800) + seed*2) % 1000000
                return id_candidate
            if IPA_YELP.global_user_counter == 0:
                IPA_YELP.global_user_counter = id_from_ns()
            else:
                IPA_YELP.global_user_counter += 1
            user_id = IPA_YELP.global_user_counter

            conversation.assigned_user_id = user_id

            conversation = IPA_YELP.run_conversation_loop(
                conversation=conversation,
                feature_handler=feature_handler,
                utterance_gen=utterance_gen,
                llm_ipa=llm_ipa,
                llm_classifier=llm_classifier,
                context=context,
                max_retries=max_retries,
                min_turns=min_turns,
                max_turns=max_turns,
                max_repeats=max_repeats,
                llm_type_utterance_gen=llm_type_utterance_gen,
            )

            result = MultiTurnSimulationOutput(
                conversation=conversation,
                model=llm_ipa,
                ipa=IPA_YELP.ipa_name,
            )
            results.append(result)

        return results