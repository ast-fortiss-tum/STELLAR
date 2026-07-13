from typing import List
import numpy as np
from llm.config import DEBUG, LLM_VALIDATOR
import random
from llm.model.models import LOS
from llm.prompts import VALIDATION_PROMPT, VALIDATION_PROMPT_RESPONSE_LOS
from llm.llms import LLMType, pass_llm
import json
from llm.utils.embeddings_local import get_similarity
from dataclasses import asdict

from llm.utils.embeddings_openai import is_equal

def similarity_los(los_in: LOS, los_out: List[LOS]):
    
    if los_in == None or los_out == None or len(los_out) == 0 or los_out[0] == "None":
        return 1

    los = los_out[0]  # assume for now to have only one los

    def safe_get_similarity(a, b):
        # take for now only the first item
        if isinstance(a, (list, tuple)) and len(a) > 0:
            a = a[0]
        if isinstance(b, (list, tuple)) and len(b) > 0:
            b = b[0]

        if a in ["None",None] or b in ["None",None]:
            return None
        similarity = is_equal(str(a), str(b))
        if similarity == float('inf'):
            return None
        return similarity
    
    def safe_float_compare(a, b):
        if a in ["None",None] or b in ["None",None]:
            return None
        return float(a) > float(b)

    def safe_contains(container, item):
        # take for now only the first item
        if isinstance(item, (list, tuple)) and len(item) > 0:
            item = item[0]
        if len(container) == 0 or item in ["None",None] or container in ["None",None]:
            return None
        for item_c in container:
            if is_equal(str(item), str(item_c)):
                return True
        else:
            return None
            
    print("los.types:", los.types)

    print(f"[DEBUG] los_in.types: {los_in.types}, los.types: {los.types}")
    sim_type = safe_get_similarity(los_in.types, los.types)
    print(f"[DEBUG] sim_type: {sim_type}")

    print(f"[DEBUG] los.ratings: {los.ratings}, los_in.ratings: {los_in.ratings}")
    sim_ratings = safe_float_compare(los.ratings, los_in.ratings)
    print(f"[DEBUG] sim_ratings: {sim_ratings}")

    print(f"[DEBUG] los.foodtypes: {los.foodtypes}, los_in.foodtypes: {los_in.foodtypes}")
    sim_foodtypes = safe_contains(los.foodtypes, los_in.foodtypes)
    print(f"[DEBUG] sim_foodtypes: {sim_foodtypes}")

    print(f"[DEBUG] los.payments: {los.payments}, los_in.payments: {los_in.payments}")
    sim_payments = safe_contains(los.payments , los_in.payments)
    print(f"[DEBUG] sim_payments: {sim_payments}")

    print(f"[DEBUG] los_in.costs: {los_in.costs}, los.costs: {los.costs}")
    sim_costs = safe_get_similarity(los_in.costs, los.costs)
    print(f"[DEBUG] sim_costs: {sim_costs}")


    # Collect non-None similarity scores
    similarity_values = []

    weight_type = 2

    if sim_type is not None:
        similarity_values.append(weight_type * sim_type)
    if sim_ratings is not None:
        similarity_values.append(sim_ratings)
    if sim_payments is not None:
        similarity_values.append(sim_payments)
    if sim_foodtypes is not None:
        similarity_values.append(sim_foodtypes)
    if sim_costs is not None:
        similarity_values.append(sim_costs)

    print("similarity values:", similarity_values)
    if len(similarity_values) > 0:
        similarity = np.mean(similarity_values)
    else:
        similarity = 1
    #sim_costs = los_out["costs"] < los_in["costs"]
    # ignore location for now
    # TODO add here more comparisons with other fields
    # TODO extend to mulitple los objects
    # TODO improve comparison logic
    return similarity

def llm_validator_response_los(question: str,
                                los_out: LOS,
                                n: int = 1, 
                                llm_type = LLMType(LLM_VALIDATOR)):
    assert n >= 1
    if len(los_out) == 0:
        return 1
    
    # TODO for now just take the first one; integrate all later
    print("los_out:", los_out)
    los_out = los_out[0]
    # system response, utterance comparison
    answers = []
    prompt_eval = VALIDATION_PROMPT_RESPONSE_LOS.format(question,json.dumps(asdict(los_out), indent=2))
    max_retries = 5
    for t in np.linspace(0, 1, n):
        retries = 0
        while retries < max_retries:
            try:
                if not DEBUG:
                    answer = pass_llm(prompt_eval, temperature=t, llm_type=llm_type)
                else:
                    answer = str(random.random()) + " \n"
                answers.append(float(answer.strip()))
                break  # Success, exit retry loop
            except Exception as e:
                retries += 1
                if retries == max_retries:
                    # Optionally handle failure, e.g. append NaN or raise
                    # For now, let's raise to inform caller
                    raise RuntimeError(f"Failed after {max_retries} retries: {e}") from e
    return np.mean(answers)

def llm_validator_los(question: str, 
                      answer: str, 
                      los_in: LOS, 
                      los_out: LOS, 
                      n: int = 1,
                      llm_type = LLMType(LLM_VALIDATOR),
                      weight_reponse: float = 2): # TODO check which weights preferrable
    assert n >= 1

    # system response, utterance comparison
    answers = []
    prompt_eval = VALIDATION_PROMPT.format(question,answer)
    for t in np.linspace(0, 1, n):
        if not DEBUG:
            answer = pass_llm(prompt_eval, 
                              temperature=t, 
                              llm_type=llm_type)
        else:
            answer = str(random.random()) + " \n"
        answers.append(float(answer.strip()))

    score_response = round(np.mean(answers),3)

    # los comparison
    score_los = similarity_los(los_in=los_in, los_out=los_out)
    
    scores_all = [score_los] + [score_response]*weight_reponse

    print("score_los:", score_los)
    print("score_response:", score_response)

    return np.mean(scores_all)