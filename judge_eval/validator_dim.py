from collections import Counter
import json
import random
import time
import numpy as np
from judge_eval.prompts import PROMPT_EVALUATE_REQUEST_RESPONSE_DIMENSIONS
from llm.config import DEBUG, LLM_VALIDATOR
from llm.llms import LLMType, pass_llm
from json_repair import repair_json

def llm_validator_question_answer(
    question: str, 
    answer: str, 
    n=1, 
    llm_type=None,
    aggregator="mean"  # "majority" or "mean"
):
    if llm_type is None:
        llm_type = LLMType(LLM_VALIDATOR)
    
    assert n >= 1
    answers = []
    justifications = []

    prompt_eval = PROMPT_EVALUATE_REQUEST_RESPONSE_DIMENSIONS.format(question, answer)
    
    # print("prompt_eval:", prompt_eval)
    
    for _ in range(n):
        if not DEBUG and (llm_type != LLMType.MOCK):
            attempts = 0
            success = False
            while attempts < 5 and not success:
                try:
                    raw_answer = pass_llm(
                        prompt_eval, 
                        temperature = 0.5 if n > 1 else 0, 
                        llm_type=llm_type
                    )
                    response_json = json.loads(repair_json(raw_answer))
                    scores = list(response_json["scores"].values())
                    answers.append(scores)
                    justifications.append(response_json.get("justification", ""))
                    success = True
                except (json.JSONDecodeError, KeyError) as e:
                    attempts += 1
                    print(f"Parsing failed (attempt {attempts}/5): {e}")
                    time.sleep(0.5)
            if not success:
                print("Failed to parse scores after 5 attempts. Using default value.")
                answers.append([0.0])
        else:
            answers.append([random.random()])

    answers_array = np.array(answers)  # shape: (n, num_categories)
    
    if aggregator == "mean":
        mean_per_category = np.mean(answers_array, axis=0)
        final_scores = [round(v) for v in mean_per_category]
    elif aggregator == "majority":
        final_scores = []
        for dim_scores in answers_array.T:  # iterate per category
            rounded_scores = [round(s) for s in dim_scores]
            counter = Counter(rounded_scores)
            most_common = counter.most_common()
            if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                final_scores.append(most_common[0][0])
            else:  # tie
                final_scores.append(round(np.mean(rounded_scores)))
    else:
        raise ValueError(f"Unknown aggregator: {aggregator}")

    return final_scores, answers, justifications

if __name__ == "__main__":
    # Example question and system response
    question = "Find me an Italian restaurant with 4 stars."
    answer = "I have found an Italian restaurant called Bella with 4 stars. Let me know if you need directions."
    
    # Call the validator
    avg_score, all_scores = llm_validator_question_answer(
        question, 
        answer, 
        n=3, 
        llm_type=LLMType.LLAMA3_2
    )
    
    print("Question:", question)
    print("Answer:", answer)
    print(f"Validation average score: {avg_score}")
    print("All individual scores:", all_scores)
