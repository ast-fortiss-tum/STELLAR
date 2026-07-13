import numpy as np
from llm.config import DEBUG, LLM_VALIDATOR
import random
import ast
from llm.llms import LLMType, pass_llm
import json
from llm.prompts import VALIDATION_PROMPT_CATEGORY

categories = ['shopping', 'chitchat', 'other', 'supermarket', 'restaurant']

def llm_validator_category(answer, n=1, model = "ollama"):
    answers = []
    for t in np.linspace(0, 1, n):
        prompt_eval = VALIDATION_PROMPT_CATEGORY.format(answer)
        max_tries = 5
        tries = 0
        retry = True
        while retry and tries < max_tries:
          try:
              if not DEBUG:
                  answer = pass_llm(prompt_eval, 
                                    temperature=t, 
                                    system_message= "You are quantitative checker to which category an utterance belongs.",
                                    context= None,
                                    llm_type=LLM_VALIDATOR)
              else:
                  answer = str(random.random()) + " \n"
              answers.append(json.loads(answer))
              retry = False
          except Exception as e:
              tries += 1
              print("retrying")
              print("output", answer)
              if tries ==max_tries:
                  raise e
    result = []
    results_add = {category: 0 for category in categories}
    for item in answers:
        for category in categories:
            if category in item:
                results_add[category] += item[category]
    
    result = {}
    for category in categories:
        result[category] = results_add[category] / len(answers)

    for cat in categories:
        if cat not in result:
            result[cat] = 0
    
    # if len([num for num in answers if num > 0.5]) > len(answers)/2:
    #     return max(answers)
    # else:
    #     print("[validator] invalid")
    #     return min(answers)
    return result

if __name__ == "__main__":
    questions = {
        "test_false" : ["Food"] * 5,
        "test":["Food"]*4,
        "restaurant_my" : [
            "I want to eat something.",
            "I need some food.",
            "Show me restaurant.",
            "I want to eat some thing now.",
            "I need some food urgently tonight.",
            "Show me the nearest restaurant please.",
            "I need some food very quickly tonight.",
            "I must eat something fast tomorrow.",
            "I require lots of food immediately tonite."
        ],
        "restaurant" : [
            "I am hungry.",
            "I am in the mood for some food.",
            "I have been craving sushi all day.",
            "I could really go for something hearty.",
            "I am really craving something sweet right now.",
            
        ],
        "supermarket" : [
            "I need to buy groceries fast.",
            "I have no ingredients for cooking.",
            "Drive me to a supermarket.",
            "Where can I buy groceries?",
            "I need to find a LIDL"
        ],
        "shopping" : [
            "It is time for shopping.",
            "Where is the closest mall?",
            "Drive me to the closest shopping mall.",
            "Where can I buy cheap jeans?",
            "I need new clothes."
        ]}

    answers = {
    "test_false" : ["I cannot answer this.",
                    "I dont know what Sushi means.",
                    "Please repeat the request for a restaurant.",
                    "I cannot find what you are looking for.",
                    "I am sorry, I cannot help you with that."],
    "test" : ["I will drive you to a place with fish.",
                        "I will drive you to a fish restaurant.",
                        "I will drive you to a food market.",
                        "Should I order food for you?"],
    "restaurant_my" : [
        "Do you want me to navigate to a nearby restaurant?",
        "Do you want me to navigate to a nearby restaurant?",
        "The closest restaurant is El Giovanni Pizzeria, 500 meters away. Should I navigate you there?",
        "Do you want me to navigate to a nearby restaurant?",
        "Do you want me to navigate to a restaurant?",
        "The closest restaurant is El Giovanni Pizzeria, 500 meters away. Should I navigate you there?",
        "Would you like me to navigate to a nearby restaurant?",
        "Would you like me to navigate you to a nearby restaurant or fast food place tomorrow?",
        "It looks like you need a place to eat tonight. Would you like me to navigate you to a nearby restaurant?"
    ],
    "restaurant": [
        "Sure, I can help you find a nearby restaurant.",
        "Let's look for a good place to eat nearby.",
        "There are several sushi places around. Want me to show you some options?",
        "I'll find a place that serves hearty meals near you.",
        "I can find dessert places nearby. Would you like ice cream or pastries?"
    ],
    "supermarket": [
        "I’ll locate the nearest supermarket for you.",
        "Let me find a place where you can get cooking ingredients.",
        "Sure, I’ll navigate to the nearest supermarket.",
        "Here are a few grocery stores nearby.",
        "LIDL locations nearby are being retrieved now."
    ],
    "shopping": [
        "Let me find a good shopping location for you.",
        "I’ll locate the nearest mall.",
        "Navigating to the closest shopping mall now.",
        "Let’s search for stores that sell affordable jeans.",
        "I’ll show you stores nearby with clothing options."
    ]
    }


    for (cat, quest) in questions.items():
        print(f"Category: {cat}")
        for i, q in enumerate(quest):
            res = llm_validator_category(answer = answers[cat][i],
                                        #model="gpt-35-turbo-1106")
                                        model="ollama")


            print(cat + ":", res[0][cat] if cat in res[0].keys() else res[0]["restaurant"])