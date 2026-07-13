import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from llm.llms import pass_llm
from llm.operators.utterance_duplicates import get_embedding

crossover_prompt = """You are an utterance recombination system. Your task is to take two input utterances, U1 and U2, and generate two new recombined utterances, U1' and U2'.  

                    - Each new utterance should contain partial information from both U1 and U2.  
                    - U1' should not be identical the same as U1, and U2' should not be same as U2.  
                    - The recombination should be performed in a meaningful way while preserving coherence.  
                    - Output exactly two recombined utterances (U1' and U2'), one per line.  
                    - Do not include explanations, just output the results.  
                    - Do not repeat any utterances from the input.
                    - The resulting sentences should not change their negative or positive semantics.

                    **Example 1:**  
                    **Input:**  
                    U1: "Navigate to the nearest gas station with the lowest fuel prices."  
                    U2: "Find the fastest route to a charging station with available spots."  

                    **Output:**  
                    "Navigate to the nearest charging station with the lowest fuel prices."  
                    "Find the fastest route to a gas station with available spots."  

                    **Example 2:**  
                    **Input:**  
                    U1: "Find a highly rated Italian restaurant nearby."  
                    U2: "Navigate to the closest café with outdoor seating."  

                    **Output:**  
                    "Find a highly rated Italian restaurant with outdoor seating."  
                    "Navigate to the closest café that serves Italian food."  

                    Now, recombine the following utterancs:
                    Utterance 1: {}
                    Utterance 2: {}  
                    """


assistant_prompt = """
                    You are an in-car AI assistant with access to relevant contextual information.
                    Your primary role is to assist the driver with navigation, nearby locations, vehicle controls, and general queries.
                    Respond concisely and accurately based on available knowledge.
                    Do not generate false information or speculate.

                    Example interactions:
                    User: Show me nearby restaurants.  
                    Assistant: The closest restaurant is El Giovanni Pizzeria, 500 meters away.  

                    User: Turn on the AC.  
                    Assistant: The air conditioning is now on.  

                    Now, respond to the following user request:  
                    User: {}
                    """

input_text_1 = "I am looking for a hotel."
input_text_2 = "I need to locate a place to eat."


if __name__ == "__main__":
    out = pass_llm(msg=crossover_prompt.format(input_text_1, input_text_2))
    print("Assistant Response:", [resp.strip().strip('"') for resp in out.split("\n")])

    out2 = pass_llm(msg=assistant_prompt.format(out))
    print("Assistant Response to Crossover Input:", out2)

###########################
# # Get the embeddings for the two responses
# reference_embedding = get_embedding(out).reshape(1, -1)  # Correct reshaping
# embedding = get_embedding(out2).reshape(1, -1)  # Correct reshaping

# # Calculate cosine similarity between the two embeddings
# score = cosine_similarity(reference_embedding, embedding)
# print("Cosine Similarity Score:", score)
