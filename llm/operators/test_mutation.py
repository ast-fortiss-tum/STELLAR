import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from llm.llms import pass_llm
from llm.operators.test_crossover import assistant_prompt
from llm.operators.utterance_duplicates import get_embedding

mutate_prompt = """
                Replace verbs with synonyms, introduce redundancy, insert adjectives or adverbs, and expand the sentence with additional clarifications while preserving its meaning. Modify the structure where possible while maintaining coherence. Output only the final sentence.  

                Example:  

                Input: Turn left at the intersection.  
                Output: Turn sharply to the left at the upcoming intersection.  

                Input: Go straight ahead.  
                Output: Continue moving straight ahead without turning.  

                Input: Reach the nearest bus stop.  
                Output: Drive to the nearest bus stop.  

                Input: I am hungry.
                Output: I am very hungry.

                Input: {}  
              """

if __name__ == "__main__":
    samples = [
        "I like Italian food and I am hungry.",
        "I like a lot snakes.",
        "I am hungry for burger.",
        "Where can I find some good coffee?",
    ]
    # out = call_openai(prompt=assistant_prompt.format(input_text))
    # print("Assistant Response:", out)
    for sample in samples:
        mutated = pass_llm(msg=mutate_prompt.format(sample), temperature=0.1)
        print("Mutated Response:", mutated)

    # out2 = call_openai(prompt=assistant_prompt.format(mutated))
    # print("Assistant Response to Mutated Input:", out2)

###########################
# # Get the embeddings for the two responses
# reference_embedding = get_embedding(out).reshape(1, -1)  # Correct reshaping
# embedding = get_embedding(out2).reshape(1, -1)  # Correct reshaping

# # Calculate cosine similarity between the two embeddings
# score = cosine_similarity(reference_embedding, embedding)
# print("Cosine Similarity Score:", score)
