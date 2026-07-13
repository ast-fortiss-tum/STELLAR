from llm.utils.embeddings_openai import get_similarity
import numpy as np

def get_similarity_individual(a,b, scale = True, invert = True):
    # Consider utterance structure
    score = get_similarity(a.get("X")[0].question, b.get("X")[0].question)
    if scale:
        score = (1 + score)/2
    return 1 - score if invert else score