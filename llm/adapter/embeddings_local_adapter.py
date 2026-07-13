from llm.utils.embeddings_local import get_similarity
import numpy as np
from llm.utils.math import euclid_distance

def get_similarity_individual(a,b, scale = False, invert = False):
    # Consider utterance structure
    score = get_similarity(a.get("X")[0].question, b.get("X")[0].question)
    if scale:
        score = (1 + score)/2
    return 1 - score if invert else score

def get_similarity_individual_discrete(a,b):
    # Consider utterance structure
    score = euclid_distance(a.get("X")[0].ordinal_vars, b.get("X")[0].ordinal_vars)
    return score

def get_disimilarity_individual(a, b, scale = True):
    return get_similarity_individual(a,b, scale=scale, invert = True)

def get_disimilarity_individual_discrete(a, b, scale = True):
    return get_similarity_individual_discrete(a,b)