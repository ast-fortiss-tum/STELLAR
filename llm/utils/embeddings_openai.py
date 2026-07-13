from pymoo.core.duplicate import ElementwiseDuplicateElimination
from sklearn.metrics.pairwise import cosine_similarity
from llm.llm_openai import get_openai_client
from llm.config import DEPLOYMENT_NAME
import numpy as np

def get_embedding(text, model="text-embedding-ada-002", model_api = "gpt-4o-mini"):
    """Fetches the embedding vector for a given text using OpenAI's API."""
    client = get_openai_client(model_api)
    response = client.embeddings.create(input=text,
                                        model=model)
    return np.array(response.data[0].embedding)

def is_equal(a, b, threshold = 0.8):
    reference_embedding = get_embedding(a).reshape(1, -1)  # Correct reshaping
    embedding = get_embedding(b).reshape(1, -1)  # Correct reshaping
    score = cosine_similarity(reference_embedding, embedding)[0][0]
    return score > threshold

def get_similarity(a,b, scale = None):
    reference_embedding = get_embedding(a).reshape(1, -1)  # Correct reshaping
    embedding = get_embedding(b).reshape(1, -1)  # Correct reshaping
    score = cosine_similarity(reference_embedding, embedding)[0][0]
    if scale is not None:
        score = score/scale
    print(score)
    return score