from typing import ClassVar, Dict, Tuple

import numpy as np
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from llm.model.models import Utterance
from llm.utils.embeddings_local import get_embedding as get_embedding_local
from llm.utils.embeddings_openai import get_embedding as get_embedding_openai
from llm.utils.math import euclid_distance

class UtterancesDistance(BaseModel):
    embeddings_distance: float = 0.0                # dissimilarity based on questions
    embeddings_answer_distance: float = 0.0         # dissimilarity based on answers
    ordinal_vars_distance: float = 0.0
    categorical_vars_distance: float = 0.0
    vars_distance: float = 0.0

    embeddings_cache: ClassVar[Dict[Tuple[str, str], np.ndarray]] = {}

    @classmethod
    def _get_embedding(cls, u: Utterance, field: str, provider: str) -> np.ndarray:
        """
        Get embedding for a given Utterance field (question/answer) and provider (local/openai).
        Uses caching to avoid recomputation.
        """
        assert field in ("question", "answer"), f"Unsupported field: {field}"
        assert provider in ("local", "openai"), f"Unsupported provider: {provider}"

        text = getattr(u, field)
        key = (provider, text)  # simplified: identical text → identical embedding

        if key in cls.embeddings_cache:
            return cls.embeddings_cache[key]

        if provider == "local":
            embedding = np.array(get_embedding_local(text)).reshape(1, -1)
        else:
            embedding = np.array(get_embedding_openai(text)).reshape(1, -1)

        cls.embeddings_cache[key] = embedding
        return embedding

    @classmethod
    def get_embeddings_similarity(
        cls, a: Utterance, b: Utterance, use_local: bool, field: str = "question"
    ) -> float:
        """
        Calculate cosine similarity between two utterances
        based on embeddings of the specified field ("question" or "answer").
        """
        provider = "local" if use_local else "openai"
        embed_a = cls._get_embedding(a, field, provider)
        embed_b = cls._get_embedding(b, field, provider)
        return cosine_similarity(embed_a, embed_b)[0][0]

    @staticmethod
    def safe_divide(numerator: float, denominator: int) -> float:
        """Safely divide, handling division by zero."""
        return numerator / max(denominator, 1)

    @classmethod
    def calculate(
        cls, a: Utterance, b: Utterance, *, use_local_embeddings: bool = True,
        calclualate_answer_distace: bool = False
    ) -> "UtterancesDistance":
        """
        Calculate distances between utterances.
        Includes both question-based and answer-based embedding dissimilarities.
        """
        # Embedding-based similarity on questions
        q_similarity = cls.get_embeddings_similarity(a, b, use_local_embeddings, "question")
        q_dissimilarity = (1.0 - q_similarity) / 2.0

        # Embedding-based similarity on answers (output diversity)
        if calclualate_answer_distace:
            a_similarity = cls.get_embeddings_similarity(a, b, use_local_embeddings, "answer")
            a_dissimilarity = (1.0 - a_similarity) / 2.0
        else:
            a_dissimilarity = 0.0

        # Ordinal variables distance
        ordinal_vars_euclidean = euclid_distance(a.ordinal_vars, b.ordinal_vars)
        ordinal_vars_distance = cls.safe_divide(
            ordinal_vars_euclidean, len(a.ordinal_vars)
        )

        # Categorical variables distance
        categorical_vars_total_distance = np.sum(
            np.array(a.categorical_vars) != np.array(b.categorical_vars)
        )
        categorical_vars_distance = cls.safe_divide(
            categorical_vars_total_distance, len(a.categorical_vars)
        )

        # Combined variable distance
        total_distance = cls.safe_divide(
            ordinal_vars_euclidean + categorical_vars_total_distance,
            len(a.ordinal_vars) + len(a.categorical_vars),
        )

        return cls(
            embeddings_distance=q_dissimilarity,
            embeddings_answer_distance=a_dissimilarity, 
            ordinal_vars_distance=ordinal_vars_distance,
            categorical_vars_distance=categorical_vars_distance,
            vars_distance=total_distance,
        )