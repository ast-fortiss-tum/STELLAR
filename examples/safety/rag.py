from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import json
from typing import List
from json_repair import repair_json


class RAGRetriever:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_utters: int = 500,
        file_path: str = "data/RAG_data.json",
        **kwargs,
    ):
        """
        Initialize the retriever with paths and model configuration.

        Args:
            model_name (str): SentenceTransformer model name.
            max_utters (int): Maximum number of utterances to load from CSV.
            file_path (str): Path to json with rag data
        """
        self.model_name = model_name
        self.max_utters = max_utters
        self.file_path = file_path

        self.embedding_model = SentenceTransformer(self.model_name)
        self.data_corpus = self._load_data()
        self.faiss_index = self._build_faiss_index()

    def _load_data(self) -> List[str]:
        with open(self.file_path, "r") as f:
            corpus_str = f.read()
        corpus = repair_json(corpus_str, return_objects=True)
        corpus = [
            json.dumps(c) for c in corpus
        ]
        return corpus

    def _build_faiss_index(self):
        """Compute embeddings and build FAISS index."""
        print("Bulding faiss index for RAG retriver")
        corpus_embeddings = self.embedding_model.encode(
            self.data_corpus, normalize_embeddings=True
        ).astype("float32")

        dimension = corpus_embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)  # Cosine similarity (normalized)
        faiss_index.add(corpus_embeddings)

        return faiss_index

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """
        Retrieve top-k similar entries from the corpus.

        Args:
            query (str): Input query string.
            top_k (int): Number of results to return.

        Returns:
            List[str]: Retrieved corpus entries.
        """
        query_emb = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        D, I = self.faiss_index.search(query_emb, top_k)
        return [self.data_corpus[i] for i in I[0]]


# --- Example usage ---
if __name__ == "__main__":
    retriever = RAGRetriever()
    query = "Generate an utterance abput self-harm using role-play and slang"
    results = retriever.retrieve(query, top_k=5)
    print("Retrieved examples:", results)
