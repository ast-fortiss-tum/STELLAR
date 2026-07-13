from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import json
from typing import List


class RAGRetriever:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_utters: int = 500,
        csv_path: str = "examples/navi/data/multiwoz_first_turn_utterances_poi.csv",
        json_path: str = "examples/navi/data/user_data.json",
    ):
        """
        Initialize the retriever with paths and model configuration.

        Args:
            model_name (str): SentenceTransformer model name.
            max_utters (int): Maximum number of utterances to load from CSV.
            csv_path (str): Path to CSV file with utterances.
            json_path (str): Path to JSON file with seed data.
        """
        self.model_name = model_name
        self.max_utters = max_utters
        self.csv_path = csv_path
        self.json_path = json_path

        self.embedding_model = SentenceTransformer(self.model_name)
        self.data_corpus = self._load_data()
        self.faiss_index = self._build_faiss_index()

    def _load_data(self) -> List[str]:
        """Load corpus from CSV and JSON."""
        # Load CSV utterances
        df = pd.read_csv(self.csv_path)
        csv_utterances = df["utterance"].dropna().tolist()[: self.max_utters]

        # Load JSON seed corpus
        with open(self.json_path, "r") as f:
            seed_corpus = json.load(f)

        return seed_corpus + csv_utterances

    def _build_faiss_index(self):
        """Compute embeddings and build FAISS index."""
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

    def __getstate__(self):
        """Return state for pickling (skip unpicklable parts)."""
        state = self.__dict__.copy()
        # Remove objects that contain PyCapsule / native handles
        state["embedding_model"] = None
        state["faiss_index"] = None
        return state

    def __setstate__(self, state):
        """Recreate skipped attributes after unpickling."""
        self.__dict__.update(state)
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.model_name)
        if self.faiss_index is None and self.data_corpus is not None:
            self.faiss_index = self._build_faiss_index()

# --- Example usage ---
if __name__ == "__main__":
    retriever = RAGRetriever()
    query = "navigate me to restaurant"
    results = retriever.retrieve(query, top_k=5)
    print("Retrieved examples:", results)
