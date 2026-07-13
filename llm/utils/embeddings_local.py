# Import required libraries
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import time
import random
import logging as log

# Load a pre-trained model from Hugging Face (e.g., a BERT-based model)
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

def is_equal(a, b, threshold = 0.9):
    embedding1 = model.encode(a)
    embedding2 = model.encode(b)
    score = cosine_similarity([embedding1], [embedding2])[0][0]
    return score > threshold

def get_embedding(text):
    return model.encode(text)

def get_similarity(a,b, scale = None):
    reference_embedding = get_embedding(a).reshape(1, -1)  # Correct reshaping
    embedding = get_embedding(b).reshape(1, -1)  # Correct reshaping
    score = cosine_similarity(reference_embedding, embedding)[0][0]
    if scale is not None:
        score = score/scale
    return score

if __name__ == "__main__":
    # Example sentences to compare (simple variation for random generation)
    sample_sentences = [
        "Anti Apples.",
        "Apples.",
        "Cats are fun.",
        "Dogs are loyal.",
        "Birds can fly.",
        "Fish swim in water.",
        "Cars drive on roads.",
        "Bicycles are eco-friendly.",
        "The sun rises in the east.",
        "Water is essential for life."
    ]

    # List to store similarity scores
    similarities = []

    # Measure the time for 10 samples
    start_time = time.time()

    for _ in range(10):
        # Randomly select two sentences from the sample set
        sentence1, sentence2 = random.sample(sample_sentences, 2)
        
        # Compute embeddings for the sentences
        embedding1 = model.encode(sentence1)
        embedding2 = model.encode(sentence2)

        # Convert embeddings to PyTorch tensors (optional)
        embedding1_tensor = torch.tensor(embedding1)
        embedding2_tensor = torch.tensor(embedding2)

        # Calculate cosine similarity between the two embeddings
        similarity = cosine_similarity([embedding1], [embedding2])

        # Append similarity score to the list
        similarities.append(similarity[0][0])

    # Measure the total time taken
    end_time = time.time()
    execution_time = end_time - start_time

    # Calculate average similarity score
    average_similarity = sum(similarities) / len(similarities)

    # Output the results
    print(f"Average Cosine Similarity: {average_similarity}")
    print(f"Total Execution Time for 10 samples: {execution_time:.4f} seconds")
    print(f"Average Execution Time per sample: {execution_time / 10:.4f} seconds")
