import numpy as np
from examples.navi.navi_utterance_generator import NaviUtteranceGenerator
from llm.features.feature_handler import FeatureHandler
from llm.llms import LLMType, pass_llm
from llm.prompts import SYSTEM_PROMPT
from llm.utils.seed import set_seed
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.metrics import silhouette_score
from itertools import combinations
from scipy.spatial.distance import pdist

from llm.model.models import Utterance
from llm.eval.utterances_distance import UtterancesDistance  # your previous class

def cluster_utterances_vars(utterances, k_min=2, k_max=None, random_state=42):
    """
    Cluster utterances based on vars_distance using k-medoids and silhouette method.

    Parameters:
        utterances (List[Utterance]): List of utterances
        k_min (int): Minimum number of clusters to try
        k_max (int): Maximum number of clusters to try
        random_state (int): For reproducibility

    Returns:
        dict: {
            'best_k': optimal number of clusters,
            'clusters': list of lists of utterance indices,
            'medoids': indices of medoid utterances,
            'avg_medoid_distance': float,
            'max_medoid_distance': float
        }
    """

    n = len(utterances)
    if k_max is None:
        k_max = n

    if n < k_min:
        print("Number of utterances smaller than k_min.")
        return {
            'best_k': None,
            'clusters': None,
            'medoids': None,
            'avg_medoid_distance': None,
            'max_medoid_distance': None,
            'avg_max_medoid_distance': None,
        }

    # --- Step 1: compute pairwise vars_distance ---
    distance_matrix = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        dist = UtterancesDistance.calculate(utterances[i], utterances[j]).vars_distance
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist

    # --- Step 2: silhouette analysis to find best k ---
    best_k = k_min
    best_silhouette = -1
    best_clusters = None
    best_medoids = None

    for k in range(k_min, min(k_max, n) + 1):
        # Initialize medoids randomly
        np.random.seed(random_state)
        initial_medoids = np.random.choice(n, size=k, replace=False).tolist()
        kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
        kmedoids_instance.process()
        clusters = kmedoids_instance.get_clusters()
        medoids = kmedoids_instance.get_medoids()

        # Create a label array for silhouette_score
        labels = np.empty(n, dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for idx in cluster:
                labels[idx] = cluster_idx

        # Silhouette requires at least 2 clusters with >1 element each
        if len(set(labels)) < 2:
            continue
        try:
            sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
        except ValueError:
            continue

        if sil_score > best_silhouette:
            best_silhouette = sil_score
            best_k = k
            best_clusters = clusters
            best_medoids = medoids

    # compute distances between medoids ---
    if best_k > 1:
        medoid_vectors = [utterances[m] for m in best_medoids]
        medoid_distances = []
        max_per_medoid = []

        for i, j in combinations(range(len(medoid_vectors)), 2):
            dist = UtterancesDistance.calculate(medoid_vectors[i], medoid_vectors[j]).vars_distance
            medoid_distances.append(dist)

        # Compute max distance per medoid
        for i, u_i in enumerate(medoid_vectors):
            distances_to_others = [
                UtterancesDistance.calculate(u_i, medoid_vectors[j]).vars_distance
                for j in range(len(medoid_vectors)) if j != i
            ]
            max_per_medoid.append(max(distances_to_others))

        avg_medoid_distance = np.mean(medoid_distances)
        max_medoid_distance = np.max(medoid_distances)
        avg_max_medoid_distance = np.mean(max_per_medoid)
    else:
        avg_medoid_distance = 0.0
        max_medoid_distance = 0.0
        avg_max_medoid_distance = 0.0

    return {
        'best_k': best_k,
        'clusters': best_clusters,
        'medoids': best_medoids,
        'avg_medoid_distance': avg_medoid_distance,
        'max_medoid_distance': max_medoid_distance,
        'avg_max_medoid_distance': avg_max_medoid_distance,
    }
if __name__ == "__main__":
    fhandler = FeatureHandler.from_json("configs/features_simple_judge.json")
    seed = 22
    set_seed(seed)
    gen = NaviUtteranceGenerator(fhandler, use_rag = False)
    utterances = []
    for i in range(50):
        sample_ord, sample_cat, continuous_cat = fhandler.sample_feature_scores()
        utter = gen.generate_utterance(seed=None,
                                    ordinal_vars=sample_ord[1],
                                    categorical_vars=sample_cat[1],
                                    llm_type=LLMType.GPT_35_TURBO
        )
        utter.answer =  pass_llm(
                            msg=utter.question,
                            llm_type=LLMType.GPT_35_TURBO,
                            temperature=0,
                            context={},
                            system_message=SYSTEM_PROMPT
                        )

        
        utterances.append(utter)
        print(fhandler.map_categorical_indices_to_labels(sample_cat[1]))
        print(fhandler.map_numerical_scores_to_labels(sample_ord[1]))
        print(utter.question)
        print(utter.answer)
        print("\n")

    res = cluster_utterances_vars(utterances=utterances,random_state=22)
    print(res)