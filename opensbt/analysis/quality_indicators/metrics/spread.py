import numpy as np

def spread(F):
    """
    Returns Delta metric of a set F. Only defined for 2D. It takes a value between 0 and 1, and has a higher value
    for worse distributions.
    Reference: K.Deb - "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II", IV. B.
    """
    N = len(F)
    if N < 3:
        # the given set should have at least three points. If not, return the worst value for the metric.
        return 1.0

    F = np.array(F)
    F.sort(axis=0)

    # Frobenius norm is used (2-norm for vectors)
    vectors = F[1:] - F[:-1]
    distances = np.linalg.norm(vectors, axis=-1)
    d_l = distances[0]
    d_f = distances[-1]
    d_average = np.mean(distances)
    nominator = d_f + d_l + np.sum(distances[1:-1] - d_average)
    denominator = d_f + d_l + (N-1)*d_average
    delta = nominator / denominator

    return delta