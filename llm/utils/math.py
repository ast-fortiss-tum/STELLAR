import math
import numpy as np

def euclid_distance(a,b):
    squared_diffs = [(x - y) ** 2 for x, y in zip(a, b)]
    distance = math.sqrt(sum(squared_diffs))
    return distance

def mae(a, b) -> float:
    a = np.array(a)
    b = np.array(b)
    return np.mean(np.abs(a - b))

def mse(a, b) -> float:
    a = np.array(a)
    b = np.array(b)
    return np.mean((a - b) ** 2)
