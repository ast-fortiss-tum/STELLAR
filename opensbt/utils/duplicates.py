import numpy as np
from llm.model.models import Utterance
from pymoo.core.population import Population
from opensbt.config import DUPLICATE_COMP_PRECISION
from llm.utils import embeddings_local

def default_is_equal(a, b, precision=DUPLICATE_COMP_PRECISION):
    # applicable on numbers
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        # If both are scalar types (int or float), compare directly with rounding
        return np.round(a, precision) == np.round(b, precision)
    
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        # If both are numpy arrays, check if their types are int or float
        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            # Perform element-wise comparison after rounding
            return np.all(np.round(a, precision) == np.round(b, precision))
     
        if np.issubdtype(a.dtype, np.str_) and np.issubdtype(b.dtype, np.str_):
            # If both are strings, use embeddings_local.is_equal for comparison
            eq =  embeddings_local.is_equal(a[0], b[0], threshold=0.9)
            return eq
        
    raise ValueError("Datatype not supported")

def duplicate_free(population, is_equal=default_is_equal):
    inds = population.get("X")
    # support for utterances, HACK, should be done earlier
    if len(inds) > 0 and isinstance(inds[0][0], Utterance):
        inds_str = np.asarray([np.asarray([ind[0].question]) for ind in inds])
        dup_free = [population[i] for i in remove_duplicates(inds_str, is_equal)]
    else:
        dup_free = [population[i] for i in remove_duplicates(inds, is_equal)]
    return Population(individuals=dup_free)

def remove_duplicates(M, is_equal=default_is_equal):
    res = []
    size = M.shape[0]
    
    if size == 1:
        return [0]
    elif size == 0:
        return []

    D = []
    for i in range(0,size):
        V = np.array([v for v in M[i]])
        D.append(V)

    D = np.asarray(D)
    if np.issubdtype(M.dtype, np.number):  # Sorting only for numerical values
        I = np.lexsort([M[:, i] for i in reversed(range(M.shape[1]))])
    else:
        I = list(range(size))
    S = D[I, :]

    i = 0
    # filter duplicates
    while i < size - 1:
        res.append(I[i])

        while is_equal(S[i, :], S[i + 1, :]):
            if i == size - 2:
                return res
            else:
                i += 1
        i = i + 1 
        if i == (size - 1):
            res.append(I[i])
            return res    
    return res

if __name__ == "__main__":
    M0 = np.asarray([[1.00001,2],[1.00,2],[1.00081,2],[1.2,3]])
    assert ( remove_duplicates(M0) == [1,0,2,3] )

    M1 = np.asarray([[1],[2],[2],[4]])
    assert ( remove_duplicates(M1) == [0,1,3] )

    M2 = np.asarray([[1],[2],[3],[4]])
    assert ( remove_duplicates(M2) == [0,1,2,3] )

    M3 = np.asarray([[1],[2],[3],[3]])
    assert ( remove_duplicates(M3) == [0,1,2] )

    M4 = np.asarray([[1.2,3]])
    assert ( remove_duplicates(M4) == [0] )

    M5 = np.asarray([[1.2,3], [1.2, 3]])
    assert ( remove_duplicates(M5) == [0] or remove_duplicates(M5) == [1] )

    M6 = np.asarray([])
    assert ( remove_duplicates(M6) == [] )
    
    M7 = np.asarray([['I am hungry'], ['bananax'],['I need food'], ['apple'], ['banana']])
    print(remove_duplicates(M7))  # Example output will vary
    
    M8 = np.asarray([['Hello'], ["Hi"], ['bones'],['bananas'], ['window'], ['house']])
    print(remove_duplicates(M8))  # Example output will vary
    
