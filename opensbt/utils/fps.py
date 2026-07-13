import itertools
import numpy as np
from scipy.spatial import Voronoi

from pymoo.core.sampling import Sampling
from pymoo.util.normalization import denormalize
from pymoo.util.misc import cdist

def fps_by_bounds(n_var, xl, xu, n_samples=1, verbose=False):
    # corners are added to the initial set
    corners = np.array(list(itertools.product(*np.array((xl, xu)).T.tolist())))
    val = corners

    # the first inner point is chosen randomly and added to the initial set
    first_point = denormalize(np.random.random((1, n_var)), xl, xu)
    val = np.vstack((val, first_point))

    # initialize the diagram
    vor = Voronoi(val, incremental=True)

    while val.shape[0] < n_samples:
        vertices = vor.vertices

        indices = np.array([np.all(vertice <= xu) and np.all(vertice >= xl) for vertice in vertices]).nonzero()[0]
        vertices = vertices[indices]

        # if none of the vertices are within the boundaries - choose a set of random points
        # the number of chosen points is the number of expected Voronoi vertices (obtained empirically)
        if vertices.size == 0:
            vertices = np.random.random((int(np.ceil(val.shape[0] * np.exp(-1.88 + n_var * 1.192))), n_var))
            vertices = denormalize(vertices, xl, xu)
            if verbose:
                print("WARNING: None of the Voronoi vertices are within the boundaries.")

        dist_matrix = cdist(val, vertices)
        arg = dist_matrix.min(axis=0).argmax()
        val = np.vstack((val, vertices[arg]))
        try:
            vor.add_points([vertices[arg]], restart=True)  # update the diagram
        except:
            vor = Voronoi(val, incremental=True)
            if verbose:
                print("WARNING: The diagram could not be updated. A new diagram was constructed.")

    if val.shape[0] > n_samples:
        if verbose:
            print("WARNING: The initial set is not complete.")
        val = val[0:n_samples]

    return val

def fps(problem, n_samples=1, verbose=False):
    return fps_by_bounds(problem.n_var, problem.xl, problem.xu, n_samples=n_samples, verbose=verbose)

class FPS(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        #verbose = kwargs["verbose"]
        return fps(problem, n_samples=n_samples, verbose=False)
