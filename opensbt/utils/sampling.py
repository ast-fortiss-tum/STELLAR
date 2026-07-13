import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.util.normalization import denormalize
import logging as log

def cartesian_by_bounds(n_var, xl, xu, n_samples_one_axis):
    """ Generates aequidistant points in a grid shape.
    
        Input:  
                n_var: number of axis
                x_l: lower bound for each axis
                x_u: upper_bound for each axis
                n_samples_one_axis: number of samples for one axis
        Output: aequidistant points in a grid shape where space is defined by axis with lower/upper bounds xl/xu
        ((TODO Pass number of samples for each axis via algorithm definition))
    """
    log.info(n_samples_one_axis)
    n_samples_by_axis = [n_samples_one_axis] * n_var
    X = [np.linspace(0, 1, n) for n in n_samples_by_axis]
    grid = np.meshgrid(*X)
    grid_reshaped = [axis.reshape(-1) for axis in grid]
    val = np.stack(grid_reshaped, axis=1)
    return denormalize(val, xl, xu)

class CartesianSampling(Sampling):
    def _do(self, problem, n_samples_one_axis = 10, **kwargs):
        return cartesian_by_bounds(problem.n_var, problem.xl, problem.xu, n_samples_one_axis=n_samples_one_axis)
    
def cartesian_reference_set(problem, n_evals_by_axis=None):
    log.info("------ Computing cartesian reference set ------")
    sampling = CartesianSampling()
    # define the sampling rate if not given
    if n_evals_by_axis is None:
        n_evals_by_axis = [20] * problem.n_var
    Z = sampling(problem, n_evals_by_axis)

    out = problem.evaluate(X=Z.get("X"), return_as_dictionary=True)
    for key in out.keys():
        Z.set(key, out[key])
    feas = np.where(Z.get("CV") <= 0)[0]
    Z_feas = Z[feas]
    labels = np.where(Z_feas.get("CB"))[0]
    Z_labeled = Z_feas[labels]
    return Z_labeled
