
import numpy as np
import pymoo.gradient.toolbox as anp
from pymoo.util.misc import at_least_2d_array
from opensbt.simulation.simulator import SimulationOutputBase

from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseEvaluationFunction
from pymoo.core.problem import LoopedElementwiseEvaluation

class SimulationProblem(Problem):
    
    """
    This class extends the pymoo's Problem class to integrate the handling of simulation output-related data.
    """
    
    def __init__(self,
                 n_var,
                 n_obj=1,
                 n_ieq_constr=0,
                 n_eq_constr=0,
                 xl=None,
                 xu=None,
                 vtype=None,
                 vars=None,
                 elementwise=False,
                 elementwise_func=ElementwiseEvaluationFunction,
                 elementwise_runner=LoopedElementwiseEvaluation(),
                 replace_nan_values_by=None,
                 exclude_from_serialization=None,
                 callback=None,
                 strict=True,
                 **kwargs):

        """

        Parameters
        ----------
        n_var : int
            Number of Variables

        n_obj : int
            Number of Objectives

        n_ieq_constr : int
            Number of Inequality Constraints

        n_eq_constr : int
            Number of Equality Constraints

        xl : np.array, float, int
            Lower bounds for the variables. if integer all lower bounds are equal.

        xu : np.array, float, int
            Upper bounds for the variable. if integer all upper bounds are equal.

        vtype : type
            The variable type. So far, just used as a type hint.

        """

        # if variables are provided directly
        if vars is not None:
            n_var = len(vars)

        # number of variable
        self.n_var = n_var

        # number of objectives
        self.n_obj = n_obj

        # number of inequality constraints
        self.n_ieq_constr = n_ieq_constr if "n_constr" not in kwargs else max(n_ieq_constr, kwargs["n_constr"])

        # number of equality constraints
        self.n_eq_constr = n_eq_constr

        # type of the variable to be evaluated
        self.data = dict(**kwargs)

        # the lower bounds, make sure it is a numpy array with the length of n_var
        self.xl, self.xu = xl, xu

        # a callback function to be called after every evaluation
        self.callback = callback

        # if the variables are provided in their explicit form
        self.vars = vars

        # the variable type (only as a type hint at this point)
        self.vtype = vtype

        # the functions used if elementwise is enabled
        self.elementwise = elementwise
        self.elementwise_func = elementwise_func
        self.elementwise_runner = elementwise_runner

        # whether the shapes are checked strictly
        self.strict = strict

        # if it is a problem with an actual number of variables - make sure xl and xu are numpy arrays
        if n_var > 0:

            if self.xl is not None:
                if not isinstance(self.xl, np.ndarray):
                    self.xl = np.ones(n_var) * xl
                self.xl = self.xl.astype(float)

            if self.xu is not None:
                if not isinstance(self.xu, np.ndarray):
                    self.xu = np.ones(n_var) * xu
                self.xu = self.xu.astype(float)

        # this defines if NaN values should be replaced or not
        self.replace_nan_values_by = replace_nan_values_by

        # attribute which are excluded from being serialized
        self.exclude_from_serialization = exclude_from_serialization



    def _evaluate_elementwise(self, X, out, *args, **kwargs):

        # create the function that evaluates a single individual
        f = self.elementwise_func(self, args, kwargs)

        # execute the runner
        elems = self.elementwise_runner(f, X)

        # for each evaluation call
        for elem in elems:

            # for each key stored for this evaluation
            for k, v in elem.items():

                # if the element does not exist in out yet -> create it
                if out.get(k, None) is None:
                    out[k] = []

                out[k].append(v)

        # convert to arrays (the none check is important because otherwise an empty array is initialized)
        for k in out:
            if out[k] is not None:
                if not isinstance(out[k][-1], SimulationOutputBase):
                    out[k] = anp.array(out[k])

    def evaluate(self,
                 X,
                 *args,
                 return_values_of=None,
                 return_as_dictionary=False,
                 **kwargs):

        if return_values_of is None:
            return_values_of = ["F"]
            if self.n_ieq_constr > 0:
                return_values_of.append("G")
            if self.n_eq_constr > 0:
                return_values_of.append("H")

        # make sure the array is at least 2d. store if reshaping was necessary
        if isinstance(X, np.ndarray) and X.dtype != object:
            X, only_single_value = at_least_2d_array(X, extend_as="row", return_if_reshaped=True)
            assert X.shape[1] == self.n_var, f'Input dimension {X.shape[1]} are not equal to n_var {self.n_var}!'
        else:
            only_single_value = not (isinstance(X, list) or isinstance(X, np.ndarray))

        # this is where the actual evaluation takes place
        _out = self.do(X, return_values_of, *args, **kwargs)

        out = {}
        for k, v in _out.items():

            # copy it to a numpy array (it might be one of jax at this point)
            v = np.array(v)

            # in case the input had only one dimension, then remove always the first dimension from each output
            if only_single_value:
                v = v[0]

            # if the NaN values should be replaced
            if self.replace_nan_values_by is not None:
                v[np.isnan(v)] = self.replace_nan_values_by

            out[k] = v
            # try:
            #     out[k] = v.astype(np.float64)
            # except:
            #     out[k] = v

        if self.callback is not None:
            self.callback(X, out)

        # now depending on what should be returned prepare the output
        if return_as_dictionary:
            return out

        if len(return_values_of) == 1:
            return out[return_values_of[0]]
        else:
            return tuple([out[e] for e in return_values_of])

    def pareto_front_n_points(self, n_points=1000):
        if self.pareto_front is not None:
            pf = self.pareto_front(n_points)
        else:
            pf = None
            print("No analytical solution provided for the given problem.")
        return pf
