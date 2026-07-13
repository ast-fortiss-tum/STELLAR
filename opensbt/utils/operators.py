from pymoo.operators.crossover.sbx import SBX # type: ignore
from pymoo.operators.mutation.pm import PM # type: ignore
from pymoo.operators.sampling.lhs import LHS # type: ignore

def select_operator(operation,
                    config, 
                    **kwargs):
    """
    Selects either the default operator or a custom operator based on the condition.
    """
    if config.operators[operation] is None:
        if operation == "mut":
            operator = PM   
            if "prob" not in kwargs:
                kwargs["prob"] = config.prob_mutation
            if "eta" not in kwargs:
                kwargs["eta"] = config.eta_mutation
        elif operation == "cx":
            operator = SBX
            if "prob" not in kwargs:
                kwargs["prob"] = config.prob_crossover
            if "eta" not in kwargs:
                kwargs["eta"] = config.eta_crossover
        elif operation == "init":
            operator = LHS
        elif operation == "dup":
            return True
    else:
        operator = config.operators[operation]
    return operator(**kwargs)  # Passes the keyword arguments to the operator