from dataclasses import dataclass
from typing import Optional, Any
import pydantic

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import DuplicateElimination
from pymoo.core.sampling import Sampling
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.operators.crossover.sbx import SBX # type: ignore
from pymoo.operators.mutation.pm import PM # type: ignore
from pymoo.operators.sampling.lhs import LHS # type: ignore
from pymoo.core.repair import Repair, NoRepair


class SearchOperators(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @pydantic.field_serializer("crossover", "sampling", "duplicate_elimination", "mutation")
    def _ser_operator(self, operator):
        return str(operator)
    
    crossover: Crossover = SBX()
    sampling: Sampling = LHS()
    duplicate_elimination: DuplicateElimination = NoDuplicateElimination()
    mutation: Mutation = PM()
    repair: Repair = NoRepair()


class SearchConfiguration(pydantic.BaseModel):
    """ This class holds all configuration parameter related to opimization algorithms
    """

    n_generations: int = 2
    population_size: int = 2
    maximal_execution_time: Any = None
    num_offsprings: Optional[int] = None
    prob_crossover: float = 0.7
    eta_crossover: int = 20
    prob_mutation: Optional[float] = None
    eta_mutation: int = 15
    
    # NSGAII-DT specific
    inner_num_gen: Optional[int] = 4
    max_tree_iterations: Optional[int] = 4
    n_func_evals_lim: Optional[int] = 500

    # metrics
    ref_point_hv: Any = None
    nadir: Any = None
    ideal:Any = None
    
    seed: Any = None
    
    operators: SearchOperators = SearchOperators()

    n_repopulate_max: float = 0.3

    archive_threshold: float = 0.5

    # related to cartesian sampling
    n_samples_per_feature: int = 5

    results_folder: str =  None

@dataclass
class DefaultSearchConfiguration:
    """ This class holds all configuration parameter initialized with default values 
    """
    #TODO create a search configuration file specific for each algorithm

    n_generations = 5 
    population_size = 20
    maximal_execution_time = None
    num_offsprings = None
    prob_crossover = 0.7
    eta_crossover = 20
    prob_mutation = None
    eta_mutation = 15

    # NSGAII-DT specific
    inner_num_gen = 4
    max_tree_iterations = 4
    n_func_evals_lim = 500

    # metrics
    ref_point_hv = None
    nadir = ref_point_hv

    seed = None
    
    operators= {
        "cx" : None,
        "mut" : None,
        "dup" : None,
        "init" : None
    }
    
    custom_params = {  # to be forwarded to operators
        
    }

    n_repopulate_max = 0.3

    archive_threshold = 0.5

    n_samples_per_feature = 5

    results_folder = None