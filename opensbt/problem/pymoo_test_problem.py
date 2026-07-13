from dataclasses import dataclass
from pymoo.core.problem import Problem
from pymoo.problems import get_problem
from opensbt.evaluation.critical import Critical

@dataclass
class PymooTestProblem(Problem):
    """ Problem class to test testing algorithms on mathematical problems implemented in pymoo. """
    
    def __init__(self, problem_name: str, critical_function: Critical, approx_eval_time: float =0.01):
        """ Instantiates a pymoo based problem with name "problem_name" as testing problem. """

        self.critical_function = critical_function
        self.problem_name = problem_name
        self.test_problem = get_problem(problem_name)
        self._instantiate_from_given_problem(self.test_problem) 
        self.design_names =  ["X" + str(index) for index in range(self.n_var)]
        self.objective_names = ["F" + str(index) for index in range(self.n_obj)]
        self.xosc = problem_name
        if approx_eval_time is not None:
            self.approx_eval_time = approx_eval_time

    def _instantiate_from_given_problem(self,problem):

        # HACK: Copy over attribute values pointer to two functions
        super().__init__(**problem.__dict__)
        if hasattr(problem, 'pareto_front'):
            #print("pareto front set")
            setattr(self,"pareto_front",problem.pareto_front)
                
        if hasattr(problem, '_calc_pareto_front'):
            setattr(self,"_calc_pareto_front",problem._calc_pareto_front)
         
        if hasattr(problem, '_calc_pareto_set'):
            setattr(self,"_calc_pareto_set",problem._calc_pareto_set)
        
    def _evaluate(self, X, out, *args, **kwargs):
        self.test_problem._evaluate(X,out,*args, **kwargs)
        # TODO retrieve fitness values and apply criticality function
        out["CB"] = []
        for f in out["F"]:
            out["CB"].append(self.critical_function.eval(f))
    
    def _calc_pareto_front(self):
        pass

    def _calc_pareto_set(self):
        pass

    def is_simulation(self):
        return False
    
    ''' In seconds '''
    def approx_sim_time(self):
        return 0.005


