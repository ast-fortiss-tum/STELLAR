from pymoo.core.population import Population
from pymoo.core.problem import Problem

def evaluate_individuals(population: Population, problem: Problem):
    out_all = {}
    problem._evaluate(population.get("X"), out_all)
    for index, ind in enumerate(population):
        dict_individual = {}
        for item,value in out_all.items():
            dict_individual[item] = value[index]
        ind.set_by_dict(**dict_individual)
    return population
