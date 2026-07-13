from pymoo.util.nds import efficient_non_dominated_sort
from pymoo.core.population import Population

def calc_nondominated_individuals(population: Population):
    F = population.get("F")
    best_inds_index = efficient_non_dominated_sort.efficient_non_dominated_sort(F)[0]
    best_inds = [population[i] for i in best_inds_index]
    return best_inds

def get_nondominated_population(population: Population):
    return Population(individuals=calc_nondominated_individuals(population))