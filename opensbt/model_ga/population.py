from pymoo.core.population import Population
from opensbt.model_ga.individual import IndividualSimulated
import numpy as np

class PopulationExtended(Population):
    """This class extends pymoos' Population class to apply population based operators.
    """
    def __new__(cls, individuals=[]):
        if isinstance(individuals, IndividualSimulated):
            individuals = [individuals]
        return np.array(individuals).view(cls)

    def divide_critical_non_critical(self):
        if self.size == 0:
            return self, self
        n_crit = int(sum(self.get("CB")))
        critical_population = Population.empty(n_crit)
        notcritical_population = Population.empty(len(self) - n_crit)
        i_crit = 0
        i_ncrit = 0
        for i in range(len(self)):
            if self[i].get("CB") == 1:
                critical_population[i_crit] = self[i]
                i_crit +=1
            else:
                notcritical_population[i_ncrit] = self[i]
                i_ncrit +=1
        return critical_population, notcritical_population

    @classmethod
    def empty(cls, size=0):
        individuals = [IndividualSimulated() for _ in range(size)]
        return PopulationExtended.__new__(cls, individuals)
            
def pop_from_array_or_individual(array, pop=None):
    # the population type can be different - (different type of individuals)
    if pop is None:
        pop = Population.empty()

    # provide a whole population object - (individuals might be already evaluated)
    if isinstance(array, Population):
        pop = array
    elif isinstance(array, np.ndarray):
        pop = pop.new("X", np.atleast_2d(array))
    elif isinstance(array, IndividualSimulated):
        pop = Population.empty(1)
        pop[0] = array
    else:
        return None
    return pop
