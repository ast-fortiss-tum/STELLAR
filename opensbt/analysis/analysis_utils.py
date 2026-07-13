
import pandas as pd
from opensbt.model_ga.individual import IndividualSimulated
from opensbt.model_ga.population import PopulationExtended

def read_critical_set(filename_critical_inds):
    individuals = []
    table = pd.read_csv(filename_critical_inds)
    n_var = -1
    k = 0
    # identify number of objectives
    for col in table.columns[1:]:
        if col.startswith("Fitness_"):
            n_var = k
            break
        k = k + 1
    for i in range(len(table)):
        X = table.iloc[i, 1:n_var + 1].to_numpy()
        F = table.iloc[i, n_var + 1:-1].to_numpy()
        ind = IndividualSimulated()
        ind.set("X", X)
        ind.set("F", F)
        ind.set("CB", 1)
        individuals.append(ind)
    return PopulationExtended(individuals=individuals)

