import pandas as pd
import numpy as np
import math


def get_n_crit_grid(individuals, b_min, b_max, n_cells):
    """ This function divides the search space into a grid width equidistance cells and evaluates how many cells have at least one
        failing test input places (diversity aware failure counting).
    """
    
    transformed = []
    d = np.abs(b_max - b_min)
    map = {}
    
    if len(individuals) > 0:
        n_f = len(individuals[0])
        # print(f"[get_n_crit_grid] problrm is {n_f} dimensional")
        for ind in individuals:
            coord = np.zeros(n_f)
            for i in range(0,n_f):
                normalized = np.abs(ind[i] - b_min[i]) /d[i]
                # print(f"[get_n_crit_grid] normalized for dimension {i}: {normalized}")
                v = np.floor(normalized*n_cells)
                # print(f"[get_n_crit_grid] value for dimension {i} is: {v}")
                if v == n_cells:
                    v = n_cells - 1
                coord[i] = v
                assert(coord[i] >= 0)
                assert(coord[i] <= n_cells)

            map[str(ind)] = coord
            transformed.append(coord)

        grid = np.zeros((int(math.pow(n_cells,n_f)),1), dtype=bool)
        for r in transformed:
            num = 0
            for i in range(0,len(r)):
                num += math.pow(n_cells, n_f - i - 1) * (r[i]) 
            # set to true
            grid[int(num)] = True
        return grid[grid>=1].size, grid
    else:
        return 0, None