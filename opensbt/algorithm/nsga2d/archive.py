from itertools import permutations
from typing import List, Tuple
import logging as log
from pymoo.core.individual import Individual

import numpy as np
from typing import List

def euclidean_dist(ind_1, ind_2, bounds = None):
    if bounds is not None:
        up = bounds[0]
        low = bounds[1]
        # Normalize the vectors ind_1 and ind_2 based on the bounds
        norm_ind_1 = (ind_1.get("X") - low) / (up - low)
        norm_ind_2 = (ind_2.get("X") - low) / (up - low)
        
        return np.linalg.norm(norm_ind_1 - norm_ind_2)
    else:
        return np.linalg.norm(ind_1.get("X") - ind_2.get("X"))

class IndividualSet(set):
    pass

class Archive(IndividualSet):
    def process_population(self, pop: List):
        raise NotImplemented()
    
class SmartArchiveInput(List):
    def __init__(self, archive_threshold, bounds = None):
        super().__init__()
        self.archive_threshold = archive_threshold
        self.bounds = bounds

    def closest_individual_from_ind(self, ind, dist_fnc = euclidean_dist):
        if len(self) == 0:
            return None, 0
        else:
            closest_ind = None
            closest_dist = 10000
            for ind_other in self:
                dist = dist_fnc(ind, ind_other, self.bounds)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_ind = ind_other
            return (closest_ind, closest_dist)
        
    def closest_individual_from_vars(self, variables, dist_fnc = euclidean_dist):
        ind = Individual()
        ind.set("X", variables)
        return self.closest_individual_from_ind(ind, dist_fnc)

    def process_individual(self, candidate, dist_fnc = euclidean_dist):
        if len(self) == 0:
            self._int_add(candidate)
            log.debug('add initial individual')
        else:
            # uses semantic_distance to exploit behavioral information
            closest_archived, candidate_archived_distance = self.closest_individual_from_ind(candidate, dist_fnc=dist_fnc)
            # print(f"candidate_archived_distance is: {candidate_archived_distance}")
            if candidate_archived_distance > self.archive_threshold:
                log.debug('candidate is far from any archived individual')
                self._int_add(candidate)
                # print('added to archive')
            else:
                pass
                # print('closest archived is too close, dont add')

    def _int_add(self, candidate):
        self.append(candidate)
        # print('archive add', candidate)