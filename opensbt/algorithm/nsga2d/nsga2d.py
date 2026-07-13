import random
import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.misc import find_duplicates, has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.core.population import Population
from pymoo.problems import get_problem
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.util.nds import efficient_non_dominated_sort
from pymoo.core.population import Population


from pymoo.core.population import Population
from pymoo.core.problem import Problem

from opensbt.algorithm.nsga2d.archive import SmartArchiveInput, euclidean_dist
from opensbt.algorithm.nsga2d.mating import MatingOpenSBT

# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------


def binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        rank_a, cd_a = pop[a].get("rank", "crowding")
        rank_b, cd_b = pop[b].get("rank", "crowding")

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class RankAndCrowdingSurvival(Survival):

    def __init__(self, nds=None) -> None:
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


# =========================================================================================================
# Implementation
# =========================================================================================================


class NSGA2D(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 survival=RankAndCrowdingSurvival(),
                 output=MultiObjectiveOutput(),
                 n_repopulate_max=0,
                 archive_threshold = 1,
                 eliminate_duplicates = True,
                 bounds_normalize = None,
                 dist_fnc = euclidean_dist,
                 **kwargs):
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            mating=MatingOpenSBT(selection,crossover,mutation, **kwargs),
            eliminate_duplicates=eliminate_duplicates,
            **kwargs)

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'
        self.n_repopulate_max = int(n_repopulate_max*pop_size) if type(n_repopulate_max) is float else n_repopulate_max 
        self.dist_fnc = dist_fnc

        if bounds_normalize is not None:
            self.archive_novelty = SmartArchiveInput(archive_threshold,
                                                     bounds=bounds_normalize
                                                    )    
        else:
            self.archive_novelty = SmartArchiveInput(archive_threshold)    
        
    def _initialize_infill(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        if self.advance_after_initial_infill:
            self.pop = self.survival.do(self.problem, infills, n_survive=len(infills), algorithm=self, **kwargs)

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]

    def _advance(self, infills=None, **kwargs):

        # the current population
        pop = self.pop
        if "algorithm" not in kwargs:
            kwargs["algorithm"] = self

        # print(f"[NSGA2D] pop shape before {pop.shape}")
        # print(f"pop: {pop.get('F')}")

        # repopulate
        if self.n_repopulate_max > 0:
            # replace k dominated solution by new individuals
            inds, indeces = calc_nondominated_individuals(self.pop)
            pop_sorted = get_individuals_rankwise(population=self.pop, number = len(self.pop))

            #################
            # print(f"non_dominated: {indeces}")
            
            #################
            # non_dominated = np.asarray(inds)          
            # print(f"self.n_repopulate_max: {self.n_repopulate_max}")
            
            n_repopulate = random.randint(1, self.n_repopulate_max)
            # n_dominated = len(self.pop) - len(non_dominated)

            print(f"n_repopulate: {n_repopulate}")
            # print(f"n_dominated: {n_dominated}")

            pop_sampled = self.initialization.sampling.do(self.problem,n_repopulate, **kwargs)
            evaluate_individuals(population=pop_sampled, problem=self.problem, **kwargs)
            
            # Add to archive of all evaluated
            self.archive = self.archive.add(pop_sampled)

            # print("[NSGA2D] new sampled individuals evaluated!")
            # print(f"[NSGA2D] fitness of new individuals: {pop_sampled.get('F')}")

            for i in range(1, n_repopulate + 1):
                pop_sorted[-i] = pop_sampled[-i]
            
            self.pop = pop_sorted

            for ind in self.pop:
                assert ind.get("F") is not None
            assert f"Current pop size is {len(self.pop)}: len(self.pop) == self.pop_size", len(self.pop) == self.pop_size

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)
        
        # update novelty archive
        # print(infills.get("F"))
        for ind in pop:
            self.archive_novelty.process_individual(ind, dist_fnc = self.dist_fnc)

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self)
        # we update the archive of optimal solutions
        # self.archive_optimal =  get_nondominated_population(Population.merge(self.archive_optimal,
        #                                                                     self.pop))

def calc_nondominated_individuals(population: Population):
    F = population.get("F")
    best_inds_index = efficient_non_dominated_sort.efficient_non_dominated_sort(F)[0]
    best_inds = [population[i] for i in best_inds_index]
    return best_inds, best_inds_index

def get_nondominated_population(population: Population):
    return Population(individuals=calc_nondominated_individuals(population)[0])

def get_individuals_rankwise(population, number):
    ranks_pop = efficient_non_dominated_sort.efficient_non_dominated_sort(population.get("F"))
    # take individuals rankwise until limit reached
    inds_to_add = []
    for i in range(0,len(ranks_pop)):
        if len(inds_to_add) < number:
            remaining = number - len(inds_to_add)
            num_next_front = min(remaining,len(ranks_pop[i]))
            inds_to_add = np.concatenate([inds_to_add,ranks_pop[i][0:num_next_front]])
        else:
            break
    pop = Population(individuals = [population[int(i)] for i in inds_to_add])
    assert len(pop) == number
    return pop

def calc_crowding_distance(F, filter_out_duplicates=True):
    n_points, n_obj = F.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)

    else:

        if filter_out_duplicates:
            # filter out solutions which are duplicates - duplicates get a zero finally
            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]
        else:
            # set every point to be unique without checking it
            is_unique = np.arange(n_points)

        # index the unique points of the array
        _F = F[is_unique]

        # sort each column and get index
        I = np.argsort(_F, axis=0, kind='mergesort')

        # sort the objective space values for the whole matrix
        _F = _F[I, np.arange(n_obj)]

        # calculate the distance from each point to the last and next
        dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), _F])

        # calculate the norm for each objective - set to NaN if all values are equal
        norm = np.max(_F, axis=0) - np.min(_F, axis=0)
        norm[norm == 0] = np.nan

        # prepare the distance to last and next vectors
        dist_to_last, dist_to_next = dist, np.copy(dist)
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divide by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        _cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        # save the final vector which sets the crowding distance for duplicates to zero to be eliminated
        crowding = np.zeros(n_points)
        crowding[is_unique] = _cd

    # crowding[np.isinf(crowding)] = 1e+14
    return crowding

def evaluate_individuals(population: Population, problem: Problem, **kwargs):
    out_all = {}
    problem._evaluate(population.get("X"), out_all, **kwargs)
    for index, ind in enumerate(population):
        dict_individual = {}
        for item,value in out_all.items():
            dict_individual[item] = value[index]
        #log.info(f"setting evaluation result {dict_individual}")
        ind.set_by_dict(**dict_individual)
    return population

parse_doc_string(NSGA2D.__init__)

if __name__ == "__main__":
    algorithm = NSGA2D(
        pop_size=10,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        eliminate_duplicates=True,
        n_repopulate_max = .3)
    
    termination = get_termination("n_gen", 12)

    save_history = True

    problem = get_problem('bnh')
    res = minimize(problem,
                    algorithm,
                    termination,
                    save_history=save_history,
                    verbose=True)
    
    history = res.history
    print(f"num optimal: {len(res.opt)}")
    # evaluation with and without repop