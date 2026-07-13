import copy
import dataclasses
from typing import List, Union, final, Any
from abc import ABC, abstractmethod

import numpy as np
import pydantic
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.individual import Individual
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair


@dataclasses.dataclass
class CustomObjectOperatorBase(ABC):
    @staticmethod
    def _transform_problem(problem: Problem, individual: Individual) -> Problem:
        problem = copy.deepcopy(problem)
        problem.n_var = len(individual.X)
        problem.xl = np.zeros(problem.n_var)
        problem.xu = np.ones(problem.n_var)
        return problem

    @staticmethod
    @abstractmethod
    def _validate_instance(obj):
        pass


class CustomObjectCrossoverBase(Crossover, CustomObjectOperatorBase, ABC):
    @final
    def _do(self, problem, X, **kwargs):
        # X shape: (n_parents, n_matings, n_var)
        _, n_matings, _ = X.shape
        matings: List[List[Any]] = []
        for i in range(n_matings):
            instances: List[Any] = []
            for j in range(self.n_parents):
                instance = X[j, i][0]
                self._validate_instance(instance)
                instances.append(instance)
            matings.append(instances)
        offsprings_list = self._instance_crossover(problem, matings)
        assert len(offsprings_list) == n_matings, (
            "Crossover should return 1 group of offsprings for each mating.\n"
            f"Expected: {n_matings}. Got: {len(offsprings_list)}"
        )
        Y = np.full((self.n_offsprings, n_matings, 1), None, dtype=object)
        for i, offsprings in enumerate(offsprings_list):
            assert len(offsprings) == self.n_offsprings, (
                f"Crossover should return {self.n_offsprings} "
                f"for each mating. Got {len(offsprings)} for the {i}th mating"
            )
            for j, offspring in enumerate(offsprings):
                self._validate_instance(offspring)
                Y[j, i, 0] = offspring
        return Y

    def _build_vars_matings(
        self,
        instance_matings: List[List[Any]],
        attribute_name: str = "style_scores",
    ) -> List[List[Individual]]:
        parents_vars = []
        for mating in instance_matings:
            mating_vars = []
            for parent in mating:
                vars = getattr(parent, attribute_name)
                if vars is None:
                    vars = []
                mating_vars.append(Individual(X=vars))
            parents_vars.append(mating_vars)
        return parents_vars

    def _vars_crossover(
        self,
        problem: Problem,
        matings: List[List[Any]],
        crossover: Crossover,
        attribute_name: str,
    ) -> List[List[List[Union[int, float]]]]:
        n_matings = len(matings)
        result: List[List[np.ndarray]] = []
        for _ in range(n_matings):
            result.append([])

        parents_vars = self._build_vars_matings(matings, attribute_name)
        for _ in range(self.n_offsprings):
            offspring = crossover.do(self._transform_problem(problem, parents_vars[0][0]), parents_vars)
            X_vars = offspring.get("X")
            for j in range(n_matings):
                result[j].append(X_vars[j].tolist())
        return result
    
    def _empty_crossover(self, n_matings: int) -> List[List[List]]:
        result: List[List] = []
        for _ in range(n_matings):
            result.append([])
            for _ in range(self.n_offsprings):
                result[-1].append([])
        return result

    @abstractmethod
    def _instance_crossover(
        self, problem, matings: List[List[Any]]
    ) -> List[List[Any]]:
        pass


class CustomObjectMutationBase(Mutation, CustomObjectOperatorBase, ABC):
    @final
    def _do(self, problem, X, **kwargs):
        n_instances = X.shape[0]
        input_instances = [X[i][0] for i in range(n_instances)]
        for u in input_instances:
            self._validate_instance(u)
        output_instances = self._instance_mutation(problem, input_instances)
        assert (
            len(output_instances) == n_instances
        ), f"The length of output {len(output_instances)} is not the same as of input {n_instances}"
        for u in output_instances:
            self._validate_instance(u)
        Y = np.full(len(X), None, dtype=object)
        for i, instance in enumerate(output_instances):
            Y[i] = np.asarray([instance])
        return Y

    def _vars_mutation(
        self,
        problem: Problem,
        instances: List[Any],
        mutation: Mutation,
        attribute_name: str,
    ) -> List[List[Union[float, int]]]:
        vars_before = [getattr(u, attribute_name) for u in instances]
        population = Population.new(X=vars_before)
        offspring_vars = mutation.do(self._transform_problem(problem, population[0]), population)
        vars_new = offspring_vars.get("X").tolist()
        vars_new = [[round(v, 2) if isinstance(v, float) else v for v in vec] for vec in vars_new]
        return vars_new
    
    def _empty_mutation(self, n_instances) -> List[List]:
        return [[] for _ in range(n_instances)]

    @abstractmethod
    def _instance_mutation(
        self, problem: Problem, instances: List[Any]
    ) -> List[Any]:
        pass


class CustomObjectSamplingBase(Sampling, CustomObjectOperatorBase, ABC):
    @final
    def _do(self, problem: Problem, n_samples: int, **kwargs):
        X = np.full(n_samples, problem.n_var, dtype=object)
        instances = self._sample_instances(problem, n_samples, **kwargs)
        assert len(instances) == n_samples, (
            "Wrong number of samples" f"Expected: {n_samples}. Got: {len(instances)}"
        )
        for i, instance in enumerate(instances):
            self._validate_instance(instance)
            X[i] = np.asarray([instance])
        return X

    @abstractmethod
    def _sample_instances(
        self, problem: Problem, n_samples: int, **kwargs
    ) -> List[Any]:
        pass


class CustomObjectDuplicateEliminationBase(
    ElementwiseDuplicateElimination, CustomObjectOperatorBase, ABC
):
    @final
    def is_equal(self, a, b):
        instance_a = a.get("X")[0]
        instance_b = b.get("X")[0]
        self._validate_instance(instance_a)
        self._validate_instance(instance_b)
        return self._instances_equal(instance_a, instance_b)

    @abstractmethod
    def _instances_equal(self, a: Any, b: Any) -> bool:
        pass


class CustomObjectRepairBase(
    Repair, CustomObjectOperatorBase, ABC
):
    def _do(self, problem, X, **kwargs):
        result = []
        for i in range(X.shape[0]):
            instance = X[i][0]
            self._validate_instance(instance)
            repaired = self._repair_instance(problem, instance, **kwargs)
            self._validate_instance(repaired)
            result.append(np.array([repaired]))
        return np.array(result)
    
    @abstractmethod
    def _repair_instance(self, problem, instance, **kwargs):
        pass
