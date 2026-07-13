
## Customization to apply to other problems

While pymoo can work with custom objects insted of numerical arrays, the interfaces pymoo provide for that are not convenient. OpenSBT provides base classes for operators to make work with custom classes less complicated. 

### Define your data model and problem

Data model should store everything you want to reppresent in your test individuals, and all the variables from the search space. Problem is responsible for representing the data space and defining how you map your search space variables into data space instances.

``` python
class Utterance(BaseModel):
    question: Optional[str] = None
    ...
```

```python
from pymoo.core.problem import Problem


class QAProblem(Problem):
    ...
```

### Implement operators

You might also want to implement base classes for the operators you use in your domain. Operators are Crossover, Mutation, Sampling, DuplicateElimination. The final classes should have cartain methods implemented. `_validate_instance` is a method that validates all the instances you return during your operator calls to avoid errors. The very basic implementation is type check for your model. Crossover operators should implement `_instance_crossover` method, that takes a list of lists of your custom class instances and treat it as list of sets of parents. For each list of parents it needs to create a list of offsprings. The output is the corresponding list of list of instances. There are similar methods for other operators.

```python
from opensbt.operators import CustomObjectCrossoverBase

class UtteranceOperatorBase(CustomObjectOperatorBase, ABC):
    """
    Base class for all utterance-based operations providing some methods 
    and implementing _validate_instance
    """
    @staticmethod
    def _validate_instance(obj):
        assert isinstance(obj, Utterance), "Population must be made of Utterance instances"


class UtteranceCrossoverBase(CustomObjectCrossoverBase, UtteranceOperatorBase, ABC):
    """
    Base class for utterance crossover with some extra helpful methods
    """
    def _style_features_crossover(...):
        ...

    def _content_features_crossover(...):
        ...

class UtteranceCrossover(UtteranceCrossoverBase):
    """
    Final crossover implementation
    """
    def __init__(
        self, crossover_rate=0.7, temperature=0.3, llm_type=LLMType(LLM_CROSSOVER)
    ):
        super().__init__(2, 2)
        self.crossover_rate = crossover_rate
        self.temperature = temperature

    def _instance_crossover(
        self, problem, matings: List[List[Utterance]]
    ) -> List[List[Utterance]]:
        result: List[List[Utterance]] = []
        for mating in matings:
            new_questions = llm_crossover(
                mating[0].question,
                mating[1].question,
                temperature=self.temperature,
                rate=self.crossover_rate,
            )
            result.append([Utterance(question=q) for q in new_questions])
        return result


```