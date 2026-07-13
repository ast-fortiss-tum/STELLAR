from opensbt.experiment.experiment import Experiment
import logging as log

class Singleton(object):
    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it

    def init(self, *args, **kwds):
        pass
    
class DefaultExperiments(Singleton):
    """ This class allows to store and load experiments globally in OpenSBT. 
    """

    def init(self):
        """Initializes the store.
        """
        self.store = {}

    def register(self, exp: Experiment):
        """Registers an experiment in the store.

        :param exp: The experiment to be registered.
        :type exp: Experiment
        :return: Output 1 if registration failed, otherwise 0.
        :rtype: int
        """
        if not exp.name in self.store:
            self.store[exp.name] = exp
            return 0
        else:
            log.info("Experiment with the given name is already registered.")
            return 1

    def load(self, experiment_name: str) -> Experiment:
        """Loads an experiment based on the experiment name from the store.

        :param experiment_name: The experiment name.
        :type experiment_name: str
        :return: Returns the experiment.
        :rtype: Experiment
        """
        if experiment_name in self.store:
            return self.store[experiment_name]
        else:
            log.info("Experiment with the given name does not exist.")
            return 1

    def get_store(self):
        """Outputs the store.

        :return: Return the store.
        :rtype: Dict
        """
        return self.store

experiments_store = DefaultExperiments()