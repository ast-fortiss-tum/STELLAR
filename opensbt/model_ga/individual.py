from pymoo.core.individual import Individual

class IndividualSimulated(Individual):
    """This class extends pymoos' Individual class to integrate simulation output data.
    """
    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config,**kwargs)
        self._SO = None
        self._CB = None

    def reset(self,data=True):
        super().reset(data=data)
        self._SO = None
        self._CB = None

    @property
    def SO(self):
        return self._SO

    @SO.setter
    def SO(self, value):
        self._SO = value

    @property
    def CB(self):
        return self._CB

    @CB.setter
    def CB(self, value):
        self._CB = value
        
    @property
    def cb(self):
        return self.CB

    @property
    def so(self):
        return self.SO
