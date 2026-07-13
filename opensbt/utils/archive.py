from pymoo.util.archive import Archive
from pymoo.core.population import merge

# Stores all individuals added to the archive
class MemoryArchive(Archive):
    def __new__(cls,
                **kwargs):
        kwargs["duplicate_elimination"] = None
        return super().__new__(cls,
                                **kwargs)

    def _find_opt(self, sols):
        # do nothing
        return sols
    
    def add(self, sols):         
        sols = merge(self, sols)

        opt = self._find_opt(sols)

        cls = self.__class__
        obj = cls.__new__(cls, individuals=opt, **self.view(Archive).__dict__)
     
        return obj
