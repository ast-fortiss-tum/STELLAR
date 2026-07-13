from enum import Enum

class AlgorithmType(Enum):
    """ Defines all available algorithms to be triggered via number in -a flag.
    """
    NSGAII = 1
    PSO = 2 
    PS_RAND= 3 
    PS_GRID = 4 
    PS_FPS = 5
    NSGAII_DT = 6
    NSGAIID = 7
