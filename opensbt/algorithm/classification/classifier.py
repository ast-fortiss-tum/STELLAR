from enum import Enum

class ClassificationType(Enum):
    """ This enumeration holds all classification models applicable to derive critical regions.
    """
    NONE = 0,
    DT = 1,
    SVM = 2,
    KNN = 3
