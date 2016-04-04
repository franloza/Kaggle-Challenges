from enum import Enum

class AutoNumber(Enum):
     def __new__(cls):
         value = len(cls.__members__) + 1
         obj = object.__new__(cls)
         obj._value_ = value
         return obj

class Strategies(AutoNumber):
    LinearSVC = ()
    SVC = ()
    NuSVC = ()
    KNeighbors = ()
    RadiusNeighbors = ()
    Bagging = ()
    RandomForest = ()
    DecisionTree = ()
    XGB = ()
    GaussianBayes = ()
