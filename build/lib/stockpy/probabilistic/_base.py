from ..base import Base
from ._bnn import BayesianNN as BNN
from ._ghmm import GaussianHMM as GHMM 
from ._dmm import DeepMarkovModel as DMM
from ..config import shared

class BayesianNN(Base):
    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(shared, key, value)

        super().__init__(model=BNN(), **kwargs)

class GaussianHMM(Base):
    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(shared, key, value)

        super().__init__(model=GHMM(), **kwargs)

class DeepMarkovModel(Base):
    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(shared, key, value)

        super().__init__(model=DMM(), **kwargs)
