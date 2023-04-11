from ..base import ModelTrainer
from ._bnn import BayesianNN as BNN
from ._ghmm import GaussianHMM as GHMM 
from ._dmm import DeepMarkovModel as DMM
from ..config import ModelArgs as args

class BayesianNN(ModelTrainer):
    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(args, key, value)

        super().__init__(model=BNN(), **kwargs)

class GaussianHMM(ModelTrainer):
    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(args, key, value)

        super().__init__(model=GHMM(), **kwargs)

class DeepMarkovModel(ModelTrainer):
    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(args, key, value)

        super().__init__(model=DMM(), **kwargs)
