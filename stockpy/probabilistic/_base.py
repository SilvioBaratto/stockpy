from ..base import ModelTrainer
from ._bnn import _BayesianNN
from ._ghmm import _GaussianHMM
from ._dmm import _DeepMarkovModel

from dataclasses import dataclass

@dataclass
class ModelArgs:
    input_size: int = 4
    hidden_size: int = 8
    rnn_dim: int = 32
    output_size: int = 1
    dropout: float = 0.2
    z_dim: int = 32
    emission_dim: int = 32
    transition_dim: int = 32
    variance: float = 0.1

class BayesianNN(ModelTrainer):

    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(ModelArgs, key, value)

        super().__init__(model=_BayesianNN(args=ModelArgs), **kwargs)

class GaussianHMM(ModelTrainer):
    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(ModelArgs, key, value)

        super().__init__(model=_GaussianHMM(args=ModelArgs), **kwargs)

class DeepMarkovModel(ModelTrainer):
    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(ModelArgs, key, value)

        super().__init__(model=_DeepMarkovModel(args=ModelArgs), **kwargs)