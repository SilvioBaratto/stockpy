from ..base import Base
from ._bnn import BayesianNNRegressor as _BayesianNNRegressor
from ._ghmm import GaussianHMMRegressor as _GaussianHMMRegressor
from ._dmm import DeepMarkovModelRegressor as _DeepMarkovModelRegressor
from ._cnn import BayesianCNNRegressor as _BayesianCNNRegressor
from ..config import Config as cfg

class BayesianNNRegressor(Base):
    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.prob, key, value)

        super().__init__(model=_BayesianNNRegressor(), **kwargs)

class GaussianHMMRegressor(Base):
    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.prob, key, value)

        super().__init__(model=_GaussianHMMRegressor(), **kwargs)

class DeepMarkovModelRegressor(Base):
    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.prob, key, value)

        super().__init__(model=_DeepMarkovModelRegressor(), **kwargs)

class BayesianCNNRegressor(Base):
    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.prob, key, value)

        super().__init__(model=_BayesianCNNRegressor(), **kwargs)
