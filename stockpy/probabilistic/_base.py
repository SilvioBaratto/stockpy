from ..base import Base
from ._bnn import BayesianNNRegressor as _BayesianNNRegressor
from ._bnn import BayesianNNClassifier as _BayesianNNClassifier
from ._ghmm import GaussianHMMRegressor as _GaussianHMMRegressor
from ._dmm import DeepMarkovModelRegressor as _DeepMarkovModelRegressor
from ._cnn import BayesianCNNRegressor as _BayesianCNNRegressor
from ._cnn import BayesianCNNClassifier as _BayesianCNNClassifier
from ..config import Config as cfg

class BaseModel(Base):
    def __init__(self, category, model_class, **kwargs):
        super().__init__()
        self._category = category
        self._model_class = model_class
  
    @property
    def category(self) -> str:
        return self._category

    @property
    def model_class(self) -> str:
        return self._model_class

class BaseRNN(BaseModel):
    def __init__(self, category, **kwargs):
        super().__init__(category, model_class="rnn", **kwargs)

class BaseFFNN(BaseModel):
    def __init__(self, category, **kwargs):
        super().__init__(category, model_class="ffnn", **kwargs)

class BaseCNN(BaseModel):
    def __init__(self, category, **kwargs):
        super().__init__(category, model_class="cnn", **kwargs)

class BayesianNNRegressor(BaseFFNN):
    def __init__(self, **kwargs):
        super().__init__(category="regressor", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _BayesianNNRegressor(input_size=input_size, 
                             output_size=output_size)
    
class BayesianNNClassifier(BaseFFNN):
    def __init__(self, **kwargs):
        super().__init__(category="classifier", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _BayesianNNClassifier(input_size=input_size, 
                             output_size=output_size)
    
class GaussianHMMRegressor(BaseRNN):
    def __init__(self, **kwargs):
        super().__init__(category="regressor", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _GaussianHMMRegressor(input_size=input_size, 
                             output_size=output_size)

class DeepMarkovModelRegressor(BaseRNN):
    def __init__(self, **kwargs):
        super().__init__(category="regressor", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _DeepMarkovModelRegressor(input_size=input_size, 
                             output_size=output_size)

class BayesianCNNRegressor(BaseCNN):
    def __init__(self, **kwargs):
        super().__init__(category="regressor", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _BayesianCNNRegressor(input_size=input_size, 
                             output_size=output_size)
    
class BayesianCNNClassifier(BaseCNN):
    def __init__(self, **kwargs):
        super().__init__(category="classifier", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _BayesianCNNClassifier(input_size=input_size, 
                             output_size=output_size)
