from ..base import Base
from ._gru import GRURegressor as _GRURegressor
from ._gru import GRUClassifier as _GRUClassifier
from ._bigru import BiGRURegressor as _BiGRURegressor
from ._bigru import BiGRUClassifier as _BiGRUClassifier
from ._bilstm import BiLSTMRegressor as _BiLSTMRegressor
from ._bilstm import BiLSTMClassifier as _BiLSTMClassifier
from ._lstm import LSTMRegressor as _LSTMRegressor
from ._lstm import LSTMClassifier as _LSTMClassifier
from ._mlp import MLPRegressor as _MLPRegressor
from ._mlp import MLPClassifier as _MLPClassifier
from ._cnn import CNNRegressor as _CNNRegressor
from ._cnn import CNNClassifier as _CNNClassifier
from ..config import Config as cfg

class BaseModel(Base):
    def __init__(self, category, model_class):
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
    def __init__(self, category):
        super().__init__(category, model_class="rnn")

class BaseFFNN(BaseModel):
    def __init__(self, category):
        super().__init__(category, model_class="ffnn")

class BaseCNN(BaseModel):
    def __init__(self, category):
        super().__init__(category, model_class="cnn")

class GRURegressor(BaseRNN):
    def __init__(self, **kwargs):
        super().__init__(category="regressor", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _GRURegressor(input_size=input_size, 
                             output_size=output_size)

class GRUClassifier(BaseRNN):
    def __init__(self, **kwargs):
        super().__init__(category="classifier", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _GRUClassifier(input_size=input_size, 
                              output_size=output_size)
    
class BiGRURegressor(BaseRNN):
    def __init__(self, **kwargs):
        super().__init__(category="regressor", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _BiGRURegressor(input_size=input_size, 
                               output_size=output_size)
    
class BiGRUClassifier(BaseRNN):
    def __init__(self, **kwargs):
        super().__init__(category="classifier", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _BiGRUClassifier(input_size=input_size, 
                                output_size=output_size)

class BiLSTMRegressor(BaseRNN):
    def __init__(self, **kwargs):
        super().__init__(category="regressor", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _BiLSTMRegressor(input_size=input_size, 
                                output_size=output_size)

class BiLSTMClassifier(BaseRNN):
    def __init__(self, **kwargs):
        super().__init__(category="classifier", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _BiLSTMClassifier(input_size=input_size, 
                                 output_size=output_size)

class LSTMRegressor(BaseRNN):
    def __init__(self, **kwargs):
        super().__init__(category="regressor", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _LSTMRegressor(input_size=input_size, 
                              output_size=output_size)

class LSTMClassifier(BaseRNN):
    def __init__(self, **kwargs):
        super().__init__(category="classifier", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _LSTMClassifier(input_size=input_size, 
                               output_size=output_size)

class MLPRegressor(BaseFFNN):
    def __init__(self, **kwargs):
        super().__init__(category="regressor", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _MLPRegressor(input_size=input_size, 
                             output_size=output_size)

class MLPClassifier(BaseFFNN):
    def __init__(self, **kwargs):
        super().__init__(category="classifier", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _MLPClassifier(input_size=input_size, 
                              output_size=output_size)
        
class CNNRegressor(BaseCNN):
    def __init__(self, **kwargs):
        super().__init__(category="regressor", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _CNNRegressor(input_size=input_size, 
                             output_size=output_size)  
    
class CNNClassifier(BaseCNN):
    def __init__(self, **kwargs):
        super().__init__(category="classifier", **kwargs)

    def _create_model(self, input_size: int, output_size: int):
        return _CNNClassifier(input_size=input_size, 
                              output_size=output_size) 