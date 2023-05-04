import os
import os
import torch
from pyro.nn import PyroModule

class BaseModel(PyroModule):
    def __init__(self, category, model_class):
        super().__init__()
        self._category = category
        self._model_class = model_class

    @property
    def model_type(self) -> str:
        return os.path.basename(os.path.dirname(__file__))

    @property
    def category(self) -> str:
        return self._category

    @property
    def name(self):
        return self.__class__.__name__
    
    @property
    def model_class(self):
        return self._model_class
    
    def to(self, device: torch.device) -> None:
        """
        Moves the model to the specified device.

        :param device: The device to move the model to.
        :type device: torch.device
        """
        super().to(device)

class BaseRegressorRNN(BaseModel):
    def __init__(self):
        super().__init__(category="regressor", model_class="rnn")

class BaseRegressorFFNN(BaseModel):
    def __init__(self):
        super().__init__(category="regressor", model_class="ffnn")

class BaseRegressorCNN(BaseModel):
    def __init__(self):
        super().__init__(category="regressor", model_class="cnn")

class BaseClassifierRNN(BaseModel):
    def __init__(self):
        super().__init__(category="classifier", model_class="rnn")

class BaseClassifierFFNN(BaseModel):
    def __init__(self):
        super().__init__(category="classifier", model_class="ffnn")

class BaseClassifierCNN(BaseModel):
    def __init__(self):
        super().__init__(category="classifier", model_class="cnn")