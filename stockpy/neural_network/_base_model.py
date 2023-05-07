import os
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def model_type(self) -> str:
        return os.path.basename(os.path.dirname(__file__))

    @property
    def name(self):
        return self.__class__.__name__
        
    def to(self, device: torch.device) -> None:
        """
        Moves the model to the specified device.

        :param device: The device to move the model to.
        :type device: torch.device
        """
        super().to(device)

class BaseRegressorRNN(BaseModel):
    def __init__(self):
        super().__init__()

class BaseRegressorFFNN(BaseModel):
    def __init__(self):
        super().__init__()

class BaseRegressorCNN(BaseModel):
    def __init__(self):
        super().__init__()

class BaseClassifierRNN(BaseModel):
    def __init__(self):
        super().__init__()

class BaseClassifierFFNN(BaseModel):
    def __init__(self):
        super().__init__()

class BaseClassifierCNN(BaseModel):
    def __init__(self):
        super().__init__()
