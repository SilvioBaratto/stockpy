from ..base import ModelTrainer
from ._gru import _GRU
from ._bigru import _BiGRU
from ._bilstm import _BiLSTM
from ._lstm import _LSTM
from ._mlp import _MLP
from dataclasses import dataclass

@dataclass
class ModelArgs:
    input_size: int = 4
    hidden_size: int = 8
    rnn_size: int = 32
    output_size: int = 1
    num_layers: int = 2
    dropout: float = 0.2
    dropout: float = 0.2
    pretrained: bool = False
    l2: float = 0.01

class GRU(ModelTrainer):

    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(ModelArgs, key, value)

        super().__init__(model=_GRU(args=ModelArgs), **kwargs)
        
class BiGRU(ModelTrainer):

    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(ModelArgs, key, value)

        super().__init__(model=_BiGRU(args=ModelArgs), **kwargs)
        
class BiLSTM(ModelTrainer):

    def __init__(self,
                **kwargs
                ):
    
        for key, value in kwargs.items():
            setattr(ModelArgs, key, value)

        super().__init__(model=_BiLSTM(args=ModelArgs), **kwargs)
        
class LSTM(ModelTrainer):

    def __init__(self,
                **kwargs
                ):
    
        for key, value in kwargs.items():
            setattr(ModelArgs, key, value)

        super().__init__(model=_LSTM(args=ModelArgs), **kwargs)
        
class MLP(ModelTrainer):

    def __init__(self,
                **kwargs
                ):
    
        for key, value in kwargs.items():
            setattr(ModelArgs, key, value)

        super().__init__(model=_MLP(args=ModelArgs), **kwargs)