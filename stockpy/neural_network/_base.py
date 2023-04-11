from ..base import ModelTrainer
from ._gru import _GRU
from ._bigru import _BiGRU
from ._bilstm import _BiLSTM
from ._lstm import _LSTM
from ._mlp import _MLP
from ..config import ModelArgs as args

class GRU(ModelTrainer):

    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(args, key, value)

        super().__init__(model=_GRU(), **kwargs)
        
class BiGRU(ModelTrainer):

    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(args, key, value)

        super().__init__(model=_BiGRU(), **kwargs)
        
class BiLSTM(ModelTrainer):

    def __init__(self,
                **kwargs
                ):
    
        for key, value in kwargs.items():
            setattr(args, key, value)

        super().__init__(model=_BiLSTM(), **kwargs)
        
class LSTM(ModelTrainer):

    def __init__(self,
                **kwargs
                ):
    
        for key, value in kwargs.items():
            setattr(args, key, value)

        super().__init__(model=_LSTM(), **kwargs)
        
class MLP(ModelTrainer):

    def __init__(self,
                **kwargs
                ):
    
        for key, value in kwargs.items():
            setattr(args, key, value)

        super().__init__(model=_MLP(), **kwargs)
