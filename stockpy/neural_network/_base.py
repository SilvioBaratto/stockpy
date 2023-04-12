from ..base import Base
from ._gru import GRU as _GRU
from ._bigru import BiGRU as _BiGRU
from ._bilstm import BiLSTM as _BiLSTM
from ._lstm import LSTM as _LSTM
from ._mlp import MLP as _MLP
from ..config import shared

class GRU(Base):

    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(shared, key, value)

        super().__init__(model=_GRU(), **kwargs)
        
class BiGRU(Base):

    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(shared, key, value)

        super().__init__(model=_BiGRU(), **kwargs)
        
class BiLSTM(Base):

    def __init__(self,
                **kwargs
                ):
    
        for key, value in kwargs.items():
            setattr(shared, key, value)

        super().__init__(model=_BiLSTM(), **kwargs)
        
class LSTM(Base):

    def __init__(self,
                **kwargs
                ):
    
        for key, value in kwargs.items():
            setattr(shared, key, value)

        super().__init__(model=_LSTM(), **kwargs)
        
class MLP(Base):

    def __init__(self,
                **kwargs
                ):
    
        for key, value in kwargs.items():
            setattr(shared, key, value)

        super().__init__(model=_MLP(), **kwargs)
