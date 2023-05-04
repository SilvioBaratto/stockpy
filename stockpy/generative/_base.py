from ..base import Base
from ._lstmseq2seq import LSTMSeq2Seq as _LSTM
from ..config import Config as cfg
        
class LSTM(Base):

    def __init__(self,
                **kwargs
                ):
    
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)

        super().__init__(model=_LSTM(), **kwargs)
        

