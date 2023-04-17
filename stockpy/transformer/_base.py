from ..base import Base
from ..config import shared
from ._transformer import Transformer as _Transformer

class Transformer(Base):

    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(shared, key, value)

        super().__init__(model=_Transformer(), **kwargs)