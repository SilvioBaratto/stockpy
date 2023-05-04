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

class GRURegressor(Base):

    def __init__(self,
                 **kwargs
                ):
        
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.nn, key, value)

        super().__init__(model=_GRURegressor(), **kwargs)

class GRUClassifier(Base):

    def __init__(self,
                 **kwargs
                ):
            
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.nn, key, value)

        super().__init__(model=_GRUClassifier(), **kwargs)

class BiGRURegressor(Base):
    
    def __init__(self,
                **kwargs
                ):
            
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.nn, key, value)
    
        super().__init__(model=_BiGRURegressor(), **kwargs)

class BiGRUClassifier(Base):
        
    def __init__(self,
                    **kwargs
                    ):
                
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.nn, key, value)
        
        super().__init__(model=_BiGRUClassifier(), **kwargs)

class BiLSTMRegressor(Base):
        
    def __init__(self,
                **kwargs
                ):
                
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.nn, key, value)
        
        super().__init__(model=_BiLSTMRegressor(), **kwargs)

class BiLSTMClassifier(Base):
            
    def __init__(self,
                **kwargs
                ):
                    
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.nn, key, value)
            
        super().__init__(model=_BiLSTMClassifier(), **kwargs)

class LSTMRegressor(Base):
                
    def __init__(self,
                **kwargs
                ):
                        
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.nn, key, value)
                
        super().__init__(model=_LSTMRegressor(), **kwargs)

class LSTMClassifier(Base):
                        
    def __init__(self,
                **kwargs
                ):
                                
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.nn, key, value)
                        
        super().__init__(model=_LSTMClassifier(), **kwargs) 

class MLPRegressor(Base):
                                
    def __init__(self,
                **kwargs
                ):
                                        
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.nn, key, value)
                                
        super().__init__(model=_MLPRegressor(), **kwargs)   

class MLPClassifier(Base):
                                        
    def __init__(self,
                **kwargs
                ):
                                                
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.nn, key, value)
                                        
        super().__init__(model=_MLPClassifier(), **kwargs)  
        

class CNNRegressor(Base):
                                
    def __init__(self,
                **kwargs
                ):
                                        
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.nn, key, value)
                                
        super().__init__(model=_CNNRegressor(), **kwargs)   

class CNNClassifier(Base):
                                        
    def __init__(self,
                **kwargs
                ):
                                                
        for key, value in kwargs.items():
            setattr(cfg.shared, key, value)
            setattr(cfg.nn, key, value)
                                        
        super().__init__(model=_CNNClassifier(), **kwargs)  
