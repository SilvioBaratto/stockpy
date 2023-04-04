from abc import ABC, abstractmethod

class ProbabilisticModel(ABC):
    """
    Interface for probabilistic models.
    """

    @abstractmethod
    def forward(self, x_data, y_data=None):
        """
        Computes the forward pass of the probabilistic model.
        """
        pass

    @property
    @abstractmethod
    def model_type(self):
        """
        Returns the type of the model (e.g. 'probabilistic').
        """
        return "probabilistic"

    @abstractmethod
    def model(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def guide(self, *args, **kwargs):
        pass
