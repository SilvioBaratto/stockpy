from abc import ABC, abstractmethod

class NeuralNetwork(ABC):
    """
    Interface for probabilistic models.
    """

    @abstractmethod
    def forward(self, x):
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
        return "neural_network"