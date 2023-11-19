from sklearn.exceptions import NotFittedError

class StockpyException(Exception):
    """
    Base exception for the Stockpy framework.

    All custom exceptions in the Stockpy framework should inherit from this class.
    """


class NotInitializedError(StockpyException, NotFittedError):
    """
    Exception raised when an operation is attempted on an uninitialized module.

    This exception is raised when attempting to use a module that requires
    initialization, such as by calling its `.initialize()` method or training
    the model with `.fit(...)`, but such initialization has not yet occurred.

    """


class StockpyAttributeError(StockpyException):
    """
    Exception for incorrect attribute assignment on a Stockpy net.

    This exception is raised when an attribute is set incorrectly on any
    object within the Stockpy framework, indicating a misuse or a potential
    error in code logic.

    """


class StockpyWarning(UserWarning):
    """
    Base warning for the Stockpy framework.

    This class is used for issuing warnings throughout the Stockpy framework, which
    are not critical enough to raise exceptions but are meant to inform the user
    of potential issues or misconfigurations.

    """


class DeviceWarning(StockpyWarning):
    """
    Warning related to device issues in the Stockpy framework.

    This warning is issued when there is a problem with a computational device,
    such as CUDA, indicating that the user should check the device configuration
    or availability.

    """


class StockpyTrainingImpossibleError(StockpyException):
    """
    Exception indicating that the net cannot be trained.

    This exception is raised when a net, for some reason, is deemed incapable of
    being trained. This could be due to an incorrect configuration, state, or
    other issues that prevent the training process from starting or completing.

    """


class LowProbabilityError(StockpyException):
    """
    Exception raised for low probability predictions in a Stockpy model.

    This exception may be raised when the prediction probabilities of a
    language model are below a certain threshold, indicating a lack of
    confidence in the predictions and potentially requiring review or adjustment
    of the model or its parameters.

    """
