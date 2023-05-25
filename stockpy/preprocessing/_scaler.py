from abc import abstractmethod, ABCMeta
import torch

class TransformMixin(metaclass=ABCMeta):
    """
    Abstract base class for transformation mixins.
    
    This class defines the interface for transformation mixins, which are classes that can be used to normalize or 
    denormalize data. The transformations that a subclass must implement are defined as abstract methods.
    """

    @abstractmethod
    def __init__(self):
        """
        Initializes the TransformMixin instance.
        """
        pass

    @abstractmethod
    def fit(self, data: torch.Tensor):
        """
        Fits the transformation to the data.
        
        This method computes and stores any necessary statistics from the data needed to perform the transformation.

        Args:
            data (torch.Tensor): The data to fit the transformation to.
        """
        pass

    @abstractmethod
    def transform(self, data: torch.Tensor):
        """
        Applies the transformation to the data.

        This method transforms the data based on the statistics computed in the fit method. It should be called after 
        the fit method.

        Args:
            data (torch.Tensor): The data to apply the transformation to.

        Returns:
            torch.Tensor: The transformed data.
        """
        pass

    @abstractmethod
    def inverse_transform(self, data: torch.Tensor):
        """
        Applies the inverse transformation to the data.

        This method undoes the transformation applied in the transform method, returning the data to its original state.
        It should be called after the transform method.

        Args:
            data (torch.Tensor): The data to apply the inverse transformation to.

        Returns:
            torch.Tensor: The inverse-transformed data.
        """
        pass

class ZScoreNormalizer(TransformMixin):
    """
    Z-Score Normalizer.
    
    This class normalizes and denormalizes data using Z-score normalization, which scales the data to have a mean of 0 
    and a standard deviation of 1. It inherits from the TransformMixin class.

    Attributes:
        mean (torch.Tensor or None): The mean of the data. Calculated when fit method is called.
        std (torch.Tensor or None): The standard deviation of the data. Calculated when fit method is called.
    """

    def __init__(self):
        """
        Initializes the ZScoreNormalizer instance.

        The mean and standard deviation are initialized to None. They will be computed when fit method is called.
        """
        self.mean = None
        self.std = None

    def fit(self, data: torch.Tensor):
        """
        Fits the normalizer to the data.

        This method computes and stores the mean and standard deviation of the data. 

        Args:
            data (torch.Tensor): The data to fit the normalizer to.
        """
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)
        # Ensure that the standard deviation is not zero
        self.std = torch.where(self.std != 0, self.std, torch.ones_like(self.std))

    def transform(self, data: torch.Tensor):
        """
        Applies the normalizer to the data.

        This method transforms the data by subtracting the mean and dividing by the standard deviation. The fit method 
        must be called before this method.

        Args:
            data (torch.Tensor): The data to apply the normalizer to.

        Returns:
            torch.Tensor: The normalized data.

        Raises:
            RuntimeError: If the fit method has not been called before this method.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError('Must fit normalizer before transforming data.')
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor):
        """
        Applies the inverse normalizer to the data.

        This method transforms the data by multiplying by the standard deviation and adding the mean. The fit and 
        transform methods must be called before this method.

        Args:
            data (torch.Tensor): The data to apply the inverse normalizer to.

        Returns:
            torch.Tensor: The denormalized data.

        Raises:
            RuntimeError: If the fit method has not been called before this method.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError('Must fit normalizer before inverse transforming data.')
        return data * self.std + self.mean

class MinMaxNormalizer(TransformMixin):
    """
    Min-Max Normalizer.
    
    This class normalizes and denormalizes data using Min-Max normalization, which scales the data to fit within a 
    specified range. It inherits from the TransformMixin class.

    Attributes:
        min (torch.Tensor or None): The minimum value of the data. Calculated when fit method is called.
        max (torch.Tensor or None): The maximum value of the data. Calculated when fit method is called.
    """

    def __init__(self):
        """
        Initializes the MinMaxNormalizer instance.

        The minimum and maximum values are initialized to None. They will be computed when fit method is called.
        """
        self.min = None
        self.max = None

    def fit(self, data: torch.Tensor):
        """
        Fits the normalizer to the data.

        This method computes and stores the minimum and maximum values of the data. 

        Args:
            data (torch.Tensor): The data to fit the normalizer to.
        """
        self.min = torch.min(data, dim=0)[0]
        self.max = torch.max(data, dim=0)[0]
        # Ensure that the maximum and minimum are not equal
        self.max = torch.where(self.max != self.min, self.max, self.min + torch.ones_like(self.max))

    def transform(self, data: torch.Tensor):
        """
        Applies the normalizer to the data.

        This method transforms the data by subtracting the minimum and dividing by the range of the data. The fit 
        method must be called before this method.

        Args:
            data (torch.Tensor): The data to apply the normalizer to.

        Returns:
            torch.Tensor: The normalized data.

        Raises:
            RuntimeError: If the fit method has not been called before this method.
        """
        if self.min is None or self.max is None:
            raise RuntimeError('Must fit normalizer before transforming data.')
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data: torch.Tensor):
        """
        Applies the inverse normalizer to the data.

        This method transforms the data by multiplying by the range of the data and adding the minimum. The fit and 
        transform methods must be called before this method.

        Args:
            data (torch.Tensor): The data to apply the inverse normalizer to.

        Returns:
            torch.Tensor: The denormalized data.

        Raises:
            RuntimeError: If the fit method has not been called before this method.
        """
        if self.min is None or self.max is None:
            raise RuntimeError('Must fit normalizer before inverse transforming data.')
        return data * (self.max - self.min) + self.min


class RobustScaler(TransformMixin):
    """
    Robust Scaler.
    
    This class normalizes and denormalizes data using Robust Scaling, which scales data using statistics that are 
    robust to outliers. It inherits from the TransformMixin class.

    Attributes:
        median (torch.Tensor or None): The median of the data. Calculated when fit method is called.
        iqr (torch.Tensor or None): The interquartile range of the data. Calculated when fit method is called.
    """

    def __init__(self):
        """
        Initializes the RobustScaler instance.

        The median and interquartile range are initialized to None. They will be computed when fit method is called.
        """
        self.median = None
        self.iqr = None

    def fit(self, data: torch.Tensor):
        """
        Fits the scaler to the data.

        This method computes and stores the median and interquartile range of the data.

        Args:
            data (torch.Tensor): The data to fit the scaler to.
        """
        self.median = torch.median(data, dim=0).values
        self.iqr = torch.quantile(data, 0.75, dim=0) - torch.quantile(data, 0.25, dim=0)

    def transform(self, data: torch.Tensor):
        """
        Applies the scaler to the data.

        This method transforms the data by subtracting the median and dividing by the interquartile range. The fit 
        method must be called before this method.

        Args:
            data (torch.Tensor): The data to apply the scaler to.

        Returns:
            torch.Tensor: The scaled data.

        Raises:
            RuntimeError: If the fit method has not been called before this method.
        """
        if self.median is None or self.iqr is None:
            raise RuntimeError('Must fit scaler before transforming data.')
        return (data - self.median) / self.iqr

    def inverse_transform(self, data: torch.Tensor):
        """
        Applies the inverse scaler to the data.

        This method transforms the data by multiplying by the interquartile range and adding the median. The fit and 
        transform methods must be called before this method.

        Args:
            data (torch.Tensor): The data to apply the inverse scaler to.

        Returns:
            torch.Tensor: The rescaled data.

        Raises:
            RuntimeError: If the fit method has not been called before this method.
        """
        if self.median is None or self.iqr is None:
            raise RuntimeError('Must fit scaler before inverse transforming data.')
        return data * self.iqr + self.median
    
class MaxAbsScaler(TransformMixin):
    """
    Max Absolute Scaler.
    
    This class normalizes and denormalizes data using Max Absolute Scaling, which scales data to lie within the range 
    [-1,1] by dividing each sample by its maximum absolute value. It inherits from the TransformMixin class.

    Attributes:
        max_abs (torch.Tensor or None): The maximum absolute value of the data. Calculated when fit method is called.
    """

    def __init__(self):
        """
        Initializes the MaxAbsScaler instance.

        The maximum absolute value is initialized to None. It will be computed when fit method is called.
        """
        self.max_abs = None

    def fit(self, data: torch.Tensor):
        """
        Fits the scaler to the data.

        This method computes and stores the maximum absolute value of the data.

        Args:
            data (torch.Tensor): The data to fit the scaler to.
        """
        self.max_abs = torch.max(torch.abs(data), dim=0).values

    def transform(self, data: torch.Tensor):
        """
        Applies the scaler to the data.

        This method transforms the data by dividing by the maximum absolute value. The fit method must be called before 
        this method.

        Args:
            data (torch.Tensor): The data to apply the scaler to.

        Returns:
            torch.Tensor: The scaled data.

        Raises:
            RuntimeError: If the fit method has not been called before this method.
        """
        if self.max_abs is None:
            raise RuntimeError('Must fit scaler before transforming data.')
        return data / self.max_abs

    def inverse_transform(self, data: torch.Tensor):
        """
        Applies the inverse scaler to the data.

        This method transforms the data by multiplying by the maximum absolute value. The fit and transform methods must 
        be called before this method.

        Args:
            data (torch.Tensor): The data to apply the inverse scaler to.

        Returns:
            torch.Tensor: The rescaled data.

        Raises:
            RuntimeError: If the fit method has not been called before this method.
        """
        if self.max_abs is None:
            raise RuntimeError('Must fit scaler before inverse transforming data.')
        return data * self.max_abs