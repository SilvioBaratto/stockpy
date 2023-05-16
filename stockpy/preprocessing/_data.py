import torch

class ZScoreNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data: torch.Tensor):
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)
        # Ensure that the standard deviation is not zero
        self.std = torch.where(self.std != 0, self.std, torch.ones_like(self.std))

    def transform(self, data: torch.Tensor):
        if self.mean is None or self.std is None:
            raise RuntimeError('Must fit normalizer before transforming data.')
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor):
        if self.mean is None or self.std is None:
            raise RuntimeError('Must fit normalizer before inverse transforming data.')
        return data * self.std + self.mean

class MinMaxNormalizer:
    def __init__(self):
        self.min = None
        self.max = None
    
    def fit(self, data: torch.Tensor):
        self.min = torch.min(data, dim=0)[0]
        self.max = torch.max(data, dim=0)[0]
        # Ensure that the maximum and minimum are not equal
        self.max = torch.where(self.max != self.min, self.max, self.min + torch.ones_like(self.max))

    def transform(self, data: torch.Tensor):
        if self.min is None or self.max is None:
            raise RuntimeError('Must fit normalizer before transforming data.')
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data: torch.Tensor):
        if self.min is None or self.max is None:
            raise RuntimeError('Must fit normalizer before inverse transforming data.')
        return data * (self.max - self.min) + self.min
    
class RobustScaler:
    def __init__(self):
        self.median = None
        self.iqr = None
    
    def fit(self, data: torch.Tensor):
        self.median = torch.median(data, dim=0).values
        self.iqr = torch.quantile(data, 0.75, dim=0) - torch.quantile(data, 0.25, dim=0)

    def transform(self, data: torch.Tensor):
        if self.median is None or self.iqr is None:
            raise RuntimeError('Must fit scaler before transforming data.')
        return (data - self.median) / self.iqr

    def inverse_transform(self, data: torch.Tensor):
        if self.median is None or self.iqr is None:
            raise RuntimeError('Must fit scaler before inverse transforming data.')
        return data * self.iqr + self.median
    
class MaxAbsScaler:
    def __init__(self):
        self.max_abs = None
    
    def fit(self, data: torch.Tensor):
        self.max_abs = torch.max(torch.abs(data), dim=0).values

    def transform(self, data: torch.Tensor):
        if self.max_abs is None:
            raise RuntimeError('Must fit scaler before transforming data.')
        return data / self.max_abs

    def inverse_transform(self, data: torch.Tensor):
        if self.max_abs is None:
            raise RuntimeError('Must fit scaler before inverse transforming data.')
        return data * self.max_abs