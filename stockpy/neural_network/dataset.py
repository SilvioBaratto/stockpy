import math
import numpy as np

import torch
from torch import nn as nn

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, 
                dataframe, 
                isValSet_bool=None, 
                sequence_length=30,
                test_size=0.2,
                ):

        len_test = len(dataframe) - int(len(dataframe) * test_size)

        self.dataframe = dataframe
        self.target= "Close"
        self.features = ['High', 'Low', 'Open', 'Volume']

        self.sequence_length = sequence_length
        self.mean, self.std = self.__normalize_mean_std()
        
        if isValSet_bool:
            dataframe = dataframe[len_test:]
            self.y = torch.tensor(dataframe[self.target].values).float()
            self.X = torch.tensor(dataframe[self.features].values).float()

        else:
            dataframe = dataframe[:len_test]
            self.y = torch.tensor(dataframe[self.target].values).float()
            self.X = torch.tensor(dataframe[self.features].values).float()


    def __normalize_mean_std(self):
        target_mean = self.dataframe[self.target].mean()
        target_stdev = self.dataframe[self.target].std()

        for i in self.dataframe.columns:
            mean = self.dataframe[i].mean()
            stdev = self.dataframe[i].std()

            self.dataframe[i] = np.divide((self.dataframe[i] - mean), stdev)
            self.dataframe[i] = np.divide((self.dataframe[i] - mean), stdev)

        return target_mean, target_stdev

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i] 
