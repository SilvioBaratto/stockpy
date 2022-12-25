import math
import numpy as np
import pandas as pd

import torch
from torch import nn as nn
from sklearn.preprocessing import StandardScaler

def normalize_stock(X_train, X_test):
    target = "Close"
    tot_mean = X_train[target].mean()
    tot_std = X_train[target].std()

    for c in X_train.columns:
        mean = X_train[c].mean()
        std = X_train[c].std()

        X_train[c] = (X_train[c] - mean) / std
        X_test[c] = (X_test[c] - mean) / std

    return tot_mean, tot_std

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, 
                dataframe, 
                sequence_length=5
                ):
        
        self.scaler = StandardScaler()
        self.dataframe = dataframe

        self.sequence_length = sequence_length
        self.target= "Close"
        self.features = ['High', 'Low', 'Open', 'Volume']
        
        self.y = torch.tensor(dataframe[self.target].values).reshape(-1,1).float()
        self.X = torch.tensor((dataframe[self.features].values)).float()


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