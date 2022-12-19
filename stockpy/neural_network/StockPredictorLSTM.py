import argparse
import datetime
import hashlib
import os
import shutil
import sys
sys.path.append("..")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.LSTM import LSTM
from dataset import StockDataset, normalize_stock

from util.logconf import logging
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns  # for coloring 
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# set style of graphs
plt.style.use('seaborn')
from pylab import rcParams
plt.rcParams['figure.dpi'] = 100


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)
from dataset import StockDataset

# TODO Implement forecasting function and plotting
# TODO Implement interface 

class StockPredictorLSTM():

    def __init__(self,
                hidden_dim=32, 
                num_layers=2,
                dropout=0.2
                ):

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        
        self.model = self.__initModel()
        self.optimizer = self.__initOptimizer()

    def __initModel(self):
        model = LSTM(hidden_dim=self.hidden_dim, 
                    num_layers=self.num_layers,
                    dropout=self.dropout
                    )

        return model

    def __initOptimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def __initTrainDl(self, x_train, batch_size, num_workers, sequence_length):
        train_dl = StockDataset(x_train, sequence_length=sequence_length)

        train_dl = DataLoader(train_dl, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers,
                                    # pin_memory=self.use_cuda,
                                    shuffle=True
                                    )

        self.__batch_size = batch_size
        self.__num_workers = num_workers
        self.__sequence_length = sequence_length

        return train_dl

    def __initValDl(self, x_test):
        val_dl = StockDataset(x_test, sequence_length=self.__sequence_length)

        val_dl = DataLoader(val_dl, 
                                    batch_size=self.__batch_size, 
                                    num_workers=self.__num_workers,
                                    # pin_memory=self.use_cuda,
                                    shuffle=False
                                    )
        
        return val_dl

    def computeBatchLoss(self, batch_tup):
            
        x = batch_tup[0]
        y = batch_tup[1]

        output = self.model(x)
        loss_function = nn.MSELoss()
        loss = loss_function(output, y)

        return loss.mean()    # This is the loss over the entire batch

    def fit(self, x_train, 
            epochs=10, 
            batch_size=8, 
            num_workers=4,
            sequence_length=30,
            save_model=True
            ):

        self.mean_train, self.std_train = self.__mean_std(x_train)

        self.__fit_transform(x_train)
        train_dl = self.__initTrainDl(x_train, 
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        sequence_length=sequence_length
                                        )

        best_score = 0.0
        total_loss = 0

        for epoch_ndx in tqdm(range(1, epochs + 1)):
            self.model.train()

            batch_iter = enumerate(train_dl)

            for batch_ndx, batch_tup in batch_iter:
                self.optimizer.zero_grad()  # Frees any leftover gradient tensors

                loss_var = self.computeBatchLoss(batch_tup)

                loss_var.backward()     # Actually updates the model weights
                self.optimizer.step()
                total_loss += loss_var
        
        # best_score = max((total_loss / len(train_dl)), best_score)
        self.saveModel('LSTM', epoch_ndx)

            # if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
            #    loss_val = self.doValidation(epoch_ndx, val_dl)
            #    best_score = max(loss_val, best_score)
            #    self.saveModel('LSTM', epoch_ndx, loss_val == best_score)

    def predict(self, 
                x_test,   
                plot=False
                ):

        self.mean_test, self.std_test = self.__mean_std(x_test)
        self.mean = self.mean_train + self.mean_test / 2
        self.std = self.std_train + self.std_test / 2

        self.__fit_transform(x_test)
        val_dl = self.__initValDl(x_test)
        batch_iter = enumerate(val_dl)

        output = torch.tensor([])
        self.model.eval()
        with torch.no_grad():
            for batch_ndx, batch_tup in batch_iter:
                y_star = self.model(batch_tup[0])
                output = torch.cat((output, y_star), 0)
        
        if plot is True:
            y_pred = output * self.std_test + self.mean_test # * self.std_train + self.mean_train
            y_test = (x_test['Close']).values * self.std_test + self.mean_test
            test_data = x_test[0: len(x_test)]
            days = np.array(test_data.index, dtype="datetime64[ms]")
            
            fig = plt.figure()
            
            axes = fig.add_subplot(111)
            axes.plot(days, y_test, label="actual") 
            axes.plot(days, y_pred, label="predicted")
            
            fig.autofmt_xdate()
            
            plt.legend()
            plt.show()
            
        return output * self.std_test + self.mean_test 

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            '..',
            '..',
            'models',
            'LSTM',
            '{}_{}.state'.format(
                type_str,
                self.time_str,
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx
        }
        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                '..',
                '..',
                'models',
                'LSTM',
                '{}_{}_{}.state'.format(
                    type_str,
                    self.time_str,
                    'best',
                )
            )
            shutil.copyfile(file_path, best_path)

            log.debug("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

    @staticmethod
    def __mean_std(dataframe):
        target = "Close"
        return dataframe[target].mean(), dataframe[target].std()

    @staticmethod
    def __fit_transform(dataframe):
        for c in dataframe.columns:
            mean = dataframe[c].mean()
            std = dataframe[c].std()

            dataframe[c] = (dataframe[c] - mean) / std

