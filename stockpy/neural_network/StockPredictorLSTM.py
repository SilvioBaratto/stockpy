import argparse
import datetime
import hashlib
import os
import shutil
import sys
import glob
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
from tqdm.auto import tqdm, trange
from sklearn.preprocessing import StandardScaler

# set style of graphs
plt.style.use('ggplot')
from pylab import rcParams
plt.rcParams['figure.dpi'] = 100


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)
from dataset import StockDataset

# TODO Implement forecasting function and plotting
# TODO Implement interface 

class normalize():
    def __init__(self,
                dataframe
                ):

        self.scaler = StandardScaler()
        self.dataframe = dataframe

    def fit_transform(self):
        df_scaled = self.scaler.fit_transform(self.dataframe)
        df_scaled = pd.DataFrame(df_scaled, 
                                columns=self.dataframe.columns,
                                index=self.dataframe.index
                                )
        return df_scaled

    def inverse_transform(self):
        df_inverse = self.scaler.inverse_transform(self.dataframe)
        df_inverse = pd.DataFrame(df_inverse, 
                                columns=self.dataframe.columns,
                                index=self.dataframe.index
                                )
        return df_inverse

    def mean(self):
        df = pd.DataFrame(self.scaler.mean_.reshape(1,-1),
                            columns=self.dataframe.columns,
                            )
        return df['Close'][0]
    
    def std(self):
        df = pd.DataFrame(self.scaler.scale_.reshape(1,-1),
                            columns=self.dataframe.columns,
                            )
        return df['Close'][0]

class StockPredictorLSTM():

    def __init__(self,
                hidden_dim=32, 
                num_layers=2,
                dropout=0.2,
                pretrained=False
                ):
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pretrained = pretrained

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        
        self.model_path = self.__initModelPath()
        self.model = self.__initModel()
        self.optimizer = self.__initOptimizer()

    def __initModelPath(self):
        local_path = os.path.join(
                '..',
                '..',
                'models',
                'LSTM',
                'LSTM_{}.state'.format('*', 'best'),
        )

        file_list = glob.glob(local_path)
        if not file_list:
            pretrained_path = os.path.join(
                    '..',
                    '..',
                    'models',
                    'LSTM',
                    'LSTM_{}.state'.format('*'),
                )
            file_list = glob.glob(pretrained_path)

        else:
            pretrained_path = None

        file_list.sort()

        try:
            return file_list[-1]
        except IndexError:
            raise

    def __initModel(self):

        if self.pretrained:
            model_dict = torch.load(self.model_path)

            model = LSTM(hidden_dim=self.hidden_dim, 
                        num_layers=self.num_layers,
                        dropout=self.dropout
                        )

            model.load_state_dict(model_dict['model_state'])
        
        else: 
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
        val_dl = StockDataset(x_test, 
                                sequence_length=self.__sequence_length
                                )

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
            save_model=True,
            validation_sequence=30,
            ):

        # self.mean_train, self.std_train = self.__mean_std(x_train)

        # self.__fit_transform(x_train)
        scaler = normalize(x_train)

        x_train = scaler.fit_transform()
        val_dl = x_train[-validation_sequence:]
        x_train = x_train[:len(x_train)-len(val_dl)]

        train_dl = self.__initTrainDl(x_train, 
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        sequence_length=sequence_length
                                        )

        val_dl = self.__initValDl(val_dl)

        best_score = 0.0
        total_loss = 0
        validation_cadence = 5
        
        for epoch_ndx in tqdm((range(1, epochs + 1)),position=0, leave=True):
            self.model.train()

            batch_iter = enumerate(train_dl)

            for batch_ndx, batch_tup in batch_iter:
                self.optimizer.zero_grad()  # Frees any leftover gradient tensors

                loss_var = self.computeBatchLoss(batch_tup)

                loss_var.backward()     # Actually updates the model weights
                self.optimizer.step()
                total_loss += loss_var
        
            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                loss_val = self.doValidation(epoch_ndx, val_dl)
                # print("loss_val {}".format(loss_val))
                best_score = max(loss_val, best_score)
                # self.saveModel('LSTM', epoch_ndx, loss_val == best_score)

    def doValidation(self, epoch_ndx, val_dl):
        total_loss = 0
        with torch.no_grad():
            self.model.eval()   # Turns off training-time behaviour

            batch_iter = enumerate(val_dl)

            for batch_ndx, batch_tup in batch_iter:
                loss_var = self.computeBatchLoss(batch_tup)
                total_loss += loss_var

        return total_loss / len(val_dl)

    def predict(self, 
                x_test,   
                plot=False
                ):

        # self.mean_test, self.std_test = self.__mean_std(x_test)
        # self.mean = self.mean_train + self.mean_test / 2
        # self.std = self.std_train + self.std_test / 2
        scaler = normalize(x_test)
        x_test = scaler.fit_transform()
        # self.__fit_transform(x_test)
        val_dl = self.__initValDl(x_test)
        batch_iter = enumerate(val_dl)

        output = torch.tensor([])
        self.model.eval()
        with torch.no_grad():
            for batch_ndx, batch_tup in batch_iter:
                y_star = self.model(batch_tup[0])
                output = torch.cat((output, y_star), 0)
        
        if plot is True:
            y_pred = output * scaler.std() + scaler.mean() # * self.std_test + self.mean_test 
            y_test = (x_test['Close']).values * scaler.std() + scaler.mean() # * self.std_test + self.mean_test
            test_data = x_test[0: len(x_test)]
            days = np.array(test_data.index, dtype="datetime64[ms]")
            
            fig = plt.figure()
            
            axes = fig.add_subplot(111)
            axes.plot(days, y_test, 'bo-', label="actual") 
            axes.plot(days, y_pred, 'r+-', label="predicted")
            
            fig.autofmt_xdate()
            
            plt.legend()
            plt.show()
            
        return output * scaler.std() + scaler.mean()# * self.std_test + self.mean_test 

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

        # log.debug("Saved model params to {}".format(file_path))

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

            # log.debug("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            hashlib.sha1(f.read()).hexdigest()

    def initModelPath(self, type_str):
        local_path = os.path.join(
            '..',
            '..',
            'models',
            type_str + '_{}.state'.format('*', '*', 'best'),
        )

        file_list = glob.glob(local_path)
        if not file_list:
            pretrained_path = os.path.join(
                '..',
                '..',
                'models',
                type_str + '_{}_{}.{}.state'.format('*', '*', '*'),
            )
            file_list = glob.glob(pretrained_path)
        else:
            pretrained_path = None

        file_list.sort()

        try:
            return file_list[-1]
        except IndexError:
            log.debug([local_path, pretrained_path, file_list])
            raise
    """
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
    """
