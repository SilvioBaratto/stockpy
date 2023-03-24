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
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler

from utils import StockDataset, normalize
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

# set style of graphs
plt.style.use('ggplot')
from pylab import rcParams
plt.rcParams['figure.dpi'] = 100


class Net(nn.Module):
    def __init__(self, 
                input_size=4,  
                hidden_size=32, 
                num_layers=2, 
                output_dim=1, 
                dropout=0.2,
                seq_length=30
                ):

        super().__init__()
        self.input_size = input_size # this is the number of features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = seq_length

        self.gru = nn.GRU(input_size, 
                          hidden_size, 
                          num_layers, 
                          batch_first=True,
                          bidirectional=True
                          )
        self.fc = nn.Linear(hidden_size*2, output_dim)
        # self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size))
        
        out, _ = self.gru(x, (h0))
        out = self.fc(out[:,-1,:]) #Final Output
       
        out = out.view(-1,1)
        return out


class BiGRU():

    def __init__(self,
                input_size=4,
                hidden_size=32, 
                num_layers=2,
                dropout=0.2,
                pretrained=False
                ):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.pretrained = pretrained
        
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')      
        # self.model_path = self.__initModelPath()
        self.model = self.__initModel()

        self.name = "bidirectionalGRU neural network"

    def _initOptimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)
    
    def _initScheduler(self):
        return lr_scheduler.StepLR(self._optimizer, 
                                   step_size=10,
                                   gamma=0.1
                                   )
    
    def _initTrainDl(self, 
                     x_train, 
                     batch_size, 
                     num_workers
                     ):
        train_dl = StockDataset(x_train)

        train_dl = DataLoader(train_dl, 
                            batch_size=batch_size * (torch.cuda.device_count() \
                                                                   if self.use_cuda else 1),  
                            num_workers=num_workers,
                            pin_memory=self.use_cuda,
                            shuffle=False
                            )

        self._batch_size = batch_size
        self._num_workers = num_workers

        return train_dl

    def _initValDl(self, x_test):
        val_dl = StockDataset(x_test)

        val_dl = DataLoader(val_dl, 
                            batch_size=self._batch_size * (torch.cuda.device_count() \
                                                    if self.use_cuda else 1), 
                            num_workers=self._num_workers,
                            pin_memory=self.use_cuda,
                            shuffle=False
                            )
        
        return val_dl
    
    def _initTrainValData(self, 
                          x_train,
                          validation_sequence,
                          batch_size,
                          num_workers
                          ):
        
        scaler = normalize(x_train)

        x_train = scaler.fit_transform()
        val_dl = x_train[-validation_sequence:]
        x_train = x_train[:len(x_train)-len(val_dl)]

        train_dl = self._initTrainDl(x_train, 
                                        batch_size=batch_size,
                                        num_workers=num_workers
                                        )

        val_dl = self._initValDl(val_dl)

        return train_dl, val_dl

    def fit(self, 
            x_train,
            epochs=10,
            sequence_length=30,
            batch_size=8, 
            num_workers=4,
            validation_sequence=30, 
            validation_cadence=5,
            patience=5
            ):
        
        train_dl, val_dl = self._initTrainValData(x_train,
                                                  validation_sequence,
                                                  batch_size,
                                                  num_workers
                                                  )
        
        self._train(epochs,
                    train_dl,
                    val_dl,
                    validation_cadence,
                    patience
                    )
        
    def _train(self, 
               epochs,
               train_dl,
               val_dl,
               validation_cadence,
               patience
               ):

        best_loss = float('inf')
        counter = 0

        for epoch_ndx in tqdm((range(1, epochs + 1)), position=0, leave=True):
            epoch_loss = 0.0
            for x_batch, y_batch in train_dl:  
                epoch_loss += self._computeBatchLoss(x_batch, y_batch)
            
            self._scheduler.step()

            if epoch_ndx % validation_cadence != 0:                
                print(f"Epoch {epoch_ndx}, Loss: {epoch_loss / len(train_dl)}")

            else:
                total_loss = self._doValidation(val_dl)

                print(f"Epoch {epoch_ndx}, Val Loss {total_loss}")

                # Early stopping
                stop, best_loss, counter = self._earlyStopping(total_loss, 
                                                               best_loss, 
                                                               counter, 
                                                               patience,
                                                               epoch_ndx
                                                               )
                if stop:
                    break

                self._model.train()

    def _computeBatchLoss(self, 
                         x_batch, 
                         y_batch
                         ):     
              
        output = self.model(x_batch)
        loss_function = nn.MSELoss()
        loss = loss_function(output, y_batch)

        return loss.mean()

    def _doValidation(self, epoch_ndx, val_dl):
        total_loss = 0
        self._model.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                loss_var = self._computeBatchLoss(x_batch, y_batch)
                total_loss += loss_var / val_dl.batch_size

        return total_loss 
    
    def _earlyStopping(self,
                       total_loss,
                       best_loss,
                       counter,
                       patience,
                       epoch_ndx
                       ):
        if total_loss < best_loss:
            best_loss = total_loss
            best_epoch_ndx = epoch_ndx
            self.saveModel('bnn', best_epoch_ndx)
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"No improvement after {patience} epochs. Stopping early.")
            return True, best_loss, counter
        else:
            return False, best_loss, counter

    def predict(self, 
                x_test,   
                plot=False
                ):

        scaler = normalize(x_test)
        x_test = scaler.fit_transform()
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

    def _initModel(self):
        
        model = Net(
                input_dim=self._input_dim, 
                z_dim=self._z_dim, 
                emission_dim=self._emission_dim,
                transition_dim=self._transition_dim, 
                rnn_dim=self._rnn_dim, 
                rnn_dropout_rate=self._rnn_dropout_rate,
                variance=self._variance
            )
        
        if self._pretrained:
            path = self._initModelPath('dmm')
            model_dict = torch.load(path)
            model.load_state_dict(model_dict['model_state'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            self._model = model.to(device)

        self._model = model

        self._optimizer = self._initOptimizer()
        self._svi = SVI(self._initStepModel(), 
                        self._initGuide(), 
                        self._optimizer, 
                        loss=TraceMeanField_ELBO()
                    )
        # Create learning rate scheduler
        self._scheduler = self._initScheduler()

    def _initModelPath(self, type_str):
        model_dir = '../../models/DMM'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        local_path = os.path.join(
            '..', 
            '..', 
            'models', 
            'DMM', 
            type_str + '_{}_{}_{}_{}_{}_{}_{}.state'.format(self._input_dim,
                                                            self._z_dim,
                                                            self._emission_dim,
                                                            self._transition_dim,
                                                            self._rnn_dim,
                                                            self._rnn_dropout_rate,
                                                            self._variance
                                                            ),
            )

        file_list = glob.glob(local_path)
        
        if not file_list:
            raise ValueError(f"No matching model found in {local_path} for the given parameters.")
        
        # Return the most recent matching file
        return file_list[0]

    def trading(self, 
                predicted, 
                real, 
                shares=0, 
                stop_loss=0.0, 
                initial_balance=10000, 
                threshold=0.0, 
                plot=True
                ):
        """
        Simulates trading on predicted values and compare to actual stock prices.
        
        Args:
            y_pred (torch.Tensor): Predicted stock prices.
            y_test (torch.Tensor): Actual stock prices.
            shares (int): Number of shares owned initially.
            stop_loss (float): The stop loss amount. If the stock price falls below this value, shares are sold.
            initial_balance (float): The initial balance available for trading.
            plot (bool): Whether to plot the trading simulation results.
        
        Returns:
            Tuple of final balance, total profit/loss, and a list of tuples representing each transaction:
            (timestamp, price, action, shares, balance).
            If `plot` is True, also returns a Matplotlib figure object.
        """
        assert predicted.shape == real.shape, "predicted and real must have the same shape"
        assert shares >= 0, "shares cannot be negative"
        assert initial_balance >= 0, "initial_balance cannot be negative"
        assert 0 <= stop_loss <= 1, "stop_loss must be between 0 and 1"

        transactions = []
        balance = initial_balance
        num_shares = shares
        total_profit_loss = 0

        if num_shares == 0 and balance >= real[0]:
            num_shares = int(balance / real[0])
            balance -= num_shares * real[0]
            transactions.append((0, real[0], "BUY", num_shares, balance))

        for i in range(1, len(predicted)):
            if predicted[i] > real[i-1] * (1 + threshold):
                if num_shares == 0:
                    num_shares = int(balance / real[i])
                    balance -= num_shares * real[i]
                    transactions.append((i, real[i], "BUY", num_shares, balance))
                elif num_shares > 0:
                    balance += num_shares * real[i]
                    total_profit_loss += (real[i] - real[i-1]) * num_shares
                    transactions.append((i, real[i], "SELL", num_shares, balance))
                    num_shares = 0
            elif predicted[i] < real[i-1] * (1 - threshold):
                if num_shares == 0:
                    continue
                elif num_shares > 0:
                    balance += num_shares * real[i]
                    total_profit_loss += (real[i] - real[i-1]) * num_shares
                    transactions.append((i, real[i], "SELL", num_shares, balance))
                    num_shares = 0

            if stop_loss > 0 and num_shares > 0 and real[i] < (real[0] - stop_loss):
                balance += num_shares * real[i]
                total_profit_loss += (real[i] - real[i-1]) * num_shares
                transactions.append((i, real[i], "SELL", num_shares, balance))
                num_shares = 0

        if num_shares > 0:
            balance += num_shares * real[-1]
            total_profit_loss += (real[-1] - real[-2]) * num_shares
            transactions.append((len(predicted)-1, real[-1], "SELL", num_shares, balance))
            num_shares = 0

        percentage_increase = (balance - initial_balance) / initial_balance * 100

        if plot:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(real, label='Real')
            ax.plot(predicted, label='Predicted')
            buy_scatter = ax.scatter([], [], c='g', marker='^', s=100)
            sell_scatter = ax.scatter([], [], c='r', marker='v', s=100)
            for transaction in transactions:
                timestamp, price, action, shares, balance = transaction
                if action == 'BUY':
                    buy_scatter = ax.scatter(timestamp, predicted[timestamp], c='g', marker='^', s=100)
                elif action == 'SELL':
                    sell_scatter = ax.scatter(timestamp, predicted[timestamp], c='r', marker='v', s=100)
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.set_title('Trading Simulation')
            fig.autofmt_xdate()
            ax.legend((ax.plot([], label='Real')[0], ax.plot([], label='Predicted')[0], buy_scatter, sell_scatter),
                    ('Real', 'Predicted', 'Buy', 'Sell'))
            ax.text(0.05, 0.05, 
                    'Percentage increase: ${:.2f}%'.format(percentage_increase[0]), 
                    ha='left', va='center',
                      transform=ax.transAxes, 
                      bbox=dict(facecolor='white', alpha=0.5)
                      )
            plt.show()
        
        return balance, total_profit_loss, percentage_increase, transactions