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

from util.util import enumerateWithEstimate
from util.logconf import logging
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# set style of graphs
plt.style.use('seaborn')
from pylab import rcParams
plt.rcParams['figure.dpi'] = 100


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class Training():
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument("--stock",
                            help="Stock to traing the model",
                            action="store",
                            default=None,
                            )
        parser.add_argument("--test-size",
                            help="Validation test size",
                            action="store",
                            default=0.2,
                            type=float,
                            )
        parser.add_argument("--hidden-dim",
                            help="Number of hidden dimension of LSTM",
                            action="store",
                            default=32,
                            type=int,
                            )
        parser.add_argument("--num-layers",
                            help="Number of hidden layers of LSTM",
                            action="store",
                            default=2,
                            type=int,
                            )
        parser.add_argument("--dropout",
                            help="Dropout coefficient",
                            action="store",
                            default=0.2,
                            type=int,
                            )
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=4,
                            type=int,
                            )
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=4,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=10,
                            type=int,
                            )
        parser.add_argument('--model',
                            help="What to model class name to use.",
                            action='store',
                            default='StockPredictorLSTM',
                            )
        parser.add_argument("--sequence-length",
                            help="Sequence to loock back in LSTM",
                            action="store",
                            default=30,
                            type=int,
                            )
        parser.add_argument('--forecasting',
                            help="Forecast untill n days",
                            action="store",
                            default=90,
                            type=int,
                            )
        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='LSTM',
                            )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.cli_args.stock = pd.read_csv('../../stock/' + self.cli_args.stock + '.csv', 
                                parse_dates=True, 
                                index_col='Date')

        self.X_train, self.X_test = train_test_split(self.cli_args.stock, 
                                            test_size=self.cli_args.test_size,
                                            shuffle=False
                                            )
        self.mean, self.std = normalize_stock(self.X_train, self.X_test)

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

        self.totalTrainingSamples_count = 0

    def initModel(self):
        model = LSTM(hidden_dim=self.cli_args.hidden_dim,
                                    num_layers=self.cli_args.num_layers,
                                    dropout=self.cli_args.dropout
                                )

        # if self.use_cuda:
        #    log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
        #    if torch.cuda.device_count() > 1:
        #        model = nn.DataParallel(model)
        #    model = model.to(self.device)

        return model

    def initTrainDl(self):
        # if self.use_cuda:
        #    batch_size *= torch.cuda.device_count()

        train_dl = StockDataset(self.X_train, 
                                self.cli_args.sequence_length
                                )

        train_dl = DataLoader(train_dl, 
                                    batch_size=self.cli_args.batch_size, 
                                    num_workers=self.cli_args.num_workers,
                                    # pin_memory=self.use_cuda,
                                    shuffle=True
                                    )

        return train_dl

    def initValDl(self):
        # if self.use_cuda:
        #    batch_size *= torch.cuda.device_count()

        val_dl = StockDataset(self.X_test, 
                                self.cli_args.sequence_length
                                )

        val_dl = DataLoader(val_dl, 
                                    batch_size=self.cli_args.batch_size, 
                                    num_workers=self.cli_args.num_workers,
                                    # pin_memory=self.use_cuda,
                                    shuffle=False
                                    )

        return val_dl

    def initOptimizer(self):
        return Adam(self.model.parameters(), lr=0.01)

    def initLoss(self):
        return nn.MSELoss()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        total_loss = 0

        batch_iter = enumerateWithEstimate( # sets up our batch looping with time estimate
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()  # Frees any leftover gradient tensors

            loss_var = self.computeBatchLoss(batch_tup)

            loss_var.backward()     # Actually updates the model weights
            self.optimizer.step()
            total_loss += loss_var

        return total_loss / len(train_dl)

    def computeBatchLoss(self, batch_tup):
            
        x = batch_tup[0]
        y = batch_tup[1]

        output = self.model(x)
        loss_function = nn.MSELoss()
        loss = loss_function(output, y)

        return loss.mean()    # This is the loss over the entire batch

    def doValidation(self, epoch_ndx, val_dl):
        total_loss = 0
        with torch.no_grad():
            self.model.eval()   # Turns off training-time behaviour

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                loss_var = self.computeBatchLoss(batch_tup)
                total_loss += loss_var
        return total_loss / len(val_dl)

    def doPredict(self, val_dl):

        output = torch.tensor([])
        self.model.eval()
        with torch.no_grad():
            for X, _ in val_dl:
                y_star = self.model(X)
                output = torch.cat((output, y_star), 0)
            
        return output

    def doPlot(self, val_dl):
        y_pred = self.doPredict(val_dl)
        y_pred = y_pred * self.std + self.mean
        y_test = (self.X_test['Close'] * self.std + self.mean).values
        test_data = self.X_test[0: len(self.X_test)]
        days = np.array(test_data.index, dtype="datetime64[ms]")
        actual_close_prices = test_data['Close']
        
        fig = plt.figure()
        
        axes = fig.add_subplot(111)
        axes.plot(days, y_test, label="actual") 
        axes.plot(days, y_pred, label="predicted")
        
        fig.autofmt_xdate()
        
        plt.legend()
        plt.show()

    def main(self):
        # log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()   # The validation data loader is very similar to training

        
        best_score = 0.0
        validation_cadence = 5 
        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if 0 else 1),
            ))

            loss_train = self.doTraining(epoch_ndx, train_dl)
            # self.logMetrics(epoch_ndx, 'trn', loss_train)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                loss_val = self.doValidation(epoch_ndx, val_dl)
                best_score = max(loss_val, best_score)

                # self.saveModel('LSTM', epoch_ndx, loss_val == best_score)

        self.doPlot(val_dl)
    
    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            '..',
            '..',
            'models',
            'LSTM',
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.cli_args.comment,
                self.totalTrainingSamples_count,
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
                '{}_{}_{}.{}.state'.format(
                    type_str,
                    self.time_str,
                    self.cli_args.comment,
                    'best',
                )
            )
            shutil.copyfile(file_path, best_path)

            log.debug("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

    

if __name__ == '__main__':
    Training().main()
