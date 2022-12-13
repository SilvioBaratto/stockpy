import argparse
import datetime
import hashlib
import os
import shutil
import sys

import argparse
import logging
import time
from os.path import exists

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from StockPredictorDMM import DMM

import pyro
import pyro.contrib.examples.polyphonic_data_loader as poly
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    TraceTMC_ELBO,
    config_enumerate,
)
from pyro.optim import ClippedAdam


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, stock, 
                isval=False,
                test_size=0.1):
        
        features = ['Open', 'High', 'Low', 'Volume']
        output = ['Close']

        stock = pd.read_csv("../../stock/" + stock + ".csv", parse_dates=True, index_col='Date')

        X_train, X_test, y_train, y_test = train_test_split(stock[features], 
                                                            stock[output],
                                                            test_size=test_size, 
                                                            shuffle=False
                                                            )
        if isval is False:                                         
            self.X = torch.from_numpy(X_train.pct_change().dropna(how='any').to_numpy())
            self.y = torch.from_numpy(y_train.pct_change().dropna(how='any').to_numpy())

        else:
            self.X = torch.from_numpy(X_test.pct_change().dropna(how='any').to_numpy())
            self.y = torch.from_numpy(y_test.pct_change().dropna(how='any').to_numpy())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]      

class TrainingApp():
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument("-s",
                            "--stock",
                            help="Stock from S%P",
                            action="store",
                            default=None,
                        )
        parser.add_argument('--test-size',
                            help="Test size training",
                            action="store",
                            default=0.3,
                            type=float,
                        )
        parser.add_argument("-n", 
                            "--num-epochs", 
                            type=int, 
                            default=5000
                            )
        parser.add_argument("-lr", 
                            "--learning-rate", 
                            type=float, 
                            default=0.0003
                            )
        parser.add_argument("-b1", 
                            "--beta1", 
                            type=float, 
                            default=0.96
                            )
        parser.add_argument("-b2", 
                            "--beta2", 
                            type=float, 
                            default=0.999
                            )
        parser.add_argument("-cn", 
                            "--clip-norm", 
                            type=float, 
                            default=10.0
                            )
        parser.add_argument("-lrd", 
                            "--lr-decay", 
                            type=float, 
                            default=0.99996
                            )
        parser.add_argument("-wd", 
                            "--weight-decay", 
                            type=float, 
                            default=2.0
                            )
        parser.add_argument("-mbs", 
                            "--mini-batch-size", 
                            type=int, 
                            default=20
                            )
        parser.add_argument("-ae", 
                            "--annealing-epochs", 
                            type=int, 
                            default=1000
                            )
        parser.add_argument("-maf", 
                            "--minimum-annealing-factor", 
                            type=float, 
                            default=0.2
                            )
        parser.add_argument("-rdr",         
                            "--rnn-dropout-rate", 
                            type=float, 
                            default=0.1
                            )
        parser.add_argument("-iafs", 
                            "--num-iafs", 
                            type=int, 
                            default=0
                            )
        parser.add_argument("-id", 
                            "--iaf-dim", 
                            type=int, 
                            default=100
                            )
        parser.add_argument("-cf", 
                            "--checkpoint-freq", 
                            type=int, 
                            default=0
                            )
        parser.add_argument("-lopt", 
                            "--load-opt", 
                            type=str, 
                            default=""
                            )
        parser.add_argument("-lmod", 
                            "--load-model", 
                            type=str, 
                            default=""
                            )
        parser.add_argument("-sopt", 
                            "--save-opt", 
                            type=str, 
                            default=""
                            )
        parser.add_argument("-smod", 
                            "--save-model", 
                            type=str, 
                            default=""
                            )
        parser.add_argument("--cuda", 
                            action="store_true"
                            )
        parser.add_argument("--jit",
                            action="store_true"
                            )
        parser.add_argument("--tmc", 
                            action="store_true"
                            )
        parser.add_argument("--tmcelbo", 
                            action="store_true"
                            )
        parser.add_argument("--tmc-num-samples", 
                            default=10, 
                            type=int
                            )
        parser.add_argument("-l", 
                            "--log", 
                            type=str, 
                            default="dmm.log"
                            )

        self.args = parser.parse_args()

        self.model = self.initModel()
        self.train = self.initTrainDt()
        self.val = self.initValDt()
        self.inference = self.initInference()

    def initModel(self):
        self.dmm = DMM(
            rnn_dropout_rate=self.args.rnn_dropout_rate,
            num_iafs=self.args.num_iafs,
            iaf_dim=self.args.iaf_dim,
            use_cuda=self.args.cuda,
        )

        if self.args.cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                self.dmm = nn.DataParallel(self.dmm)
            self.dmm = self.dmm.to(self.device)

    def initOptimizer(self):

        # setup optimizer
        adam_params = {
            "lr": self.args.learning_rate,
            "betas": (self.args.beta1, self.args.beta2),
            "clip_norm": self.args.clip_norm,
            "lrd": self.args.lr_decay,
            "weight_decay": self.args.weight_decay,
        }
        return ClippedAdam(adam_params) 

    def initInference(self):
        # setup inference algorithm
        adam = self.initOptimizer()

        if self.args.tmc:
            if self.args.jit:
                raise NotImplementedError("no JIT support yet for TMC")
            tmc_loss = TraceTMC_ELBO()
            dmm_guide = config_enumerate(
                self.dmm.guide,
                default="parallel",
                num_samples=self.args.tmc_num_samples,
                expand=False,
            )
            svi = SVI(self.dmm.model, dmm_guide, adam, loss=tmc_loss)
        elif self.args.tmcelbo:
            if self.args.jit:
                raise NotImplementedError("no JIT support yet for TMC ELBO")
            elbo = TraceEnum_ELBO()
            dmm_guide = config_enumerate(
                self.dmm.guide,
                default="parallel",
                num_samples=self.args.tmc_num_samples,
                expand=False,
            )
            svi = SVI(self.self.dmm.model, dmm_guide, adam, loss=elbo)
        else:
            elbo = JitTrace_ELBO() if self.args.jit else Trace_ELBO()
            svi = SVI(self.dmm.model, self.dmm.guide, adam, loss=elbo)

        return svi

    def initTrainDt(self):
        train_df = StockDataset(self.args.stock,
                                isval=False)

        train = DataLoader(  # An. off-the-shelf class
            train_df,
            batch_size=24,  # batching is done automatically
            num_workers=4,
        )

        return train

    def initValDt(self):
        val_df = StockDataset(self.args.stock,
                                isval=True)

        val = DataLoader(  # An. off-the-shelf class
            val_df,
            batch_size=24,  # batching is done automatically
            num_workers=4,
        )

        return val

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()   # The validation data loader is very similar to training

        best_score = 0.0
        validation_cadence = 5 if not self.cli_args.finetune else 1
        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)

                # TODO: this 'cls' will need to change for the malignant classifier
                self.saveModel('cls', epoch_ndx, score == best_score)


        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def process_minibatch(epoch, which_mini_batch, shuffled_indices):
        if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
            # compute the KL annealing factor appropriate
            # for the current mini-batch in the current epoch
            min_af = args.minimum_annealing_factor
            annealing_factor = min_af + (1.0 - min_af) * \
                (float(which_mini_batch + epoch * N_mini_batches + 1) /
                float(args.annealing_epochs * N_mini_batches))
        else:
            # by default the KL annealing factor is unity
            annealing_factor = 1.0

        # compute which sequences in the training set we should grab
        mini_batch_start = (which_mini_batch * args.mini_batch_size)
        mini_batch_end = np.min([(which_mini_batch + 1) * args.mini_batch_size,
                                N_train_data])
        mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
        # grab the fully prepped mini-batch using the helper function in the data loader
        mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
            = poly.get_mini_batch(mini_batch_indices, training_data_sequences,
                                training_seq_lengths, cuda=args.cuda)
        # do an actual gradient step
        loss = svi.step(mini_batch, mini_batch_reversed, mini_batch_mask,
                        mini_batch_seq_lengths, annealing_factor)
        # keep track of the training loss
        return loss

    # prepare a mini-batch and take a gradient step to minimize -elbo
    def compute_batch_loss(self, epoch, batch_ndx, shuffled_indices):
        N_mini_batches = list(enumerate(self.train))[-1][0]
        if self.args.annealing_epochs > 0 and epoch < self.args.annealing_epochs:
            # compute the KL annealing factor appropriate
            # for the current mini-batch in the current epoch
            min_af = self.args.minimum_annealing_factor
            annealing_factor = min_af + (1.0 - min_af) * \
                (float(batch_ndx + self.epoch * N_mini_batches + 1) /
                float(self.args.annealing_epochs * N_mini_batches))
        else:
            # by default the KL annealing factor is unity
            annealing_factor = 1.0
        # compute which sequences in the training set we should grab
        mini_batch_start = (batch_ndx * self.args.mini_batch_size)
        mini_batch_end = np.min([(batch_ndx + 1) * self.args.mini_batch_size,
                                len(self.train.dataset)])
        mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
        # grab the fully prepped mini-batch using the helper function in the data loader
        mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
            = poly.get_mini_batch(mini_batch_indices, training_data_sequences,
                                training_seq_lengths, cuda=args.cuda)
        # do an actual gradient step
        loss = svi.step(mini_batch, mini_batch_reversed, mini_batch_mask,
                        mini_batch_seq_lengths, annealing_factor)
        # keep track of the training loss
        return loss

    @staticmethod
    def rep(x):
        rep_shape = torch.Size([x.size(0) * 1]) + x.size()[1:]
        repeat_dims = [1] * len(x.size())
        repeat_dims[0] = 1
        return (
            x.repeat(repeat_dims)
            .reshape(1, -1)
            .transpose(1, 0)
            .reshape(rep_shape)
        )

    """
    def doValidation(self, epoch_ndx, val_dl):
        # put the RNN into evaluation mode (i.e. turn off drop-out if applicable)
        self.dmm.rnn.eval()

        test_nll = self.inference.evaluate_loss(
            test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths
        ) / float(torch.sum(test_seq_lengths))

        # put the RNN back into training mode (i.e. turn on drop-out if applicable)
        self.dmm.rnn.train()
        return test_nll
    """

    def doTraining(self):
        times = [time.time()]
        N_mini_batches = int(
            len(self.train.dataset) / self.args.mini_batch_size
            + int(len(self.train.dataset) % self.args.mini_batch_size > 0)
        )
        for epoch in range(self.args.num_epochs):
            # accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
            epoch_nll = 0.0
            # prepare mini-batch subsampling indices for this epoch
            shuffled_indices = torch.randperm(len(self.train.dataset))

            # process each mini-batch; this is where we take gradient steps
            for which_mini_batch in range(N_mini_batches):
                epoch_nll += self.compute_batch_loss(which_mini_batch, shuffled_indices)

            # report training diagnostics
            times.append(time.time())
            epoch_time = times[-1] - times[-2]
            logging.info(
                "[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)"
                % (epoch, epoch_nll, epoch_time)
            )

if __name__ == '__main__':
    TrainingApp().doTraining()