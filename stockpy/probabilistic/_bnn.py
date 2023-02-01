import datetime
import hashlib
import os
import shutil
import sys
import glob
sys.path.append("..")
from os.path import exists

import torch
import pyro
from torch import nn
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.nn import PyroModule
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm.auto import trange, tqdm

from util.StockDataset import normalize, StockDataset
from torch.utils.data import DataLoader
from pyro.infer.autoguide import AutoDiagonalNormal
from tqdm.auto import tqdm, trange

from util.logconf import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# set style of graphs
plt.style.use('ggplot')
from pylab import rcParams
plt.rcParams['figure.dpi'] = 100

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


from util.StockDataset import StockDataset, normalize

# TODO Implement forecasting function and plotting
# TODO Implement interface 

class Net(PyroModule):
    def __init__(self,
                input_size=4,
                hidden_size=32,
                output_dim=1,
                ):
        super(Net, self).__init__()
        self.name = "deterministic_network"

        self.model = PyroModule[nn.Sequential](
                PyroModule[nn.Linear](input_size, hidden_size),
                PyroModule[nn.ReLU](),
                PyroModule[nn.Dropout](0.2),
                PyroModule[nn.Linear](hidden_size, 16),
                PyroModule[nn.ReLU](),
                PyroModule[nn.Dropout](0.2),
                PyroModule[nn.Linear](16, output_dim),
            )       

    def forward(self, x_data, y_data=None):
        x = self.model(x_data)
        mu = x.squeeze()
        sigma = pyro.sample("sigma", dist.Uniform(0., 1.))
        with pyro.plate("data", x_data.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), 
                                        obs=y_data)
        return mu

class BNN(PyroModule):

    def __init__(self, 
                input_size=4, 
                hidden_size=32, 
                output_size=1,
                pretrained=False
                ):
        # initialize PyroModule
        super(BNN, self).__init__()

        self.pretrained = pretrained  
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size  
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        # self.model_path = self.__initModelPath()
        self.model = self.__initModel()
        self.optimizer = self.__initOptimizer()
        self.guide = self.__initGuide()

        self.name = "bayesian_network"

    def __initModel(self):

        if self.pretrained:
            model_dict = torch.load(self.model_path)

            model = Net(input_size=self.input_size, 
                        hidden_size=self.hidden_size, 
                        output_dim=self.output_size
                        )

            model.load_state_dict(model_dict['model_state'])
        
        else: 
            model = Net(input_size=self.input_size, 
                        hidden_size=self.hidden_size, 
                        output_dim=self.output_size
                        )

        return model
    
    def __initGuide(self):
        return AutoDiagonalNormal(self.model)
    
    def __initOptimizer(self):
        return pyro.optim.Adam({"lr": 0.01})

    def __initTrainDl(self, x_train, batch_size, num_workers):
        train_dl = StockDataset(x_train)

        train_dl = DataLoader(train_dl, 
                              batch_size=batch_size, 
                              num_workers=num_workers,
                              # pin_memory=self.use_cuda,
                              shuffle=False
                              )
        
        self.__batch_size = batch_size
        self.__num_workers = num_workers

        return train_dl

    def __initValDl(self, x_test):
        val_dl = StockDataset(x_test)

        val_dl = DataLoader(val_dl, 
                            batch_size=self.__batch_size, 
                            num_workers=self.__num_workers,
                            # pin_memory=self.use_cuda,
                            shuffle=False
                            )
        
        return val_dl

    def fit(self, 
            x_train,
            epochs=10,
            batch_size=8, 
            num_workers=4 
            ):

        scaler = normalize(x_train)
        x_train = scaler.fit_transform()
        train_loader = self.__initTrainDl(x_train, 
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        )

        svi = SVI(self.model, 
                  self.guide, 
                  self.optimizer, 
                  loss=Trace_ELBO())

        pyro.clear_param_store()
        for epoch_ndx in tqdm((range(1, epochs + 1)),position=0, leave=True):
            loss = 0.0
            for x_batch, y_batch in train_loader:         
                loss = svi.step(x_data=x_batch, y_data=y_batch)

    def predict(self, 
                x_test, 
                plot=False
                ):

        scaler = normalize(x_test)
        x_test = scaler.fit_transform()
        test_loader = self.__initValDl(x_test)

        output = torch.tensor([])
        for x_batch, y_batch in test_loader:
            predictive = Predictive(model=self.model, 
                                            guide=self.guide, 
                                            num_samples=self.__batch_size,
                                            return_sites=("linear.weight", 
                                                        "obs", 
                                                        "_RETURN")
                                            )

            samples = predictive(x_batch)
            site_stats = {}
            for k, v in samples.items():
                site_stats[k] = {
                    "mean": torch.mean(v, 0)
                }

            y_pred = site_stats['_RETURN']['mean']
            output = torch.cat((output, y_pred), 0)

        if plot is True:
            y_pred = output.detach().numpy() * scaler.std() + scaler.mean() # * self.std_test + self.mean_test 
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

        output = output.detach().numpy() * scaler.std() + scaler.mean()
        
        return output
    
    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            '..',
            '..',
            'models',
            'MLP',
            '{}_{}_{}.state'.format(
                    type_str,
                    self.hidden_dim,
                    self.num_layers
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
                'MLP',
                '{}_{}_{}.{}.state'.format(
                    type_str,
                    self.hidden_dim,
                    self.num_layers,
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