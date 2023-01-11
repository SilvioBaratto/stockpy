import datetime
import sys
from tqdm.auto import tqdm, trange
sys.path.append("..")

import numpy as np
from util.StockDataset import normalize

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from util.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# set style of graphs
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pylab import rcParams
plt.rcParams['figure.dpi'] = 100

class LINEAR():
    def __init__(self):
        self.predictor = LinearRegression()

    def fit(self, 
            x_train,
            features=['Open', 'High', 'Low', 'Volume'],
            target=['Close']
            ):

        self._features = features
        self._target = target
        scaler = normalize(x_train)
        x_train = scaler.fit_transform()
        self.predictor.fit(x_train[features], x_train[target].squeeze())

    def predict(self,
                x_test,
                plot=False
                ):
        scaler = normalize(x_test)
        x_test = scaler.fit_transform()

        y_pred = self.predictor.predict(x_test[self._features])
    

        if plot is True:
            y_pred = y_pred * scaler.std() + scaler.mean() # * self.std_test + self.mean_test 
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
            
        return y_pred * scaler.std() + scaler.mean()# * self.std_test + self.mean_test 