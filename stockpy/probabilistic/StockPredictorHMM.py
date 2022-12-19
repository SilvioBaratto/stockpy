import warnings
import logging
import itertools
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from docopt import docopt
import argparse
import sys
import matplotlib.pyplot as plt  # for visualization 
import seaborn as sns  # for coloring 
from sklearn.metrics import mean_squared_error

# set style of graphs
plt.style.use('ggplot')
from pylab import rcParams
plt.rcParams['figure.dpi'] = 100


class StockRegressorHMM():

    def __init__(self, sys_argv=None, 
                hs=4,
                ld=10,
                fc=50,
                fh=10,
                fl=10):

        self.hidden_states = hs
        self.latency_days = ld
        self.frac_change = fc
        self.frac_high = fh
        self.frac_low = fl

        self.hmm = GaussianHMM(n_components=self.hidden_states)
        self.__compute_all_possible_outcomes()

    def __compute_all_possible_outcomes(self):
        frac_change_range = np.linspace(-0.1, 0.1, self.frac_change, dtype=np.float32)
        frac_high_range = np.linspace(0, 0.1, self.frac_high, dtype=np.float32)
        frac_low_range = np.linspace(0, 0.1, self.frac_low, dtype=np.float32)
 
        self._possible_outcomes = np.array(list(itertools.product(
            frac_change_range, frac_high_range, frac_low_range)))

    def __get_most_probable_outcome(self, X_test, day_index):
        previous_data_start_index = max(0, day_index - self.latency_days)
        previous_data_end_index = max(0, day_index - 1)
        previous_data = X_test.iloc[previous_data_end_index: previous_data_start_index]
        previous_data_features = StockRegressorHMM.__extract_features(previous_data)
 
        outcome_score = np.array([], dtype=np.float32)
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack(
                (previous_data_features, possible_outcome))
            outcome_score = np.append(outcome_score, self.hmm.score(total_data))
        most_probable_outcome = self._possible_outcomes[np.argmax(outcome_score)]
 
        return most_probable_outcome

    def __predict_close_price(self, X_test, day_index):
        if len(X_test) == 1:
            open_price = X_test
        else:
            open_price = X_test.iloc[day_index]['Open']
        predicted_frac_change, _, _ = self.__get_most_probable_outcome(X_test, day_index)
        return np.multiply(open_price, (1 + predicted_frac_change))

    def fit(self, X_train):
        feature_vector = StockRegressorHMM.__extract_features(X_train)    
        self.hmm.fit(feature_vector)

    def predict(self, X_test, plot=False):
        predicted_close_prices = np.array([], dtype=np.float32)

        for day_index in tqdm(range(len(X_test))):
           predicted_close_prices = np.append(predicted_close_prices,
                                               self.__predict_close_price(X_test, day_index))

        if plot:
            test_data = X_test[0: len(X_test)]
            days = np.array(test_data.index, dtype="datetime64[ms]")
            actual_close_prices = test_data['Close']
 
            fig = plt.figure()
 
            axes = fig.add_subplot(111)
            axes.plot(days, actual_close_prices, 'bo-', label="actual")
            axes.plot(days, predicted_close_prices, 'r+-', label="predicted")
 
            fig.autofmt_xdate()
 
            plt.legend()
            plt.show()
 
        return predicted_close_prices

    @staticmethod
    def __extract_features(x_train):
        """
        Instead of directly using the opening, closing, low, and high prices of a stock, 
        extract the fractional changes in each of them that would be used to train your HMM. 
        """
        
        open_price = np.array(x_train['Open'], dtype=np.float32)
        close_price = np.array(x_train['Close'], dtype=np.float32)
        high_price = np.array(x_train['High'], dtype=np.float32)
        low_price = np.array(x_train['Low'], dtype=np.float32)
 
        frac_change = np.divide((close_price - open_price),  open_price)
        
        frac_high = np.divide((high_price - open_price), open_price)
        frac_low = np.divide((open_price - low_price), open_price)
    
        return np.column_stack((frac_change, frac_high, frac_low))
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
