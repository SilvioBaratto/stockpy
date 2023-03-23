import sys
sys.path.append("..")

import itertools
import numpy as np
from hmmlearn.hmm import GaussianHMM
from tqdm.auto import tqdm

import matplotlib.pyplot as plt  # for visualization 

# set style of graphs
plt.style.use('ggplot')
from pylab import rcParams
plt.rcParams['figure.dpi'] = 100


class GHMM():

    @staticmethod
    def __extract_features(dataframe):
        """
        This function takes as input a dataframe containing stock values
        and return the fractional changes 

        frac_change = (close - open) / open
        frac_high = (high - open) / open
        frac_low = (open - low) / open

        Input: Stock dataframe
        Output: Fractional changes 
        """
        open = np.array(dataframe['Open'], dtype=np.float32)
        close = np.array(dataframe['Close'], dtype=np.float32)
        high = np.array(dataframe['High'], dtype=np.float32)
        low = np.array(dataframe['Low'], dtype=np.float32)
 
        fc = np.divide((close - open),  open)
        
        fh = np.divide((high - open), open)
        fl = np.divide((open - low), open)
    
        return np.column_stack((fc, fh, fl))

    def __init__(self, 
                hs=4,
                ld=10,
                fc=50,
                fh=10,
                fl=10
                ):

        self.hs = hs # hidden states
        self.fc = fc    
        self.fh = fh
        self.fl = fl 
        self.ld = ld    # latent days to predict the closing price
                        # of the current day  

        self.hmm = GaussianHMM(n_components=self.hs)
        self.__possible_outcome()


    def fit(self, X_train):
        """
        Function to estimate the model parameter. 
        An initialization step is performed before entering the
        EM algorithm.

        Input: Matrix of individual samples
        """
        feature_vector = GHMM.__extract_features(X_train)    
        self.hmm.fit(feature_vector)
        

    def __possible_outcome(self):
        """
        This function generate all possible permutations of values for the features.
        The function do the Cartesian product across a range of values for each feature.
        In this function we assume:

            1. the distribution of each feature is across an evenely spaced interval instead
            of being fully continuous
            2. Possible values for the start and end of the intervals

        Output: Cartesian product across a range of values for each feature
        """
        # np.linspace(Minimum value, Maximum Value, Number of Points)
        _fc = np.linspace(-0.1, 0.1, self.fc, dtype=np.float32)
        _fh = np.linspace(0, 0.1, self.fh, dtype=np.float32)
        _fl = np.linspace(0, 0.1, self.fl, dtype=np.float32)
 
        self._possible_outcomes = np.array(list(itertools.product(_fc, _fh, _fl)))

    def __most_probable_outcome(self, X_test, idx):
        """
        This function
        try each of the outcomes in __possible_outcome to see 
        which sequence generate the highest score. 
        """

        # Calculate the start and end indices
        previous_start = max(0, idx - self.ld)
        previous_end = max(0,idx)

        # Acquire test data features for these days
        previous_data = GHMM.__extract_features(X_test.iloc[previous_start:previous_end])

        # Append each outcome one by one with replacement to see which sequence generates
        # the highest score
        outcome_scores = np.array([], dtype=np.float32)

        for outcome in self._possible_outcomes:
            total_data = np.row_stack((previous_data, outcome))
            outcome_scores = np.append(outcome_scores, self.hmm.score(total_data))

        # Take the most probable outcome as the one with the highest score
        most_probable_outcome = self._possible_outcomes[np.argmax(outcome_scores)]

        return most_probable_outcome[0]

    def __close_price(self, X_test, idx):
        """
        The outcome from most_possible_outcome
        that generates the highest score is then used to make the
        prediction for that day's closing price
        """
        open = X_test.iloc[idx]['Open']
        fc = self.__most_probable_outcome(X_test, idx)

        return open * (1 + fc)

    def predict(self, X_test, plot=False):
        y_pred = np.array([],dtype=np.float32)

        for idx in tqdm(range(len(X_test))):
            y_pred = np.append(y_pred, self.__close_price(X_test, idx))
        
        if plot:
            days = np.array(X_test.index, dtype="datetime64[ms]")        
            fig = plt.figure()         
            axes = fig.add_subplot(111)
            axes.plot(days, X_test['Close'], 'bo-', label="actual")
            axes.plot(days, y_pred, 'r+-', label="predicted")          
            fig.autofmt_xdate()
            plt.legend()

        return y_pred
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
