import argparse
import os
import shutil
import sys
import glob
sys.path.append("..")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from neural_network.LSTM import LSTM
from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw
import matplotlib.pyplot as plt

# set style of graphs
plt.style.use('ggplot')
from pylab import rcParams
plt.rcParams['figure.dpi'] = 100

def evaluate(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred, squared=True)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    # dtw_distance = fastdtw(y_test, y_pred, dist=euclidean)[0] 
    print('Model Performance')
    print("Mean squared error = {:0.3f}".format(mse))
    print("Root mean squared error = {:0.3f}".format(rmse))
    print('Mean absolute percentage error = {:0.3f}%.'.format((mape)*100))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--stock',
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

    parser.add_argument('--hidden-size',
                        help="Number of neural network ",
                        action="store",
                        default=32,
                        type=int,
                    )

    parser.add_argument('--num-layers',
                        help="Number of layers",
                        action="store",
                        default=1,
                        type=int,
                    )
    parser.add_argument('--batch-size',
                        help="Number of forecasted days",
                        action="store",
                        default=16,
                        type=int,
                    )

    parser.add_argument('--epochs',
                        help="Number of latency days",
                        action="store",
                        default=10,
                        type=int,
                    )

    parser.add_argument('--plot',
                        help="Number of latency days",
                        action="store_true",
                    )
    parser.add_argument('--folder',
                        help="Number of latency days",
                        action="store",
                        default="../../stock/",
                    )     

    cli_args = parser.parse_args()
    stock_predictor = LSTM(hidden_size=cli_args.hidden_size,
                           num_layers=cli_args.num_layers)

    data = pd.read_csv(cli_args.folder + cli_args.stock.upper() + '.csv',
                        parse_dates=True, index_col='Date').dropna(how="any")

    X_train, X_test = train_test_split(data, 
                                test_size=cli_args.test_size, 
                                shuffle=False)

    stock_predictor.fit(X_train, 
                        batch_size=cli_args.batch_size, 
                        epochs=cli_args.epochs)

    y_test = X_test['Close']

    y_pred = stock_predictor.predict(x_test=X_test, plot=cli_args.plot)
    evaluate(y_test, y_pred)

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
