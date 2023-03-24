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
from probabilistic.GHMM import GHMM
from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw

def evaluate(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred, squared=True)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    # dtw_distance = fastdtw(y_test, y_pred, dist=euclidean)[0] 
    print('Model Performance')
    print("Mean squared error = {:0.3f}".format(mse))
    print("Root mean squared error = {:0.3f}".format(rmse))
    print('Mean absolute percentage error = {:0.3f}%.'.format((mape)*100))
    # print('DTW distance = {:0.3f} '.format(dtw_distance))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--stock',
                        help="Stock from S%P",
                        action="store",
                        default=None,
                    )
    parser.add_argument('--forecast',
                        help="Test size training",
                        action="store",
                        default=0.3,
                        type=float,
                    )

    parser.add_argument('--hidden-states',
                        help="Number of hidden states",
                        action="store",
                        default=4,
                        type=int,
                    )

    parser.add_argument('--latency-days',
                        help="Number of latency days",
                        action="store",
                        default=10,
                        type=int,
                    )
    parser.add_argument('--forecasting',
                        help="Number of forecasted days",
                        action="store",
                        default=50,
                        type=int,
                    )

    parser.add_argument('--frac-change',
                        help="Number of latency days",
                        action="store",
                        default=50,
                        type=int,
                    )

    parser.add_argument('--frac-high',
                        help="Number of latency days",
                        action="store",
                        default=10,
                        type=int,
                    )

    parser.add_argument('--frac-low',
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
    stock_predictor = GHMM(hs=cli_args.hidden_states,
                                        ld=cli_args.latency_days,
                                        fc=cli_args.frac_change,
                                        fh=cli_args.frac_high,
                                        fl=cli_args.frac_low)

    df = pd.read_csv(cli_args.folder + cli_args.stock.upper() + '.csv')
    X_train = df[:len(df)-cli_args.forecast]
    X_test = df[-cli_args.forecast:]
    stock_predictor.fit(X_train=X_train)
    y_test = X_test['Close']
    y_pred = stock_predictor.predict(X_test=X_test, plot=cli_args.plot)
    evaluate(y_test, y_pred)

if __name__ == "__main__":
    main()
