import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from StockPredictorHMM import StockRegressorHMM

def evaluate(input, prediction):
    errors = abs(prediction - input)
    mape = 100 * np.mean(errors / input)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

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
                        default="../stock/",
                    )     

    cli_args = parser.parse_args()
    stock_predictor = StockRegressorHMM(hs=cli_args.hidden_states,
                                        ld=cli_args.latency_days,
                                        fc=cli_args.frac_change,
                                        fh=cli_args.frac_high,
                                        fl=cli_args.frac_low)

    data = pd.read_csv(cli_args.folder + cli_args.stock + '.csv')
    train_data, test_data = train_test_split(
    data, test_size=cli_args.test_size, shuffle=False)
    y_test = test_data['Close']
    stock_predictor.fit(X_train=train_data)
    y_pred = stock_predictor.predict(X_test=test_data, plot=cli_args.plot)
    evaluate(y_test, y_pred)

if __name__ == "__main__":
    main()