## Table of contents
* [Description](#description)
* [Documentation](#documentation)
* [Testing](#testing)
* [Examples and Tutorials](#examples-and-tutorials)
* [How to contribute](#how-to-contribute)
* [License](#license)

## Description
**stockpy** is a Python Machine Learning library to detect relevant trading patterns and make investmen predictions. At the moment it supports the following algorithms:

- Long Short Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Multilayer Perceptron (MLP)
- Gaussian Hidden Markov Models (GHMM)
- Bayesian Neural Networks (BNN)
- Deep Markov Model (DMM)

## Documentation
In order to test the library we can perform the following code 
```Python
from sklearn.model_selection import train_test_split
import pandas as pd
from stockpy.neural_network import MLP, GRU, BiLSTM, BiGRU, LSTM


df = pd.read_csv('../../stock/AAPL.csv', parse_dates=True, index_col='Date').dropna(how="any")
X_train, X_test = train_test_split(df, test_size=0.1, shuffle=False)

predictor = LSTM()
predictor.fit(X_train, batch_size=24, epochs=10)
y_pred = predictor.predict(X_test, plot=False)
```
The given code imports different types of neural network models from the stockpy.neural_network module. The imported model in this case is LSTM.

After that, the code reads a CSV file 'AAPL.csv' containing stock market data for Apple (AAPL) using the pandas library. The code then creates an instance of the LSTM model and fits it to the training data using the fit method.  Finally, the code uses the predict method of the LSTM model to make predictions on the test data. Overall, this code reads stock market data for a company named Apple, splits the data into training and testing sets, fits an LSTM model to the training data, and uses the model to make predictions on the test data. 
## Testing

## How to contribute

## License