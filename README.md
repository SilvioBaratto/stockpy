## Table of contents
* [Description](#description)
* [Documentation](#documentation)
* [Data Downloader](#data-downloader)
* [Testing](#testing)
* [How to contribute](#how-to-contribute)
* [License](#license)

## Description
**stockpy** is a Python Machine Learning library to detect relevant trading patterns and make investmen predictions. At the moment it supports the following algorithms:

- Long Short Term Memory (LSTM)
- Bidirectional Long Short Term Memory (BiLSTM)
- Gated Recurrent Unit (GRU)
- Bidirectional Gated Recurrent Unit (BiGRU)
- Multilayer Perceptron (MLP)
- Gaussian Hidden Markov Models (GHMM)
- Bayesian Neural Networks (BNN)
- Deep Markov Model (DMM)

## Documentation
In order to test the library we can perform the following code 
```Python
from sklearn.model_selection import train_test_split
import pandas as pd
from stockpy.neural_network import LSTM


df = pd.read_csv('../../stock/AAPL.csv', parse_dates=True, index_col='Date').dropna(how="any")
X_train, X_test = train_test_split(df, test_size=0.1, shuffle=False)

predictor = LSTM()
predictor.fit(X_train, batch_size=24, epochs=10)
y_pred = predictor.predict(X_test, plot=False)
```

This code reads a CSV file 'AAPL.csv' containing stock market data for Apple (AAPL) using the pandas library. The code then creates an instance of the LSTM model and fits it to the training data using the fit method.  Finally, the code uses the predict method of the LSTM model to make predictions on the test data. Overall, this code reads stock market data for a company named Apple, splits the data into training and testing sets, fits an LSTM model to the training data, and uses the model to make predictions on the test data. 

The following structure can be applied to all models in the library, just make sure to import from the correct location.
```Python
from stockpy.neural_network import LSTM, GRU, BiLSTM, BiGRU, MLP
from stockpy.probabilistic import BNN, DMM
from stockpy.ensemble import RF, XGB
from stockpy.linear_model import Lasso, Linear, Quantile, Ridge, SGD, SVR
```
## Data downloader
This is a terminal application located on the cmd folder under the name of data.py and has been tested and verified using Ubuntu 22.04 LST. Below the documentation to explain the available commands:

| Parameter       | Explanation
|-----------------|-------------------------------------|
| `--download`| This command download all the S&P 500 stocks. If no start and end dates are specified, the default range is between ”2017-01-01” and today’s date. from ”2017-01-01” to the actual day.                |
| `--stock`| This command download a specific stock specified by the user. If no start and end dates are specified, the default range is between ”2017-01-01” and today’s date. from ”2017-01-01” to the actual day.                |
| `--update`| This command update all the stocks presents in the folder containing the files. It is possible to update the files to any range of dates. If the stocks wasn’t listed before a specific date for default it will downloaded from the day it enters in the public market. |
|`--update.stock`| This command update a specific stock specified by the user. It is possible to update the files to any range of dates specifying the start and end. |
|`--start`| This command specify the start date. |
|`--end`| This command specify the end date. |
|`--delete`| Delete all files present in the files folder. | 
|`--delete-stock`| Delete a specific stock present in the files folder. | 
|`--folder`| Choose the folder where to read or download all the files. |

```Python
# Download all the data chossing a specific range
python3 data.py --download --start="2017-01-01" --end="2018-01-01"
# To download data from apple from "2017-01-01" to today
python3 data_app.py --stock="AAPL" --end="today"
# Update all the data choosing a specific range
python3 data.py --download --start="2014-01-01" --end="2020-01-01"
# Update a specific stock from a specific start
# untill the last days present in the stock file
python3 data.py --update-stock --start="2014-01-01"
# Download all the data chossing a specific range and folder,
# start and end default values are "2017-01-01" and today
python3 data.py --download --folder="../../example"
```
## License

See the [LICENSE](LICENSE) file for license rights and limitations (MIT).