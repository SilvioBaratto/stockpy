<p align="center">
  <a href="https://github.com/SilvioBaratto/stockpy" target="_blank" >
    <img alt="stockpy" src="docs/source/_static/img/logo.png" width="400" />
  </a>
</p>

![Python package](https://github.com/SilvioBaratto/stockpy/workflows/Python%20package/badge.svg?branch=master)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
<img src='https://img.shields.io/badge/Code%20style-Black-%23000000'/>
[![Documentation Status](https://readthedocs.org/projects/bispy-bisimulation-in-python/badge/?version=latest)](https://bispy-bisimulation-in-python.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/stockpy.svg)](https://badge.fury.io/py/stockpy)

## Table of contents
* [Description](#description)
* [Documentation](#documentation)
* [Data Downloader](#data-downloader)
* [License](#license)

## Description
**stockpy** is a Python Machine Learning library designed to facilitate stock market data analysis and predictions. It currently supports the following algorithms:

- Deep Markov Model (DeepMarkovModel)
- Gaussian Hidden Markov Models (GaussianHMM)
- Bayesian Neural Networks (BayesianNN)
- Long Short Term Memory (LSTM)
- Bidirectional Long Short Term Memory (BiLSTM)
- Gated Recurrent Unit (GRU)
- Bidirectional Gated Recurrent Unit (BiGRU)
- Multilayer Perceptron (MLP)

**stockpy** can be used to perform a range of tasks such as detecting relevant trading patterns, making predictions and generating trading signals.

## Usage
To use **stockpy** and perform predictions on stock market data, start by importing the relevant models from the `stockpy.neural_network` and `stockpy.probabilistic` modules. The library can be used with various types of input data, such as CSV files, pandas dataframes and numpy arrays.

To demonstrate the usage of stockpy, we can perform the following code to read a CSV file containing stock market data for Apple (AAPL), split the data into training and testing sets, fit an LSTM model to the training data, and use the model to make predictions on the test data:
```Python
from sklearn.model_selection import train_test_split
import pandas as pd
from stockpy.probabilistic import DeepMarkovModelRegressor
from stockpy.neural_network import LSTMRegressor

# read CSV file and drop missing values
df = pd.read_csv('../stock/AAPL.csv', parse_dates=True, index_col='Date').dropna(how="any")

# split data into training and test set
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

# create model instance and fit to training data
predictor = DeepMarkovModelRegressor()
predictor.fit(X_train, y_train, batch_size=24, epochs=10)

# predictions on test data
y_pred = predictor.predict(X_test)
```
The above code can be applied to all models in the library, just make sure to import from the correct location, either `stockpy.neural_network` or `stockpy.probabilistic`.

## Dependencies and installation
**stockpy** requires the modules `numpy, torch, pyro-ppl`. The code is tested for _Python 3_. It can be installed using `pip` or directly from the source cod.

### Installing via pip

To install the package:
```bash
> pip install stockpy-learn
```
To uninstall the package:
```bash
> pip uninstall stockpy-learn
```
### Installing from source

You can clone this repository on your local machines using:

```bash
> git clone https://github.com/SilvioBaratto/stockpy
```

To install the package:

```bash
> cd stockpy
> ./install.sh
```
## Data downloader
The data downloader is a command-line application located named `data.py`, which can be used to download and update stock market data. The downloader has been tested and verified using Ubuntu 22.04 LTS.

| Parameter       | Explanation
|-----------------|-------------------------------------|
| `--download`| Download all the S&P 500 stocks. If no start and end dates are specified, the default range is between "2017-01-01" and today's date.                |
| `--stock`| Download a specific stock specified by the user. If no start and end dates are specified, the default range is between "2017-01-01" and today's date.                |
| `--update`| Update all the stocks present in the folder containing the files. It is possible to update the files to any range of dates. If a stock wasn't listed before a specific date, it will be downloaded from the day it enters the public market. |
|`--update.stock`| Update a specific stock specified by the user. It is possible to update the files to any range of dates by specifying the start and end dates. |
|`--start`| Specify the start date for downloading or updating data. |
|`--end`| Specify the end date for downloading or updating data. |
|`--delete`| Delete all files present in the files folder. | 
|`--delete-stock`| Delete a specific stock present in the files folder. | 
|`--folder`| Choose the folder where to read or download all the files. |
### Usage example
Below are some examples of how to use the downloader:
```Python
# Download all the data between "2017-01-01" and "2018-01-01"
python3 data.py --download --start="2017-01-01" --end="2018-01-01"

# Download data for Apple (AAPL) from "2017-01-01" to today's date
python3 data.py --stock="AAPL" --end="today"

# Update all the data between "2014-01-01" and "2020-01-01"
python3 data.py --update --start="2014-01-01" --end="2020-01-01"

# Update a specific stock from "2014-01-01" until the last day present in the stock file
python3 data.py --update-stock --stock="AAPL" --start="2014-01-01"

# Download all the data between "2017-01-01" and today's date, 
# choosing the folder where to download the files
python3 data.py --download --folder="../../example"
```

## TODOS
- Implementing other functionalities as portfolio optimization
- Implement transformers
- Implement stockGPT to make long term predictions

## Authors and acknowledgements
**stockpy** is currently developed and mantained by **Silvio Baratto**. You can contact me at:
- silvio.baratto22 at gmail.com

## Reporting a bug
The best way to report a bug is using the
[Issues](https://github.com/fAndreuzzi/BisPy/issues) section. Please, be clear,
and give detailed examples on how to reproduce the bug (the best option would
be the graph which triggered the error you are reporting).

## How to contribute

We are more than happy to receive contributions on tests, documentation and
new features. Our [Issues](https://github.com/fAndreuzzi/BisPy/issues)
section is always full of things to do.

Here are the guidelines to submit a patch:

1. Start by opening a new [issue](https://github.com/fAndreuzzi/BisPy/issues)
   describing the bug you want to fix, or the feature you want to introduce.
   This lets us keep track of what is being done at the moment, and possibly
   avoid writing different solutions for the same problem.

2. Fork the project, and setup a **new** branch to work in (_fix-issue-22_, for
   instance). If you do not separate your work in different branches you may
   have a bad time when trying to push a pull request to fix a particular
   issue.

3. Run [black](https://github.com/psf/black) before pushing
   your code for review.

4. Provide menaningful **commit messages** to help us keeping a good _git_
   history.

5. Finally you can submbit your _pull request_!

## License

See the [LICENSE](LICENSE) file for license rights and limitations (MIT).

## stockpy Legal Disclaimer

Please read this legal disclaimer carefully before using stockpy-learn library. By using stockpy-learn library, you agree to be bound by this disclaimer.

stockpy-learn library is provided for informational and educational purposes only and is not intended as a recommendation, offer or solicitation for the purchase or sale of any financial instrument or securities. The information provided in the stockpy-learn library is not to be construed as financial, investment, legal, or tax advice, and the use of any information provided in stockpy-learn library is at your own risk.

stockpy-learn library is not a substitute for professional financial or investment advice and should not be relied upon for making investment decisions. You should consult a qualified financial or investment professional before making any investment decision.

We make no representation or warranty, express or implied, as to the accuracy, completeness, or suitability of any information provided in stockpy, and we shall not be liable for any errors or omissions in such information.

We shall not be liable for any direct, indirect, incidental, special, consequential, or exemplary damages arising from the use of stockpy library or any information provided therein.

stockpy-learn library is provided "as is" without warranty of any kind, either express or implied, including but not limited to the implied warranties of merchantability, fitness for a particular purpose, or non-infringement.

We reserve the right to modify or discontinue stockpy-learn library at any time without notice. We shall not be liable for any modification, suspension, or discontinuance of stockpy-learn library.

By using stockpy-learn library, you agree to indemnify and hold us harmless from any claim or demand, including reasonable attorneys' fees, made by any third party due to or arising out of your use of stockpy-learn library, your violation of this disclaimer, or your violation of any law or regulation.

This legal disclaimer is governed by and construed in accordance with the laws of Italy, and any disputes relating to this disclaimer shall be subject to the exclusive jurisdiction of the courts of Italy.

If you have any questions about this legal disclaimer, please contact us at silvio.baratto22@gmail.com.

By using stockpy-learn library, you acknowledge that you have read and understood this legal disclaimer and agree to be bound by its terms and conditions.