<div align="center">
  <a href="https://github.com/SilvioBaratto/stockpy"> <img width=600 src="docs/source/_static/img/stockpi_v3.svg"></a>
</div>

![Python package](https://github.com/SilvioBaratto/stockpy/workflows/Python%20package/badge.svg?branch=master)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
<img src='https://img.shields.io/badge/Code%20style-Black-%23000000'/>
[![Documentation Status](https://readthedocs.org/projects/stockpy/badge/?version=latest)](https://stockpy.readthedocs.io/?badge=latest)
[![PyPI version](https://badge.fury.io/py/stockpy-learn.svg)](https://badge.fury.io/py/stockpy-learn)

## Table of Contents
* [Description](#description)
* [Documentation](https://stockpy.readthedocs.io/)
* [Installation](#installation)
* [Usage](#usage)
* [Examples](#examples)
* [Data Downloader](#data-downloader)
* [License](#license)
* [Contributing](#contributing)
* [TODOs](#todos)

## Description
**stockpy** is a versatile Python Machine Learning library initially designed for stock market data analysis and predictions. It has now evolved to handle a wider range of datasets, supporting tasks such as regression and classification. It currently supports the following algorithms, each with regression and classification implementations:

- Bayesian Neural Networks (BNN)
- Long Short Term Memory (LSTM)
- Bidirectional Long Short Term Memory (BiLSTM)
- Gated Recurrent Unit (GRU)
- Bidirectional Gated Recurrent Unit (BiGRU)
- Multilayer Perceptron (MLP)
- Deep Markov Model (DMM) 
- Gaussian Hidden Markov Models (GHMM) 

## Usage
To use **stockpy**, start by importing the relevant models from the `stockpy.neural_network` and `stockpy.probabilistic` modules. The library can be used with various types of input data, such as CSV files, pandas dataframes, numpy arrays and torch arrays.

Here's an example to demonstrate the usage of stockpy for regression. In this example, we read a CSV file containing stock market data for Apple (AAPL), split the data into training and testing sets, fit an LSTM model to the training data, and use the model to make predictions on the test data:

```Python
from stockpy.neural_network import CNNRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('stock/AAPL.csv', parse_dates=True, index_col='Date').dropna(how="any")

# Define features and target
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the data to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
y_train = torch.tensor(y_train.values, dtype=torch.float)

# Fit the model
predictor = CNNRegressor(hidden_size=32)

predictor.fit(X_train, 
              y_train, 
              batch_size=32, 
              lr=0.01, 
              optimizer=torch.optim.Adam, 
              epochs=50)
```

Here's an example to demonstrate the usage of stockpy for classification. In this example, we read a pickle file containing labeled data, split the data into training and testing sets, fit an LSTM model to the training data, and use the model to make classification on the test data:

```Python
from stockpy.neural_network import LSTMClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=10000, 
                           n_features=20, 
                           n_informative=15, 
                           n_redundant=5, 
                           n_classes=5, 
                           random_state=0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)

# Scale the data and convert to torch tensors
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.int64)

predictor = LSTMClassifier()

predictor.fit(X_train, 
              y_train, 
              batch_size=32, 
              lr=0.01, 
              optimizer=torch.optim.Adam, 
              epochs=50)
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
> pip install .
```

## TODOs
Below is a list of planned enhancements and features that are in the pipeline for **stockpy**. Contributions and suggestions are always welcome!

- [ ] Implement a dedicated `test` directory with comprehensive unit tests to ensure reliability and facilitate continuous integration.
- [ ] Expand the documentation to include more detailed tutorials and code explanations, aiding users in effectively utilizing **stockpy**.
- [ ] Enrich the algorithmic suite by adding additional models for regression and classification, catering to a broader range of data science needs.
- [ ] Integrate generative models into the library to provide advanced capabilities for data synthesis and pattern discovery.
- [ ] Develop and incorporate sophisticated prediction models that can handle complex forecasting tasks with higher accuracy.

*Note: A checked box (✅) indicates that the task has been completed.*

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