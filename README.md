## Table of contents
* [Description](#description)
* [Documentation](#documentation)
* [Data Downloader](#data-downloader)
* [License](#license)

## Description
**stockpy** is a Python Machine Learning library designed to facilitate stock market data analysis and predictions. It currently supports the following algorithms:

- Long Short Term Memory (LSTM)
- Bidirectional Long Short Term Memory (BiLSTM)
- Gated Recurrent Unit (GRU)
- Bidirectional Gated Recurrent Unit (BiGRU)
- Multilayer Perceptron (MLP)
- Gaussian Hidden Markov Models (GaussianHMM)
- Bayesian Neural Networks (BayesianNN)
- Deep Markov Model (DeepMarkovModel)

**stockpy** can be used to perform a range of tasks such as detecting relevant trading patterns, making predictions and generating trading signals.

## Usage
To use **stockpy** and perform predictions on stock market data, start by importing the relevant models from the `stockpy.neural_network` and `stockpy.probabilistic` modules. The library can be used with various types of input data, such as CSV files, pandas dataframes and numpy arrays.

To demonstrate the usage of stockpy, we can perform the following code to read a CSV file containing stock market data for Apple (AAPL), split the data into training and testing sets, fit an LSTM model to the training data, and use the model to make predictions on the test data:
```Python
from sklearn.model_selection import train_test_split
import pandas as pd
from stockpy.neural_network import LSTM

# read CSV file and drop missing values
df = pd.read_csv('AAPL.csv', parse_dates=True, index_col='Date').dropna(how="any")

# split data into training and testing sets
X_train, X_test = train_test_split(df, test_size=0.1, shuffle=False)

# create LSTM model instance and fit to training data
predictor = LSTM()
predictor.fit(X_train, batch_size=24, epochs=10)

# use LSTM model to make predictions on test data
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
> python setup.py install
```
## Data downloader
The data downloader is a command-line application located in the cmd folder under the name of data.py, which can be used to download and update stock market data. The downloader has been tested and verified using Ubuntu 22.04 LTS.

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