Getting Started
===============

Training a Model
----------------

This quickstart guide provides a concise introduction to using the ``stockpy`` library for machine learning tasks in the stock market domain.

Below are examples to help you get started with both regression and classification models using ``stockpy``.

Regression Example
~~~~~~~~~~~~~~~~~~

In this regression example, we'll predict the closing price of Apple Inc. (AAPL) stock using historical data.

.. code:: python

   from stockpy.neural_network import LSTMRegressor
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   import torch

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

   # Initialize and fit the model
   predictor = LSTMRegressor(hidden_size=32)
   predictor.fit(X_train, y_train, batch_size=32, lr=0.01, optimizer=torch.optim.Adam, epochs=50)

Classification Example
~~~~~~~~~~~~~~~~~~~~~~

For classification, this example will use synthetic data to categorize inputs into one of five classes.

.. code:: python

   from stockpy.neural_network import LSTMClassifier
   import numpy as np
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import MinMaxScaler
   import torch

   # Generate synthetic data
   X, y = make_classification(n_samples=10000, 
                              n_features=20, 
                              n_informative=15, 
                              n_redundant=5, 
                              n_classes=5, 
                              random_state=0)

   # Split the dataset
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)

   # Scale the data and convert to torch tensors
   scaler = MinMaxScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)

   X_train = torch.tensor(X_train, dtype=torch.float)
   y_train = torch.tensor(y_train, dtype=torch.int64)

   # Initialize and fit the model
   predictor = LSTMClassifier(hidden_size=32)
   predictor.fit(X_train, y_train, batch_size=32, lr=0.01, optimizer=torch.optim.Adam, epochs=50)

What's Next?
------------

To delve deeper into the functionality and capabilities of ``stockpy``, please visit the :ref:`tutorials` page. There you'll find additional examples and comprehensive guides for different models and data preprocessing techniques tailored to stock market analysis.


