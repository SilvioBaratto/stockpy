Getting Started
===============

Training a Forecaster
---------------------

This quickstart guide provides a concise introduction to using ``stockpy`` for
time-series forecasting with encoder-decoder neural architectures.

Forecasting Example
~~~~~~~~~~~~~~~~~~~

The example below trains an :class:`~stockpy.forecasters.LSTMForecaster` to
predict the next ``pred_len`` time steps of a multivariate series from a fixed
``context_len`` window.

.. code:: python

   import numpy as np
   import torch
   from stockpy.forecasters import LSTMForecaster

   # Build a multivariate time series of shape (T, n_features).
   T, n_features = 500, 4
   series = np.random.default_rng(0).standard_normal((T, n_features)).astype(np.float32)

   # Initialize the forecaster: read the last 20 steps, predict the next 5.
   forecaster = LSTMForecaster(
       context_len=20,
       pred_len=5,
       rnn_size=32,
       hidden_size=32,
       max_epochs=20,
       lr=1e-3,
       optimizer=torch.optim.Adam,
   )

   forecaster.fit(series)

   # Predict from the most recent context window. Output shape:
   # (n_samples, pred_len, n_features).
   y_pred = forecaster.predict(series[-20:][None, :, :])
   print(y_pred.shape)

What's Next?
------------

To delve deeper into the functionality and capabilities of ``stockpy``, please
visit the :ref:`tutorials` page. There you'll find additional examples and
comprehensive guides covering the different encoder-decoder forecasters
(LSTM, GRU, BiLSTM, BiGRU, TCN, DMM, Transformer) and the preprocessing
utilities tailored to time-series forecasting.
