Core Functions
===============

The foundational components of StockPy are encapsulated within the :class:`.BaseEstimator` class and its subclasses, providing a seamless integration of Scikit-learn's interface with the capabilities of PyTorch's :class:`~torch.nn.Module`.

To harness these estimators, one would utilize methods reminiscent of Scikit-learn, most notably :func:`~stockpy.base.BaseEstimator.fit` and :func:`~stockpy.base.BaseEstimator.predict`. An illustrative example would be:

.. code:: python

    from stockpy.estimators import MLPRegressor

    # Instantiate the MLP Regressor
    mlp_regressor = MLPRegressor()

    # Fit the model on the training data
    mlp_regressor.fit(X_train, y_train)

    # Predict outcomes on the validation set
    y_predicted = mlp_regressor.predict(X_valid)

StockPy automates and simplifies several processes, such as:

- Converting :class:`numpy.ndarray` objects to :class:`~torch.Tensor` when necessary.
- Abstracting the training loop to minimize manual coding.
- Managing data batching, streamlining an otherwise complex operation.

This design significantly reduces boilerplate code, freeing the user to focus on the analytical aspects of machine learning. Despite its simplicity, StockPy is designed to be highly extensible and unobtrusive, allowing for straightforward customization for advanced use cases.


Key Arguments and Methods
-------------------------

For a detailed exposition of all the arguments and methods of :class:`.BaseEstimator`, please refer to the StockPy API documentation. This section highlights the principal ones.

optimizer
^^^^^^^^^

The optimizer should be a valid PyTorch or Pyro optimizer, such as :class:`~torch.optim.Adam` or :class:`~pyro.optim.Adam`. Post-initialization, the ``optimizer_`` attribute holds the initialized optimizer.

batch_size
^^^^^^^^^^

The ``batch_size`` argument controls the size for both training and validation iterators. Setting ``batch_size=128`` is equivalent to setting ``iterator_train__batch_size=128`` and ``iterator_valid__batch_size=128``. Specific batch size settings for training or validation will override this general setting.

train_split
^^^^^^^^^^^

The train_split function in :class:`.NeuralNet` governs the internal train/validation data split, typically reserving 20% of data for validation by default. Setting it to ``None`` allocates all data to training, omitting validation.

To safeguard against potential information leakage into the training set, consider employing an independent validation set, especially if utilizing callbacks like :class:`~stockpy.callbacks.EarlyStopping`.

Callback Implementation
^^^^^^^^^^^^^^^^^^^^^^^

By default, :class:`.BaseEstimator` and its subclasses are equipped with a selection of useful callbacks, specified within the :func:`~stockpy.base.BaseEstimator.get_default_callbacks` method. Users can add their own callbacks, which are invoked post-default callbacks except for :class:`~stockpy.callbacks.PrintLog`, which is called last.

.. code:: python

    predictor.fit(
        X_train, 
        y_train, 
        batch_size=32, 
        lr=0.01, 
        optimizer=torch.optim.Adam,
        callbacks=[
            MyCallback1(...),
            MyCallback2(...),
        ],
    )

warm_start
^^^^^^^^^^

The ``warm_start`` argument decides if subsequent calls to :func:`~stockpy.base.BaseEstimator.fit` reinitialize the :class:`.BaseEstimator`. By default, a new call to ``fit()`` resets the model parameters, discarding prior training. When ``warm_start=True``, training continues from the last state.

device
^^^^^^

The ``device`` setting dictates the computation device, ``'cuda'`` for GPU acceleration, or ``'cpu'`` for the central processing unit. Disabling device management by StockPy is possible with ``device=None``.

fit(X, y)
^^^^^^^^^

The ``fit`` method encompasses the complete model training process. It assumes X as input data and y as the target. For memory-intensive datasets, ``partial_fit`` can be used to incrementally train the model.

predict(X) and predict_proba(X)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These methods perform inference and return numpy.ndarray results. The ``predict_proba`` method yields the output of the forward method of the module, while ``predict`` is specific to classifiers and tries to provide the class labels

score(X, y)
^^^^^^^^^^^

This method is model-specific, providing accuracy for classifiers and R^2 score for regressors.

Input Data
^^^^^^^^^^

``StockPy`` supports various input types including numpy arrays, torch tensors, scipy sparse CSR matrices, and pandas DataFrames



