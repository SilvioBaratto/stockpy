Dataset
=======

This module encompasses classes and functions designed for efficient and flexible data management.


Dataset Integration
-------------------

- The :class:`~torch.utils.data.Dataset` in PyTorch acts as a data container, requiring implementation of `__len__()` and `__getitem__(<int>)`.
- The :class:`~torch.utils.data.DataLoader` manages batch processing, including sampling, shuffling, and parallelization.

stockpy uses the PyTorch :class:`~torch.utils.data.DataLoader`\s by default.
stockpy supports PyTorch's :class:`~torch.utils.data.Dataset` when calling
:func:`~stockpy.base.BaseEstimator.fit` or 
:func:`~stockpy.base.BaseEstimator.partial_fit`. Details on how to use PyTorch's
In order to support other data formats, we provide our own
:class:`.StockpyDataset` class that is compatible with:

- :class:`numpy.ndarray`\s
- PyTorch :class:`~torch.Tensor`\s
- scipy sparse CSR matrices
- pandas DataFrames or Series

Note that currently, sparse matrices are cast to dense arrays during
batching, given that PyTorch support for sparse matrices is still very
incomplete. If you would like to prevent that, you need to override
the ``transform`` method of :class:`~torch.utils.data.Dataset`.

In addition to the types above, you can pass dictionaries or lists of
one of those data types, e.g. a dictionary of
:class:`numpy.ndarray`\s. When you pass dictionaries, the keys of the
dictionaries are used as the argument name for the
:meth:`~torch.nn.Module.forward` method of the net's
``module``. Similarly, the column names of pandas ``DataFrame``\s are
used as argument names. 

Note that the keys in the dictionary ``X`` exactly match the argument
names in the :meth:`~torch.nn.Module.forward` method. This way, you
can easily work with several different types of input features.

The :class:`.StockpyDataset` from stockpy makes the assumption that you always
have an ``X`` and a ``y``, where ``X`` represents the input data and
``y`` the target. However, you may leave ``y=None``, in which case
:class:`.StockpyDataset` returns a dummy variable.

:class:`.StockpyDataset` applies a transform final transform on the data
before passing it on to the PyTorch
:class:`~torch.utils.data.DataLoader`. By default, it replaces ``y``
by a dummy variable in case it is ``None``. If you would like to
apply your own transformation on the data, you should subclass
:class:`.StockpyDataset` and override the
:func:`~skorch.dataset.Dataset.transform` method, then pass your
custom class to :class:`.NeuralNet` as the ``dataset`` argument.

ValidSplit
----------

This class is responsible for performing the :class:`.BaseEstimator`\'s
internal cross validation. For this, it sticks closely to the sklearn
standards. For more information on how sklearn handles cross
validation, look `here
<http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators>`_.

The first argument that :class:`.ValidSplit` takes is ``cv``. It works
analogously to the ``cv`` argument from sklearn
:class:`~sklearn.model_selection.GridSearchCV`,
:func:`~sklearn.model_selection.cross_val_score`, etc. For those not
familiar, here is a short explanation of what you may pass:

- ``None``: Use the default 3-fold cross validation.
- integer: Specifies the number of folds in a ``(Stratified)KFold``,
- float: Represents the proportion of the dataset to include in the
  validation split (e.g. ``0.2`` for 20%).
- An object to be used as a cross-validation generator.
- An iterable yielding train, validation splits.

Furthermore, :class:`.ValidSplit` takes a ``stratified`` argument that
determines whether a stratified split should be made (only makes sense
for discrete targets), and a ``random_state`` argument, which is used
in case the cross validation split has a random component.

One difference to sklearn\'s cross validation is that skorch
makes only a single split. In sklearn, you would expect that in a
5-fold cross validation, the model is trained 5 times on the different
combination of folds. This is often not desirable for neural networks,
since training takes a lot of time. Therefore, skorch only ever
makes one split.

If you would like to have all splits, you can still use skorch in
conjunction with the sklearn functions, as you would do with any
other sklearn\-compatible estimator. Just remember to set
``train_split=None``, so that the whole dataset is used for
training.