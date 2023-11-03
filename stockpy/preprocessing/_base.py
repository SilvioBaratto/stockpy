import warnings
from collections.abc import Mapping
from functools import partial
from numbers import Number

import numpy as np
from scipy import sparse
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import check_cv
import torch
import torch.utils.data
from torch.utils.data import Dataset

from stockpy.utils import flatten
from stockpy.utils import is_pandas_ndframe
from stockpy.utils import check_indexing
from stockpy.utils import to_numpy
import stockpy

def _apply_to_data(data, func, unpack_dict=False):
    """Apply a function to data, trying to unpack different data
    types.

    """
    apply_ = partial(_apply_to_data, func=func, unpack_dict=unpack_dict)

    if isinstance(data, Mapping):
        if unpack_dict:
            return [apply_(v) for v in data.values()]
        return {k: apply_(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        try:
            # e.g.list/tuple of arrays
            return [apply_(x) for x in data]
        except TypeError:
            return func(data)

    return func(data)

def _is_sparse(x):
    try:
        return sparse.issparse(x) or x.is_sparse
    except AttributeError:
        return False

def _len(x):
    if _is_sparse(x):
        return x.shape[0]
    return len(x)

def get_len(data):
    if isinstance(data, Mapping) and (data.get('input_ids') is not None):
        # Special casing Huggingface BatchEncodings because they are lists of
        # lists and thus their length would be determined incorrectly, returning
        # the sequence length instead of the number of samples.
        return len(data['input_ids'])
    lens = [_apply_to_data(data, _len, unpack_dict=True)]
    lens = list(flatten(lens))
    len_set = set(lens)
    if len(len_set) != 1:
        raise ValueError("Dataset does not have consistent lengths.")
    return list(len_set)[0]

class StockpyDataset(Dataset):
    """
    General dataset wrapper for use with PyTorch's DataLoader.

    This class is a general-purpose dataset wrapper that can be used in
    conjunction with PyTorch's DataLoader. The dataset always yields a tuple
    of two values: first the input data (X) and then the target (y).
    The target can be None, in which case a dummy tensor is returned
    since DataLoader doesn't work well with None.

    Attributes
    ----------
    X : various types
        Input data. Supported types include numpy arrays, PyTorch tensors,
        scipy sparse CSR matrices, pandas NDFrame, dictionaries of the
        aforementioned types, and lists/tuples of the aforementioned types.
    y : various types or None
        Target data similar to X, or None if not applicable.
    length : int, optional
        Manually specify the length of the dataset.

    Methods
    -------
    __len__()
        Returns the length of the dataset.
    transform(X, y)
        Performs additional transformations on X and y.
    __getitem__(i)
        Returns the i-th item from the dataset.

    """

    def __init__(self, X, y=None, length=None):
        """
        Initialize the dataset object.

        Parameters
        ----------
        X : various types
            Input data as described in class attributes.
        y : various types or None, optional
            Target data or None.
        length : int or None, optional
            Manually specify the length of the dataset.

        """
        self.X = X
        self.y = y

        # Check the indexing type of X and y.
        self.X_indexing = check_indexing(X)
        self.y_indexing = check_indexing(y)

        # Check if X is a pandas NDFrame.
        self.X_is_ndframe = is_pandas_ndframe(X)

        if length is not None:
            self._len = length
            return

        # Calculate the length of X.
        len_X = get_len(X)
        if y is not None:
            len_y = get_len(y)
            if len_y != len_X:
                raise ValueError("X and y have inconsistent lengths.")
        self._len = len_X

    def __len__(self):
        """
        Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return self._len

    def transform(self, X, y):
        """
        Perform additional transformations on X and y.

        By default, X and y are cast to PyTorch tensors. Override this method
        for custom behavior. When used with DataLoader, this method is
        called separately for each row of the batch.

        Parameters
        ----------
        X : various types
            The input data for the specific index.
        y : various types or None
            The target data for the specific index, or None if not applicable.

        Returns
        -------
        tuple
            The transformed X and y as a tuple.

        """
        # DataLoader can't handle None, so we use a tensor with value 0 as a placeholder.
        y = torch.Tensor([0]) if y is None else y

        # PyTorch doesn't work well with sparse matrices, make it dense.
        if sparse.issparse(X):
            X = X.toarray().squeeze(0)
        
        return X, y
        
class ValidSplit:
    """Class to perform internal train/validation split on a dataset.
    
    This class provides an abstraction over the internal validation split 
    of the dataset. It accepts different types of ``cv`` arguments, 
    similar to scikit-learn's cross-validation options.
    
    .. note::
       Only the first split is used. For full cycle splits, 
       external cross-validation methods should be used.
    
    Parameters
    ----------
    cv : int, float, cross-validation generator or iterable, optional
        Determines the cross-validation splitting strategy. 
        Options include:
        
        - None: For default 3-fold cross-validation.
        - int: Number of folds in a (Stratified)KFold.
        - float: Proportion of dataset to include in the validation split.
        - object: A cross-validation generator.
        - iterable: An iterable yielding train, validation splits.
        
    stratified : bool, optional (default=False)
        Indicates whether the split should be stratified. 
        Applicable only for binary or multiclass classification problems.
        
    random_state : int, RandomState instance, or None, optional (default=None)
        Controls the random state when ``(Stratified)ShuffleSplit`` is used. 
        This is applicable when a float value is passed to ``cv``.
    
    Attributes
    ----------
    stratified : bool
        Whether the data split should be stratified.
        
    random_state : int, RandomState instance, or None
        The random state used for shuffling.
    
    """
    def __init__(
            self,
            cv=5,
            stratified=False,
            random_state=None,
    ):
        """Initialize the ValidSplit object.
        
        Parameters are stored as attributes for later use.
        
        """
        self.stratified = stratified  # Whether the split should be stratified
        self.random_state = random_state  # Random state for shuffling

        if isinstance(cv, Number) and (cv <= 0):
            raise ValueError("Numbers less than 0 are not allowed for cv "
                             "but ValidSplit got {}".format(cv))

        if not self._is_float(cv) and random_state is not None:
            raise ValueError(
                "Setting a random_state has no effect since cv is not a float. "
                "You should leave random_state to its default (None), or set cv "
                "to a float value.",
            )

        self.cv = cv

    def _is_stratified(self, cv):
        return isinstance(cv, (StratifiedKFold, StratifiedShuffleSplit))

    def _is_float(self, x):
        if not isinstance(x, Number):
            return False
        return not float(x).is_integer()

    def _check_cv_float(self):
        cv_cls = StratifiedShuffleSplit if self.stratified else ShuffleSplit
        return cv_cls(test_size=self.cv, random_state=self.random_state)

    def _check_cv_non_float(self, y):
        return check_cv(
            self.cv,
            y=y,
            classifier=self.stratified,
        )

    def check_cv(self, y):
        """Resolve which cross validation strategy is used."""
        y_arr = None
        if self.stratified:
            # Try to convert y to numpy for sklearn's check_cv; if conversion
            # doesn't work, still try.
            try:
                y_arr = to_numpy(y)
            except (AttributeError, TypeError):
                y_arr = y

        if self._is_float(self.cv):
            return self._check_cv_float()
        return self._check_cv_non_float(y_arr)

    def _is_regular(self, x):
        return (x is None) or isinstance(x, np.ndarray) or is_pandas_ndframe(x)

    def __call__(self, dataset, y=None, groups=None):
        bad_y_error = ValueError(
            "Stratified CV requires explicitly passing a suitable y.")
        if (y is None) and self.stratified:
            raise bad_y_error

        cv = self.check_cv(y)
        if self.stratified and not self._is_stratified(cv):
            raise bad_y_error

        # pylint: disable=invalid-name
        len_dataset = get_len(dataset)
        if y is not None:
            len_y = get_len(y)
            if len_dataset != len_y:
                raise ValueError("Cannot perform a CV split if dataset and y "
                                 "have different lengths.")

        args = (np.arange(len_dataset),)
        if self._is_stratified(cv):
            args = args + (to_numpy(y),)

        idx_train, idx_valid = next(iter(cv.split(*args, groups=groups)))
        dataset_train = torch.utils.data.Subset(dataset, idx_train)
        dataset_valid = torch.utils.data.Subset(dataset, idx_valid)
        return dataset_train, dataset_valid

    def __repr__(self):
        # pylint: disable=useless-super-delegation
        return super(ValidSplit, self).__repr__()