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

def _apply_to_data(data, func, unpack_dict=False):
    """
    Apply a given function to the input data.

    This function supports data in the form of mappings (like dictionaries),
    lists, tuples, or a single value. If the input is a mapping and `unpack_dict` 
    is True, it applies the function to the values of the dictionary.

    Parameters
    ----------
    data : Any
        The input data on which to apply the function. This can be a mapping, 
        list, tuple, or a single value.
    func : Callable
        The function to apply to the elements of `data`. This function must 
        take a single argument and return a value.
    unpack_dict : bool, optional
        Determines whether to apply the function to the values of a mapping or 
        not. If False, the function is applied to the entire item (key-value pair). 
        Default is False.

    Returns
    -------
    Any
        The result of applying `func` to `data`. The return type matches the 
        structure of `data`: if `data` is a dictionary, a dictionary is returned; 
        if a list or tuple, a list is returned; otherwise, a single value is returned.

    Examples
    --------
    >>> data = {'a': 1, 'b': 2}
    >>> func = lambda x: x * 2
    >>> _apply_to_data(data, func)
    {'a': 2, 'b': 4}

    >>> data = [1, 2, 3]
    >>> _apply_to_data(data, func)
    [2, 4, 6]

    >>> data = 5
    >>> _apply_to_data(data, func)
    10

    >>> data = {'a': [1, 2], 'b': [3, 4]}
    >>> _apply_to_data(data, func, unpack_dict=True)
    [[2, 4], [6, 8]]

    Notes
    -----
    - If `data` is a list or tuple containing types that `func` cannot handle, 
      a TypeError will be raised.
    - The function does not apply `func` to the keys of a mapping.
    """
    apply_ = partial(_apply_to_data, func=func, unpack_dict=unpack_dict)

    if isinstance(data, Mapping):
        if unpack_dict:
            return [apply_(v) for v in data.values()]
        return {k: apply_(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        try:
            return [apply_(x) for x in data]
        except TypeError:
            return func(data)

    return func(data)

def _is_sparse(x):
    """
    Check whether the input is a sparse matrix.

    This function checks if `x` is a sparse matrix using `scipy.sparse.issparse`
    and also checks if `x` has an attribute `is_sparse` which might be defined in
    custom sparse matrix classes.

    Parameters
    ----------
    x : Any
        The input data to check.

    Returns
    -------
    bool
        True if `x` is a sparse matrix, False otherwise.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> x = csr_matrix([0, 1, 2])
    >>> _is_sparse(x)
    True

    >>> x = [0, 1, 2]
    >>> _is_sparse(x)
    False
    """
    try:
        return sparse.issparse(x) or x.is_sparse
    except AttributeError:
        return False

def _len(x):
    """
    Get the length of the input data.

    If the input data `x` is a sparse matrix, it returns the first dimension size,
    otherwise, it returns the length of `x`.

    Parameters
    ----------
    x : Any
        The input data to get the length of. Can be a sparse matrix or any other
        object that supports the `len` function.

    Returns
    -------
    int
        The length of the first dimension if `x` is sparse, or the length of `x`.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> x = csr_matrix([[0, 1], [2, 3]])
    >>> _len(x)
    2

    >>> x = [0, 1, 2, 3]
    >>> _len(x)
    4
    """
    if _is_sparse(x):
        return x.shape[0]
    return len(x)

def get_len(data):
    """
    Get the consistent length of the input data.

    This function is particularly designed to work with datasets that may include
    batch encodings from Huggingface's tokenizers. It ensures that all elements in
    the dataset have the same length.

    Parameters
    ----------
    data : Mapping or other collections
        The input data for which to determine the length. This can be a dictionary
        that possibly includes a 'input_ids' key, which is common in Huggingface
        BatchEncodings, or any other collection type.

    Returns
    -------
    int
        The consistent length of the dataset.

    Raises
    ------
    ValueError
        If the elements of the dataset do not have the same length.

    Examples
    --------
    >>> data = {'input_ids': [[1, 2, 3], [4, 5, 6]], 'attention_mask': [[1, 1, 1], [1, 1, 1]]}
    >>> get_len(data)
    2

    >>> data = [[1, 2, 3], [4, 5, 6, 7]]
    >>> get_len(data)
    ValueError: Dataset does not have consistent lengths.
    """
    if isinstance(data, Mapping) and (data.get('input_ids') is not None):
        # Special casing for Huggingface BatchEncodings
        return len(data['input_ids'])

    lens = [_apply_to_data(data, _len, unpack_dict=True)]
    lens = list(flatten(lens))
    len_set = set(lens)

    if len(len_set) != 1:
        raise ValueError("Dataset does not have consistent lengths.")

    return list(len_set)[0]

class StockpyDataset(Dataset):
    """
    Custom dataset class for Stockpy designed to handle a variety of data types.

    The `StockpyDataset` class extends PyTorch's `Dataset` class, providing a flexible
    dataset wrapper that supports inputs such as numpy arrays, PyTorch tensors, scipy
    sparse matrices, pandas dataframes, and their dictionary or list aggregations. It
    can handle cases where the target data may not be applicable, returning a dummy tensor
    for compatibility with PyTorch's `DataLoader`.

    Parameters
    ----------
    X : various
        The input features; supported formats include numpy arrays, tensors, scipy sparse
        matrices, pandas dataframes, and collections of these types.
    y : various or None
        The target labels or values; supports the same formats as `X`. If None, the dataset
        yields a dummy tensor for `y`.
    length : int, optional
        The total number of samples in the dataset. If not specified, it is inferred from `X`
        and validated against `y` if `y` is provided.

    Methods
    -------
    __len__(self)
        Returns the total number of samples in the dataset.
    __getitem__(self, index)
        Retrieves the input-target pair at the specified index in the dataset.

    Raises
    ------
    ValueError
        If `length` is not provided and the inferred lengths of `X` and `y` do not match.
    """

    def __init__(self, X, y=None, length=None):
        """
        Initializes the `StockpyDataset`.
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
        Applies transformations to the data, converting it to the format required for model processing.

        This method is called for each item in the dataset before batching. It converts the data to PyTorch
        tensors and handles cases where the target `y` is None by providing a placeholder tensor. Override
        this method in subclasses to implement custom transformations.

        Parameters
        ----------
        X : various
            The input data, which can be of any type supported by the dataset class, such as numpy arrays,
            PyTorch tensors, or scipy sparse matrices.
        y : various or None
            The target data, which can be of the same types as `X`, or None. If None, a placeholder tensor
            is used instead.

        Returns
        -------
        tuple of torch.Tensor
            The transformed `X` and `y`, ready for use in a PyTorch model. If `X` is a sparse matrix,
            it is converted to a dense array first. If `y` is None, it is replaced with a tensor of a
            single zero.
        """
        # DataLoader can't handle None, so we use a tensor with value 0 as a placeholder.
        y = torch.Tensor([0]) if y is None else y

        # PyTorch doesn't work well with sparse matrices, make it dense.
        if sparse.issparse(X):
            X = X.toarray().squeeze(0)
        
        return X, y
        
class ValidSplit:
    """
    Class to perform internal train/validation split on a dataset.

    This class abstracts the process of creating validation splits from a dataset
    using various cross-validation strategies. It is similar to the cross-validation
    utilities provided by scikit-learn. The class allows for simple train/validation
    splits as well as more complex strategies like k-folds.

    Parameters
    ----------
    cv : int, float, cross-validation generator, or iterable, default=5
        Determines the cross-validation splitting strategy. Options include:
        - None: Uses default 3-fold cross-validation.
        - int: Specifies the number of folds in a (Stratified)KFold.
        - float: Represents the proportion of the dataset to include in the 
        validation split (ShuffleSplit strategy).
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

    stratified : bool, default=False
        Whether to use stratified splits. This is only applicable for binary or 
        multiclass classification problems to ensure that each fold retains the 
        percentage of samples for each class.

    random_state : int, RandomState instance or None, default=None
        Random state to control the randomness of the training/validation splits. 
        This is only applicable when a float is passed to `cv`, signifying a 
        ShuffleSplit strategy.

    Attributes
    ----------
    cv : int, float, cross-validation generator, or iterable
        The cross-validation splitting strategy defined by the user.

    stratified : bool
        Indicates if the split should be stratified.

    random_state : int, RandomState instance, or None
        The random seed for generating deterministic shuffling sequences. None
        represents no fixed seed.

    Examples
    --------
    >>> from stockpy import ValidSplit
    >>> X, y = np.arange(10).reshape((5, 2)), range(5)
    >>> vs = ValidSplit(cv=2, stratified=False)
    >>> for train_index, test_index in vs.split(X, y):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    TRAIN: [3 4] TEST: [0 1 2]
    TRAIN: [0 1 2] TEST: [3 4]

    Notes
    -----
    This class uses only the first validation split. For models requiring
    validation across multiple splits, an external cross-validation strategy
    should be employed.

    See Also
    --------
    sklearn.model_selection.KFold : K-Fold cross-validator.
    sklearn.model_selection.ShuffleSplit : ShuffleSplit cross-validator.
    """

    def __init__(
            self,
            cv=5,
            stratified=False,
            random_state=None,
    ):
        """
        Constructs the ValidSplit object.

        Stores the parameters as attributes for future use. Validates the cv
        parameter and ensures that random_state is only used when applicable.

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
        """
        Checks if the cross-validation strategy is stratified.

        This method is used internally to determine whether the provided cross-validator
        will ensure that each fold has the same proportion of observations with a given
        categorical target as the entire dataset.

        Parameters
        ----------
        cv : cross-validation generator or None
            The cross-validation splitting strategy to check. This could be an instance
            of `StratifiedKFold`, `StratifiedShuffleSplit`, or None if no strategy has
            been defined.

        Returns
        -------
        bool
            True if `cv` is an instance of `StratifiedKFold` or `StratifiedShuffleSplit`,
            indicating that the cross-validation strategy is stratified. False otherwise.

        Notes
        -----
        This method is designed for internal use to validate the `cv` parameter during
        the initialization of the `ValidSplit` object. It is not intended for external use.
        """
        return isinstance(cv, (StratifiedKFold, StratifiedShuffleSplit))

    def _is_float(self, x):
        """
        Determines whether the input is a non-integer floating-point number.

        This private method checks if the given input `x` is a number and, if so,
        whether it is a float that does not represent an integer value. This is
        useful for interpreting inputs that could be either the number of folds in
        cross-validation or a proportion of the dataset for a split.

        Parameters
        ----------
        x : Number or any
            The input value to check.

        Returns
        -------
        bool
            True if `x` is a non-integer float, False otherwise.

        Notes
        -----
        This method is a utility function for internal validation of parameters and
        is not intended for public use. It helps determine the nature of the `cv`
        parameter in the `ValidSplit` class.
        """
        if not isinstance(x, Number):
            return False
        return not float(x).is_integer()

    def _check_cv_float(self):
        """
        Creates a cross-validation splitter instance for a floating-point `cv` value.

        This method is intended to be called internally when the `cv` parameter
        is a float. It instantiates the appropriate cross-validation splitter class
        based on whether the split should be stratified. The splitter is then used
        to generate train/test splits according to the proportion specified by `cv`.

        Returns
        -------
        StratifiedShuffleSplit or ShuffleSplit
            An instance of `StratifiedShuffleSplit` if `self.stratified` is True,
            otherwise an instance of `ShuffleSplit`.

        Notes
        -----
        This method should only be used when `cv` is a float. It is responsible for
        the internal logic of handling floating-point `cv` values and is not
        intended for public use.
        """
        cv_cls = StratifiedShuffleSplit if self.stratified else ShuffleSplit
        return cv_cls(test_size=self.cv, random_state=self.random_state)

    def _check_cv_non_float(self, y=None):
        """
        Validates and converts the cv parameter into a cross-validator object when cv is not a float.

        This method is intended for internal use to handle non-floating-point values of the `cv` parameter.
        It ensures that `cv` is a valid input for creating cross-validator objects, which are used to generate
        training and test splits. If `self.stratified` is True and the target `y` is provided, a stratified
        cross-validator will be used.

        Parameters
        ----------
        y : array-like, optional
            The target variable array. Necessary if a stratified cross-validator is required.

        Returns
        -------
        cross-validation generator
            A cross-validator object that can be used to generate train/test splits.

        Notes
        -----
        This method does not support float `cv` values. For floating-point values of `cv`, 
        use the `_check_cv_float` method instead. The `y` parameter is only needed if `self.stratified` 
        is True and stratification of splits is required.
        """
        return check_cv(
            self.cv,
            y=y,
            classifier=self.stratified,
        )

    def check_cv(self, y):
        """
        Resolves and returns the appropriate cross-validation generator based on the `cv` attribute.

        This method determines which cross-validation strategy is to be used, taking into account
        whether `cv` is a float, indicating a proportion for a ShuffleSplit, or an int/None/object
        specifying a K-fold strategy or another cross-validator. If `self.stratified` is True,
        it attempts to convert `y` into a numpy array format suitable for stratified splitting.

        Parameters
        ----------
        y : array-like or None, optional
            The target labels. If provided and `self.stratified` is True, a stratified
            cross-validator will be utilized, requiring `y` to perform the splits.

        Returns
        -------
        cross-validation generator
            The cross-validation generator object that will be used to create train/test splits.

        Notes
        -----
        The method internally calls `_check_cv_float` if `cv` is a float, otherwise, it calls
        `_check_cv_non_float`. If `self.stratified` is True and `y` cannot be converted to numpy format,
        the original `y` will be used for stratification.
        """
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
        """
        Determines if the input is a regular data type for processing.

        Regular data types are considered to be either None, NumPy arrays, or
        pandas DataFrames or Series. This utility function is typically used to
        verify that the input data can be processed without the need for special
        handling or conversions that are required for other data types.

        Parameters
        ----------
        x : various types
            The input data to be checked.

        Returns
        -------
        bool
            True if `x` is None, a NumPy array, or a pandas DataFrame/Series.
            False otherwise.
        """
        return (x is None) or isinstance(x, np.ndarray) or is_pandas_ndframe(x)

    def __call__(self, dataset, y=None, groups=None):
        """
        Splits a dataset into training and validation subsets.

        This method uses the cross-validation strategy specified in the `ValidSplit`
        object to split the dataset into training and validation subsets. If the
        strategy is stratified, it ensures that `y` is provided and that the
        cross-validation generator is appropriate for stratified splits. It then
        generates the indices for the split and creates corresponding subsets.

        Parameters
        ----------
        dataset : Dataset
            The complete dataset to be split.
        y : array-like, optional
            The target variable array. Required for stratified splits.
        groups : array-like, optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        tuple
            A tuple (dataset_train, dataset_valid) where `dataset_train` is a
            `Subset` of the original dataset used for training, and `dataset_valid`
            is a `Subset` used for validation.

        Raises
        ------
        ValueError
            If `y` is not provided for a stratified split or if the lengths of
            `dataset` and `y` do not match.
        """

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
        """
        Returns a string representation of the ValidSplit object.

        This method provides a human-readable representation of the `ValidSplit` object,
        including the class name and its initialization parameters. This is useful for
        clarity and debugging purposes.

        Returns
        -------
        str
            A string representation of the `ValidSplit` object.
        """
        return super(ValidSplit, self).__repr__()