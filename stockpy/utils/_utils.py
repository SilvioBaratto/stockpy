from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from enum import Enum
from functools import partial
import io
from itertools import tee
import pathlib
import warnings

import numpy as np
from scipy import sparse
import sklearn
from sklearn.exceptions import NotFittedError
from sklearn.utils import _safe_indexing as safe_indexing
from sklearn.utils.validation import check_is_fitted as sk_check_is_fitted
import torch
from torch import nn
from torch.nn import BCELoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from pyro.nn import PyroModule
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data.dataset import Subset

from ..exceptions import DeviceWarning
from ..exceptions import NotInitializedError

try:
    import torch_geometric
    TORCH_GEOMETRIC_INSTALLED = True
except ImportError:
    TORCH_GEOMETRIC_INSTALLED = False


class Ansi(Enum):
    """
    Enumeration for ANSI escape codes to set terminal text color.

    Attributes
    ----------
    BLUE : str
        ANSI code for blue text.
    CYAN : str
        ANSI code for cyan text.
    GREEN : str
        ANSI code for green text.
    MAGENTA : str
        ANSI code for magenta text.
    RED : str
        ANSI code for red text.
    ENDC : str
        ANSI code to reset text color to default.
    """

    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'

def is_torch_data_type(x):
    """
    Check if an object is a PyTorch data type.

    This function checks if the provided object is an instance of a PyTorch
    Tensor or a PackedSequence, commonly used in PyTorch for representing
    sequences of tensors.

    Parameters
    ----------
    x : object
        The object to check.

    Returns
    -------
    bool
        True if `x` is a PyTorch Tensor or PackedSequence, False otherwise.

    Examples
    --------
    >>> is_torch_data_type(torch.tensor([1, 2, 3]))
    True
    >>> is_torch_data_type([1, 2, 3])
    False
    """
    # pylint: disable=protected-access
    return isinstance(x, (torch.Tensor, PackedSequence))

def is_dataset(x):
    """
    Check if an object is a PyTorch dataset.

    Parameters
    ----------
    x : object
        The object to check.

    Returns
    -------
    bool
        True if `x` is an instance of torch.utils.data.Dataset, False otherwise.

    Examples
    --------
    >>> is_dataset(torch.utils.data.TensorDataset(torch.tensor([1, 2, 3])))
    True
    >>> is_dataset([1, 2, 3])
    False
    """
    return isinstance(x, torch.utils.data.Dataset)

def is_geometric_data_type(x):
    """
    Check if an object is a PyTorch Geometric data type.

    This function checks if the provided object is an instance of Data from the
    PyTorch Geometric library, which is used for representing graph data.

    Parameters
    ----------
    x : object
        The object to check.

    Returns
    -------
    bool
        True if `x` is an instance of torch_geometric.data.Data, False otherwise.

    Examples
    --------
    >>> from torch_geometric.data import Data
    >>> is_geometric_data_type(Data())
    True
    >>> is_geometric_data_type(torch.tensor([1, 2, 3]))
    False
    """
    from torch_geometric.data import Data
    return isinstance(x, Data)

def to_tensor(X, device, accept_sparse=False):
    """
    Convert input data to a PyTorch tensor, ensuring it is on the specified device.

    This function is versatile and can handle various input data types, including
    PyTorch's `PackedSequence`, NumPy arrays, existing torch Tensors, scipy sparse CSR
    matrices, and even complex structures like lists, tuples, and dictionaries that contain
    these data types.

    Parameters:
        X : various types
            The input data to convert. It supports a wide variety of types:
            - `torch.nn.utils.rnn.PackedSequence`
            - NumPy ndarray
            - torch Tensor
            - scipy sparse CSR matrix (if `accept_sparse=True`)
            - list or tuple containing any of the above types
            - dictionary with values containing any of the above types

        device : str or torch.device
            The compute device to be used (e.g., 'cpu' or 'cuda'). If `torch.Tensor` is
            provided, it will be transferred to the given device.

        accept_sparse : bool, optional (default=False)
            Indicates whether scipy sparse matrices should be accepted. If `True`, they are
            converted to torch sparse COO tensors. If `False`, an error will be raised when
            encountering a sparse matrix.

    Returns:
        torch.Tensor
            The input data `X` converted to a PyTorch tensor on the specified `device`.

    Raises:
        TypeError
            If the input data type cannot be converted to a tensor or if a sparse matrix is
            passed and `accept_sparse` is `False`.

    Examples:
        >>> import numpy as np
        >>> X_np = np.array([[1, 2], [3, 4]])
        >>> tensor = to_tensor(X_np, device='cpu')
        >>> tensor
        tensor([[1, 2],
                [3, 4]])

        >>> from scipy.sparse import csr_matrix
        >>> X_sparse = csr_matrix(X_np)
        >>> tensor = to_tensor(X_sparse, device='cpu', accept_sparse=True)
        >>> tensor.is_sparse
        True
    """
    to_tensor_ = partial(to_tensor, device=device)

    if is_torch_data_type(X):
        return to_device(X, device)
    if TORCH_GEOMETRIC_INSTALLED and is_geometric_data_type(X):
        return to_device(X, device)
    if hasattr(X, 'convert_to_tensors'):
        # huggingface transformers BatchEncoding
        return X.convert_to_tensors('pt')
    if isinstance(X, Mapping):
        return {key: to_tensor_(val) for key, val in X.items()}
    if isinstance(X, (list, tuple)):
        return [to_tensor_(x) for x in X]
    if np.isscalar(X):
        return torch.as_tensor(X, device=device)
    if isinstance(X, Sequence):
        return torch.as_tensor(np.array(X), device=device)
    if isinstance(X, np.ndarray):
        return torch.as_tensor(X, device=device)
    if sparse.issparse(X):
        if accept_sparse:
            return torch.sparse_coo_tensor(X.nonzero(), X.data, size=X.shape).to(device)
        raise TypeError("Sparse matrices are not supported. Set accept_sparse=True to allow sparse matrices.")

    raise TypeError("Cannot convert this data type to a torch tensor.")


def _is_slicedataset(X):
    """
    Check if the input is an instance of a sliced dataset.

    This function is a utility designed to identify whether the given input `X`
    represents a sliced dataset. It verifies this by checking for the presence
    of certain attributes that are characteristic of sliced datasets, namely
    'dataset', 'idx', and 'indices'. This is used instead of `isinstance` to
    avoid dependencies on external modules or specific class implementations.

    Parameters:
        X : object
            The input data to be checked. This could be any object, and the function
            will determine if it conforms to the sliced dataset pattern.

    Returns:
        bool
            True if `X` has the attributes 'dataset', 'idx', and 'indices', indicating
            it is a sliced dataset. False otherwise.

    Examples:
        >>> class MockSlicedDataset:
        ...     def __init__(self):
        ...         self.dataset = [1, 2, 3]
        ...         self.idx = 0
        ...         self.indices = [0, 1, 2]
        >>> sliced_data = MockSlicedDataset()
        >>> _is_slicedataset(sliced_data)
        True

        >>> regular_data = [1, 2, 3]
        >>> _is_slicedataset(regular_data)
        False

    Notes:
        This function does not check the validity of the attributes themselves or
        their types; it only checks for their existence.

    """
    return hasattr(X, 'dataset') and hasattr(X, 'idx') and hasattr(X, 'indices')


def to_numpy(X):
    """
    Convert a PyTorch tensor to a NumPy array.

    This function is designed to handle various data structures that may contain
    PyTorch tensors, such as dictionaries, lists, or custom dataset objects.
    It is capable of handling these containers by unpacking the tensors and
    converting them to NumPy arrays without diving deeper into nested structures.

    Parameters:
        X : various types
            The input data to convert. Can be one of the following:
            - NumPy ndarray: is returned as is.
            - Mapping: a dictionary where values are converted to NumPy arrays.
            - pandas DataFrame or Series: its `.values` are returned.
            - tuple or list: a similar container where each element is converted.
            - SlicedDataset: a custom dataset object that is converted to an array.
            - torch.Tensor: is converted to a NumPy array.

    Returns:
        numpy.ndarray
            The input data `X` converted to a NumPy array, or the data as is if it's
            already a NumPy array.

    Raises:
        TypeError
            If `X` is not a supported data type for conversion.

    Examples
    --------
    >>> import torch
    >>> X_tensor = torch.tensor([1, 2, 3])
    >>> X_numpy = to_numpy(X_tensor)
    >>> type(X_numpy)
    <class 'numpy.ndarray'>

    Notes
    -----
    If the tensor is on GPU, it will be moved to CPU before the conversion.
    If the tensor has the `requires_grad` attribute set, it will be detached
    before the conversion.

    """
    if isinstance(X, np.ndarray):
        return X

    if isinstance(X, Mapping):
        return {key: to_numpy(val) for key, val in X.items()}

    if is_pandas_ndframe(X):
        return X.values

    if isinstance(X, (tuple, list)):
        return type(X)(to_numpy(x) for x in X)

    if _is_slicedataset(X):
        return np.asarray(X.dataset[idx] for idx in X.indices)

    if not is_torch_data_type(X):
        raise TypeError("Cannot convert this data type to a numpy array.")

    if X.is_cuda:
        X = X.cpu()

    if hasattr(X, 'is_mps') and X.is_mps:
        X = X.cpu()

    if X.requires_grad:
        X = X.detach()

    return X.numpy()



def to_device(X, device):
    """
    Move input data to the specified device.

    This function transfers tensors or models (modules) to the desired device (e.g., CPU or GPU).
    If `X` is a distribution or `device` is None, the input is returned unmodified.

    Parameters:
        X : torch.Tensor, tuple, list, dict, torch.nn.Module, or PackedSequence
            The data to be moved to the specified device. Can handle various types of input:
            - Single torch tensor
            - Tuple or list of torch tensors
            - Dictionary of torch tensors
            - PackedSequence instance
            - An instance of torch.nn.Module (i.e., a neural network layer or model)

        device : str or torch.device, optional
            The device to move `X` to. If this is None, `X` is returned without modification.
            Otherwise, `X` is moved to the specified device.

    Returns:
        object
            The data moved to the specified device, or unmodified if `device` is None or `X`
            is a PyTorch distribution object.

    Examples:
        >>> tensor = torch.tensor([1, 2, 3])
        >>> to_device(tensor, 'cuda')
        tensor([1, 2, 3], device='cuda:0')

        >>> module = torch.nn.Linear(2, 2)
        >>> to_device(module, 'cuda')
        Linear(in_features=2, out_features=2, bias=True).to('cuda:0')

    Notes:
        - The function does not modify `X` in place; it returns a new object on the specified device.
        - PyTorch distribution objects are not moved as they do not support the `.to()` method.

    """
    if device is None:
        return X

    if isinstance(X, Mapping):
        # dict-like but not a dict
        return type(X)({key: to_device(val, device) for key, val in X.items()})

    # PackedSequence class inherits from a namedtuple
    if isinstance(X, (tuple, list)) and (type(X) != PackedSequence):
        return type(X)(to_device(x, device) for x in X)

    if isinstance(X, torch.distributions.distribution.Distribution):
        return X

    return X.to(device)


def get_dim(y):
    """
    Get the number of dimensions of a tensor or array.

    This function returns the number of dimensions of the input, whether it's
    a PyTorch tensor or a NumPy array. It tries to access the `ndim` attribute
    common in NumPy arrays, and if that fails (i.e., an AttributeError is caught),
    it attempts to call the `dim()` method used in PyTorch tensors.

    Parameters:
        y : array-like or torch.Tensor
            The input from which to determine the number of dimensions. Can be a NumPy
            array-like object or a PyTorch tensor.

    Returns:
        int
            The number of dimensions of the input.

    Examples:
        >>> import numpy as np
        >>> y_np = np.array([1, 2, 3])
        >>> get_dim(y_np)
        1

        >>> import torch
        >>> y_torch = torch.tensor([[1, 2], [3, 4]])
        >>> get_dim(y_torch)
        2

    Raises:
        AttributeError
            If the input `y` does not have a `ndim` attribute or a `dim()` method,
            indicating it is neither a NumPy array nor a PyTorch tensor.

    """
    try:
        return y.ndim
    except AttributeError:
        return y.dim()


def is_pandas_ndframe(x):
    """
    Check if the object is a pandas NDFrame (e.g., DataFrame or Series).

    This function determines if the passed object is a pandas NDFrame instance
    by checking for the presence of the 'iloc' attribute, which is common to
    pandas data structures.

    Parameters:
        x : object
            The object to check.

    Returns:
        bool
            True if `x` has the 'iloc' attribute, False otherwise.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> is_pandas_ndframe(df)
        True

        >>> is_pandas_ndframe([1, 2, 3])
        False

    """
    return hasattr(x, 'iloc')


def flatten(arr):
    """
    Flatten a nested structure of tuples, lists, and dictionaries into a flat generator.

    This function takes a nested structure containing tuples, lists, and dictionaries
    and yields its elements in a flat structure. If an element within the passed
    structure is itself a tuple, list, or dictionary, the function will recursively
    yield from that sub-element until non-iterable items are reached.

    Parameters:
        arr : iterable
            An iterable which may contain nested structures like tuples, lists, and dictionaries.

    Yields:
        item : object
            The next flat item in the nested structure.

    Examples:
        >>> list(flatten([1, [2, 3], (4, 5), {'a': 6, 'b': 7}]))
        [1, 2, 3, 4, 5, 6, 7]

        >>> list(flatten([(1, 2), [3, [4, 5]]]))
        [1, 2, 3, 4, 5]

    """
    for item in arr:
        if isinstance(item, (tuple, list, Mapping)):
            yield from flatten(item)
        else:
            yield item

def _indexing_none(data, i):
    """
    Returns None for any given index.

    This function is a placeholder indexing function that returns None regardless
    of the input data or index.

    Parameters:
        data : Any
            The data to index (unused in this function).
        i : int or slice or array-like
            The index to use (unused in this function).

    """
    return None


def _indexing_dict(data, i):
    """
    Perform indexing on a dictionary where each value is indexed by `i`.

    This function iterates over the key-value pairs in the dictionary and applies
    indexing to each value using the provided index `i`.

    Parameters:
        data : dict
            Dictionary with indexable values.
        i : int or slice or array-like
            The index to apply to each value of the dictionary.

    Returns:
        dict
            A new dictionary with each original value indexed by `i`.
    """

    return {k: v[i] for k, v in data.items()}


def _indexing_list_tuple_of_data(data, i, indexings=None):
    """
    Index into a list or tuple of data structures using provided indexing functions or ad hoc determination.

    The function handles indexing into each element of the list or tuple, which can be any indexable data
    structure (e.g., numpy array).

    Parameters:
        data : list or tuple
            A list or tuple of indexable data structures.
        i : int or slice or array-like
            The index to apply to each element of the list or tuple.
        indexings : list of callable, optional
            A list of functions that define how to index into each element of `data`. If not provided, indexing
            functions are determined ad hoc.

    Returns:
        list or tuple
            A new list or tuple with each original element indexed by `i`.
    """

    if not indexings:
        return [multi_indexing(x, i) for x in data]
    return [multi_indexing(x, i, indexing)
            for x, indexing in zip(data, indexings)]


def _indexing_ndframe(data, i):
    """
    Perform indexing on a pandas NDFrame or on a dictionary equivalent.

    If `data` has an `iloc` attribute, it is assumed to be a pandas NDFrame and
    indexed accordingly. Otherwise, it falls back to dictionary indexing.

    Parameters:
        data : pandas NDFrame or dict
            The NDFrame or dictionary-like object to index.
        i : int or slice or array-like
            The index to apply.

    Returns:
        pandas NDFrame or dict
            The indexed data, as an NDFrame if `data` was an NDFrame, or as a dictionary
            if `data` was dictionary-like.
    """

    if hasattr(data, 'iloc'):
        return data.iloc[i]
    return _indexing_dict(data, i)


def _indexing_other(data, i):
    """
    Perform indexing on various data structures that are not explicitly handled by other indexing functions.

    This function deals with indexing on data structures not covered by other specific indexing functions,
    such as tuples or structures that are compatible with sklearn's `safe_indexing`. It first checks if the index
    is of type int, numpy integer, slice, or tuple, and applies the index directly. Otherwise, it falls back to using
    `safe_indexing`.

    Parameters:
        data : various
            The data to be indexed, which could be any type not covered by other specific indexing functions.
        i : int or np.integer or slice or tuple or array-like
            The index to apply. If it is a basic type (int, np.integer, slice, tuple), direct indexing is used;
            otherwise, `safe_indexing` is applied.

    Returns:
        various
            The subset of `data` indexed by `i`. The type of the return value is dependent on the input data type.
    """

    # sklearn's safe_indexing doesn't work with tuples since 0.22
    if isinstance(i, (int, np.integer, slice, tuple)):
        return data[i]
    return safe_indexing(data, i)


def check_indexing(data):
    """
    Determine the most appropriate indexing function for the given data.

    Based on the type of the input data, this function selects an indexing function that can be used
    to access elements or slices of the data. This function supports a range of data types including
    lists, tuples, dictionaries, pandas data structures, and others.

    Parameters:
        data : object
            The data for which an indexing method needs to be determined. This could be a list, tuple,
            dictionary, pandas DataFrame, pandas Series, or other data structures.

    Returns:
        function
            A function that is appropriate for indexing the provided data. This function will accept
            two arguments: the data and the index/key.

    Examples:
        >>> data_list = [1, 2, 3]
        >>> indexing_function = check_indexing(data_list)
        >>> indexing_function(data_list, 0)
    1

    Notes:
        The returned indexing function will be one of several predefined functions designed for specific data types.
        If the data is `None`, the function returns `_indexing_none` which will always return `None` when called.
        For dictionary data, `_indexing_dict` is returned, which applies the index to each value in the dictionary.
        For list or tuple data, `_indexing_list_tuple_of_data` is used to index each element in the list or tuple.
        For pandas NDFrame objects, `_indexing_ndframe` is returned, which uses the `.iloc` indexer for pandas.
        For all other types, `_indexing_other` is returned, which attempts to index the data directly or falls back
        to using `safe_indexing` from scikit-learn.

    Raises:
        TypeError
            If the data type is not supported for indexing, a TypeError may be raised when the returned
            indexing function is called.
    """
    # If data is None, return the corresponding indexing function
    if data is None:
        return _indexing_none

    # Check if the data is a dictionary and return the corresponding indexing function
    if isinstance(data, Mapping):
        return _indexing_dict

    # Check for list or tuple types
    if isinstance(data, (list, tuple)):
        try:
            multi_indexing(data[0], 0)
            indexings = [check_indexing(x) for x in data]
            return partial(_indexing_list_tuple_of_data, indexings=indexings)
        except TypeError:
            return _indexing_other

    # Check for pandas NDFrame (DataFrame or Series)
    if is_pandas_ndframe(data):
        return _indexing_ndframe

    # For other types like torch tensor and numpy ndarray
    return _indexing_other


def _normalize_numpy_indices(i):
    """
    Normalize numpy array indices to lists or tuples for general use.

    If the input is a boolean numpy array, it will be converted to a tuple of lists representing
    the indices where the value is True. If the input is an integer numpy array, it will be
    converted to a list of integers.

    Parameters:
        i : numpy.ndarray
            The numpy array index to be normalized. Can be an array of booleans or integers.
        
    Returns:
        list or tuple
            A list of indices if the input array is of integer dtype, or a tuple of lists if the
            input array is of boolean dtype. If the input is not a numpy array, it will be returned
            unchanged.

    Examples:
        >>> import numpy as np
        >>> bool_idx = np.array([True, False, True])
        >>> _normalize_numpy_indices(bool_idx)
        ([0, 2],)

        >>> int_idx = np.array([1, 3, 5])
        >>> _normalize_numpy_indices(int_idx)
        [1, 3, 5]
    """
    # Check if the input i is a numpy array
    if isinstance(i, np.ndarray):
        # Convert boolean numpy array to a tuple of lists with True indices
        if i.dtype == bool:
            # Nonzero returns a tuple of arrays, one for each dimension of i,
            # containing the indices of the True elements in that dimension.
            # The .tolist() converts these arrays to Python lists.
            i = tuple(j.tolist() for j in i.nonzero())
        # Convert integer numpy array to a list of integers
        elif i.dtype == int:
            # .tolist() converts the numpy array to a Python list
            i = i.tolist()
    # Return the normalized index
    return i

def multi_indexing(data, i, indexing=None):
    """
    Perform indexing on various data structures with support for different index types.

    This function is capable of indexing standard Python data structures as well as 
    numpy arrays, pandas frames, and torch tensors. It can automatically determine the 
    correct indexing method or use a provided one.

    Parameters:
        data : object
            The data structure to be indexed. This could be a list, tuple, dictionary,
            numpy array, pandas DataFrame, or any other object that supports indexing.
            
        i : int, slice, or numpy.ndarray
            The index or indices to apply. This could be a single position for simple 
            indexing, a slice object for range indexing, or a numpy array for advanced 
            indexing scenarios.
            
        indexing : callable, optional
            An optional function that performs indexing on `data`. If provided, this 
            function will be used directly. If None, an appropriate indexing function will 
            be determined based on the type of `data`.
        
    Returns:
        object
            The result of indexing `data` with `i`. The type of the return value is dependent
            on the type of `data` and the operation performed.
    
    Examples:
        >>> multi_indexing([1, 2, 3], 1)
        2
        
        >>> multi_indexing({'a': 1, 'b': 2}, 'a')
        1
        
        >>> import numpy as np
        >>> multi_indexing(np.array([1, 2, 3]), np.array([True, False, True]))
        array([1, 3])
    """
    # Normalize numpy array indices to be compatible with other data types
    i = _normalize_numpy_indices(i)

    # If a custom indexing function is provided, use it
    if indexing is not None:
        return indexing(data, i)

    # If no custom indexing function is provided, determine the appropriate one
    # The check_indexing function checks the type of `data` and returns the
    # appropriate indexing function to be used.
    return check_indexing(data)(data, i)

def duplicate_items(*collections):
    """
    Identify duplicate items across multiple collections.

    This function takes an arbitrary number of collections and identifies items
    that are present in more than one of the collections. It works with any type
    of iterable collections like lists, sets, tuples, and dictionaries. For 
    dictionaries, only the keys are considered.

    Parameters:
        *collections : iterable
            An arbitrary number of collections where each collection is an iterable. 
            This could be lists, sets, tuples, or dictionaries. If dictionaries are 
            provided, their keys are used for comparison.

    Returns:
        set
            A set containing the items that appear in more than one of the input collections.
            If there are no duplicates, an empty set is returned.

    Examples:
        >>> duplicate_items([1, 2], [3])
        set()
        
        >>> duplicate_items({1: 'a', 2: 'b'}, {2: 'c', 3: 'd'})
        {2}
        
        >>> duplicate_items(['a', 'b', 'a'])
        {'a'}
        
        >>> duplicate_items([1, 2], {3: 'hi', 4: 'ha'}, (2, 3))
        {2, 3}
    """
    duplicates = set()  # Will hold the items that are found in more than one collection
    seen = set()  # Will hold all items that have been seen so far

    # Flatten all collections and iterate over each item
    for item in flatten(collections):
        # If an item has been seen already, it's a duplicate
        if item in seen:
            duplicates.add(item)
        else:
            # If it hasn't been seen, add it to the set of seen items
            seen.add(item)
    return duplicates


def params_for(prefix, kwargs):
    """
    Extract parameters for a specified prefix from keyword arguments.
    
    This function is particularly useful for extracting parameters that are meant
    for a specific component of a composite model in scikit-learn. Each parameter
    is assumed to be prefixed with the name of the component and two underscores.
    For example, 'component_name__parameter'.

    Parameters:
        prefix : str
            The prefix of the submodule to filter parameters for. The prefix should not
            contain trailing underscores as they are added automatically.
        
        kwargs : dict
            A dictionary of keyword arguments where each key is a string consisting
            of the prefix and the parameter name, separated by two underscores.
    
    Returns:
        dict
            A dictionary containing only the parameters that belong to the specified prefix.
    
    Examples:
        >>> kwargs = {'encoder__a': 3, 'encoder__b': 4, 'decoder__a': 5}
        >>> params_for('encoder', kwargs)
        {'a': 3, 'b': 4}
        
        >>> params_for('', kwargs)
        {'encoder__a': 3, 'encoder__b': 4, 'decoder__a': 5}  # Returns all parameters if prefix is empty

    Note:
        The function assumes that the parameters for each component of a composite model
        are uniquely identified by a prefix followed by two underscores. This is a common
        convention in scikit-learn when using estimators within a Pipeline, FeatureUnion,
        or ColumnTransformer.
    """
    # If the prefix is empty, return the entire kwargs as no filtering is required
    if not prefix:
        return kwargs
    
    # Ensure the prefix ends with double underscores for proper filtering
    if not prefix.endswith('__'):
        prefix += '__'
    
    # Filter and return only those parameters whose keys start with the prefix
    return {
        key[len(prefix):]: val  # Remove the prefix part of the key
        for key, val in kwargs.items()  # Iterate over all key, value pairs in kwargs
        if key.startswith(prefix)  # Select only those that start with the prefix
    }


class _none:
    pass


def data_from_dataset(dataset, X_indexing=None, y_indexing=None):
    """
    Extract feature and target data (X, y) from a dataset object.

    This function is designed to work with `skorch.dataset.Dataset` instances or subsets thereof.
    It attempts to retrieve the data by directly accessing `X` and `y` attributes or, for subsets,
    by applying the provided indexing functions or inferring them if none are provided.

    Parameters:
        dataset : skorch.dataset.Dataset or torch.utils.data.Subset
            The dataset from which to extract `X` and `y`. Expected to be an instance of
            `skorch.dataset.Dataset` or `torch.utils.data.Subset` thereof.
            
        X_indexing : callable, optional
            A function used to index into the `X` data. If None, an indexing method will be
            inferred based on the type of the dataset.
            
        y_indexing : callable, optional
            A function used to index into the `y` data. If None, an indexing method will be
            inferred based on the type of the dataset.

    Returns:
        tuple
            A tuple containing the extracted `X` and `y` data.

    Raises:
        AttributeError
            If the function is unable to access `X` and `y` attributes from the provided dataset.

    Examples:
        >>> from skorch import NeuralNetClassifier
        >>> from skorch.dataset import Dataset
        >>> from torch.utils.data import Subset
        >>> net = NeuralNetClassifier(...)
        >>> ds = Dataset(X, y)
        >>> X, y = data_from_dataset(ds)
        
        >>> subset_indices = [0, 1, 2]
        >>> sub_ds = Subset(ds, subset_indices)
        >>> X_sub, y_sub = data_from_dataset(sub_ds)

    Notes:
        - If the dataset is a `Subset`, the function will recursively call itself to access the underlying
        dataset's `X` and `y`, then apply indexing to get the subset of data.
        - For `torch.utils.data.dataset.TensorDataset`, it assumes that the dataset contains exactly two
        tensors, where the first tensor is `X` and the second is `y`.

    """

    X, y = _none, _none

    if isinstance(dataset, Subset):
        X, y = data_from_dataset(
            dataset.dataset, X_indexing=X_indexing, y_indexing=y_indexing)
        X = multi_indexing(X, dataset.indices, indexing=X_indexing)
        y = multi_indexing(y, dataset.indices, indexing=y_indexing)
    elif hasattr(dataset, 'X') and hasattr(dataset, 'y'):
        X, y = dataset.X, dataset.y
    elif isinstance(dataset, torch.utils.data.dataset.TensorDataset):
        if len(items := dataset.tensors) == 2:
            X, y = items

    if (X is _none) or (y is _none):
        raise AttributeError("Could not access X and y from dataset.")
    return X, y


def is_stockpy_dataset(ds):
    """
    Determine whether a dataset is an instance of `StockpyDataset`.

    This function checks if the supplied dataset is a `StockpyDataset` instance, which is expected
    to be part of the stockpy library. It is recursive to account for `StockpyDataset` instances
    that might be wrapped within `torch.util.data.Subset`.

    Parameters:
        ds : skorch.dataset.Dataset or torch.utils.data.Subset
            The dataset to check. This can be a direct instance of `StockpyDataset` or a `Subset`
            that contains a `StockpyDataset`.

    Returns:
        bool
            `True` if `ds` is an instance of `StockpyDataset`, otherwise `False`.

    Examples:
        >>> from stockpy.preprocessing import StockpyDataset
        >>> dataset = StockpyDataset(X, y)
        >>> is_stockpy_dataset(dataset)
        True

        >>> from torch.utils.data import Subset
        >>> subset_dataset = Subset(dataset, indices=[0, 1, 2])
        >>> is_stockpy_dataset(subset_dataset)
        True

    Notes:
        - The function checks for the type recursively. If the dataset is a `Subset`, the function
        calls itself with the underlying dataset to perform the check. This allows it to handle
        nested `Subset` structures.

    """
    from stockpy.preprocessing import StockpyDataset  # Assuming StockpyDataset is defined here

    # Recursive check for instances of StockpyDataset, even within Subsets
    if isinstance(ds, Subset):
        return is_stockpy_dataset(ds.dataset)

    # Direct instance check
    return isinstance(ds, StockpyDataset)


# pylint: disable=unused-argument
def noop(*args, **kwargs):
    """
    A no-op (no operation) function that ignores arguments and returns `None`.

    This can be utilized where a function is required for interface compatibility or as a
    placeholder, but where the function's operation is not needed or should be skipped.

    Parameters:
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

    Returns:
        None
            This function returns `None`, indicating it has no return value or effect.

    Examples:
        Here's an example of using `noop` as a placeholder:

        >>> def optional_processing(callback=noop):
        ...     # Some processing here
        ...     result = "Processed data"
        ...     callback(result)  # `noop` ignores the result and does nothing
        ...
        >>> optional_processing()
        'Processed data'

        Here's another example where `noop` is used in a callback list:

        >>> callbacks = [noop, print, noop]
        >>> for cb in callbacks:
        ...     cb('Hello, World!')  # `noop` callbacks will do nothing
        ...
        Hello, World!

    Notes:
        - `noop` is designed to be compatible with any set of arguments passed to it.
        - Using `noop` may be helpful to define default callbacks that should not perform any action.
        - The use of `noop` can make it clearer that a certain callback or operation is intentionally
        being skipped, as opposed to omitting the callback or passing `None`.

    """
    pass  


@contextmanager
def open_file_like(f, mode):
    """
    A context manager for opening and ensuring the closure of a file-like object.

    This context manager accepts either a string path or a pathlib.Path object, which
    it will open as a file. It can also accept an already open file-like object, in which
    case it will simply yield that object without opening a new file. The file-like
    object is guaranteed to be closed when exiting the context.

    Parameters:
        f : str or pathlib.Path or file-like object
            The file path to open or the file-like object to be handled.
        mode : str
            The mode in which the file should be opened (e.g., 'r' for read, 'w' for write).

    Yields:
        file-like object
            An opened file-like object ready for reading or writing.

    Examples:
        Using a file path:

        >>> with open_file_like('example.txt', 'r') as file_obj:
        ...     contents = file_obj.read()
        ... # file is automatically closed after the with block

        Using an already open file-like object:

        >>> with open(some_file_path, 'r') as existing_file_obj:
        ...     with open_file_like(existing_file_obj, 'r') as file_obj:
        ...         contents = file_obj.read()
        ... # file is automatically closed after the with block

    Raises:
        IOError
            If opening the file fails for any reason, an IOError will be raised.

    Notes:
        - This function is especially useful in cases where the file-like object may
        or may not already be open, abstracting away the check and ensuring consistent
        cleanup.
        - When passing an open file-like object, ensure that the mode it was opened with
        is compatible with the intended use within the context.

    """
    # Attempt to open the file or use the given file-like object
    if isinstance(f, (str, pathlib.Path)):
        file_like = open(f, mode)
    else:
        file_like = f

    # Yield the file-like object to the context
    try:
        yield file_like
    finally:
        # Ensure the file-like object is closed after use
        file_like.close()


def train_loss_score(net, X=None, y=None):
    """
    Extracts the training loss from the last recorded batch in the training history.

    This function is typically used as a scoring callback to retrieve the training loss
    from a neural network's training history.

    Parameters:
        net : skorch.NeuralNet
            The neural network instance containing the training history.
        X : array-like, optional
            Ignored. Included for compatibility with skorch's scoring callback signature.
        y : array-like, optional
            Ignored. Included for compatibility with skorch's scoring callback signature.

    Returns:
        float
            The training loss of the last batch from the last epoch.

    Examples:
        >>> from skorch import NeuralNetClassifier
        >>> net = NeuralNetClassifier(module=MyModule, callbacks={'train_loss': train_loss_score})
        >>> net.fit(X_train, y_train)
        >>> last_train_loss = train_loss_score(net)

    Note:
        This function expects the neural network's history to be populated, which happens
        during the fitting process. Accessing this function without previously fitting the
        net will result in IndexError.
    """
    return net.history[-1, 'batches', -1, 'train_loss']


def valid_loss_score(net, X=None, y=None):
    """
    Extracts the validation loss from the last recorded batch in the training history.

    Similar to `train_loss_score`, this function is used as a scoring callback to
    retrieve the validation loss from the neural network's training history.

    Parameters:
        net : skorch.NeuralNet
            The neural network instance containing the training history.
        X : array-like, optional
            Ignored. Included for compatibility with skorch's scoring callback signature.
        y : array-like, optional
            Ignored. Included for compatibility with skorch's scoring callback signature.

    Returns:
        float
            The validation loss of the last batch from the last epoch.

    Examples:
        >>> from skorch import NeuralNetClassifier
        >>> net = NeuralNetClassifier(module=MyModule, callbacks={'valid_loss': valid_loss_score})
        >>> net.fit(X_train, y_train)
        >>> last_valid_loss = valid_loss_score(net)

    Note:
        This function expects the neural network's history to be populated, which happens
        during the fitting process. Accessing this function without previously fitting the
        net will result in IndexError.
        """
    return net.history[-1, 'batches', -1, 'valid_loss']



class FirstStepAccumulator:
    """
    Store and retrieve the train step data.

    This class is designed to accumulate the training step data. It stores the first train step value and consistently 
    returns this first value upon request. This can be particularly useful when using optimization algorithms that 
    involve multiple train steps per iteration, and you are only interested in the first step of each iteration.

    Attributes:
        step : object
            The data object that represents the first train step. Initially set to None until the first step is stored.

    Methods:
        store_step(step)
            Stores the first step value if no step has been stored yet.
        get_step()
            Returns the stored step value.

    Examples:
        >>> accumulator = FirstStepAccumulator()
        >>> accumulator.store_step(1)
        >>> accumulator.get_step()
        1

    Note:
        This accumulator is used by default in `skorch.NeuralNet` for optimizers that call the train step once. If using 
        an optimizer that calls the train step multiple times (e.g., LBFGS), and you wish to use a different accumulating 
        strategy (like accumulating the last step), you would need to implement a custom accumulator.
    """

    def __init__(self):
        """
        Initializes the `FirstStepAccumulator` instance, setting the initial step value to None.
        """
        self.step = None

    def store_step(self, step):
        """
        Stores the first step value.

        This method will only store the step if no previous step has been stored. If a step is already present, 
        it does nothing.

        Parameters:
            step : object
                The step data to store as the first step.

        Examples:
            >>> accumulator = FirstStepAccumulator()
            >>> accumulator.store_step({'loss': 0.1})
            >>> accumulator.get_step()
            {'loss': 0.1}
        """
        if self.step is None:
            self.step = step

    def get_step(self):
        """
        Returns the stored step value.

        If no step has been stored yet, this will return None.

        Returns:
            object
                The step data stored by the `store_step` method.

        Examples:
            >>> accumulator = FirstStepAccumulator()
            >>> print(accumulator.get_step())
            None
            >>> accumulator.store_step({'loss': 0.1})
            >>> print(accumulator.get_step())
            {'loss': 0.1}
        """
        return self.step


def _make_split(X, y=None, valid_ds=None, **kwargs):
    """
    Create a split of the data suitable for model validation.

    This utility function is primarily used by the `predefined_split` callback to
    create a data split that can be used for validation purposes. This split is
    made prior to fitting a model and is used to monitor the model's performance on
    a validation set during training.

    Parameters:
        X : array-like
            The training data to split.
        y : array-like, optional
            The target values. Default is None.
        valid_ds : Dataset, optional
            A predefined Dataset to be used as a validation set. Default is None.

    Returns:
        tuple
            A tuple containing the training data and the validation dataset.

    Other Parameters:
        **kwargs
            Additional keyword arguments.

    Examples:
        >>> X_train, y_train = np.array([...]), np.array([...])
        >>> valid_dataset = MyCustomDataset(X_valid, y_valid)
        >>> X, valid_ds = _make_split(X_train, y_train, valid_ds=valid_dataset)
    """

    return X, valid_ds


def freeze_parameter(param):
    """
    Set the `requires_grad` attribute of a PyTorch parameter to `False`.

    This function is used to freeze a parameter within a PyTorch model, which means
    that the parameter will not update during training. This is commonly used during
    transfer learning when certain layers of a pre-trained model are kept static.

    Parameters:
        param : torch.nn.Parameter
            The parameter to freeze.

    Examples:
        >>> for param in model.parameters():
        ...     freeze_parameter(param)
    """
    param.requires_grad = False


def unfreeze_parameter(param):
    """
    Set the `requires_grad` attribute of a PyTorch parameter to `True`.

    This function is used to unfreeze a parameter within a PyTorch model, allowing
    it to update during training. This is typically used after a certain number of
    training steps or epochs have occurred, to begin fine-tuning a model.

    Parameters:
        param : torch.nn.Parameter
            The parameter to unfreeze.

    Examples:
        >>> for param in model.parameters():
        ...     unfreeze_parameter(param)
    """

    param.requires_grad = True


def get_map_location(target_device, fallback_device='cpu'):
    """
    Determine the appropriate device location for mapping loaded data, such as model weights.

    This function is particularly useful when loading a saved model's state dict. It ensures that
    the state dict is loaded onto the correct device, especially useful when a user requests loading
    onto a CUDA device when no CUDA-capable devices are available, prompting a fallback to CPU.

    Parameters:
        target_device : str or torch.device
            The desired device to load the data onto. Can be a string (e.g., 'cuda:0' or 'cpu')
            or a `torch.device` object. If `None`, will default to the `fallback_device`.
        fallback_device : str, optional
            The device to fall back to if the `target_device` is not available. Default is 'cpu'.

    Returns:
        torch.device
            The device to map the loaded data to.

    Raises:
        DeviceWarning
            If the `target_device` is CUDA but no CUDA devices are available, a warning is raised and
            the `fallback_device` is used instead.

    Examples:
        >>> get_map_location('cuda')
        device(type='cuda', index=0)  # if a CUDA device is available
        >>> get_map_location('cuda')
        device(type='cpu')  # if no CUDA devices are available
    """

    if target_device is None:
        target_device = fallback_device

    map_location = torch.device(target_device)

    # The user wants to use CUDA but there is no CUDA device
    # available, thus fall back to CPU.
    if map_location.type == 'cuda' and not torch.cuda.is_available():
        warnings.warn(
            'Requested to load data to CUDA but no CUDA devices '
            'are available. Loading on device "{}" instead.'.format(
                fallback_device,
            ), DeviceWarning)
        map_location = torch.device(fallback_device)
    return map_location


def check_is_fitted(estimator, attributes=None, msg=None, all_or_any=all):
    """
    Ensure that an estimator is initialized or fitted.

    This function checks if an estimator has been fitted or initialized by
    checking the existence of specific attributes. It's particularly useful
    to verify whether an estimator is ready for making predictions or further
    processing.

    Parameters:
        estimator : estimator instance
            An estimator instance for which the check is performed.
        attributes : str, list of str, or None, optional
            The attribute names to check for. This can either be a single string
            or a list of strings. If `None`, this is determined by the `estimator`.
        msg : str or None, optional
            The message to use for the error. If `None`, a default message is used.
        all_or_any : callable, optional
            A callable that determines the logic to use when multiple attributes are
            passed in `attributes`. The callable should be either `all` (default) or `any`.

    Raises:
        NotInitializedError
            If the given estimator is not initialized.

    Notes:
        This function calls `sklearn.utils.validation.check_is_fitted` internally
        with the same arguments and logic. The key difference lies in the exception
        raised when the estimator is deemed not initialized, where it raises
        `skorch.exceptions.NotInitializedError` instead of
        `sklearn.exceptions.NotFittedError`.

    Examples:
        >>> from sklearn.linear_model import LogisticRegression
        >>> check_is_fitted(LogisticRegression(), 'coef_')
        Traceback (most recent call last):
            ...
        NotInitializedError: This LogisticRegression instance is not initialized yet...
    """

    try:
        sk_check_is_fitted(estimator, attributes, msg=msg, all_or_any=all_or_any)
    except NotFittedError as exc:
        if msg is None:
            msg = ("This %(name)s instance is not initialized yet. Call "
                   "'initialize' or 'fit' with appropriate arguments "
                   "before using this method.")

        raise NotInitializedError(msg % {'name': type(estimator).__name__}) from exc


def _identity(x):
    """
    Identity function that returns the input unchanged.

    Parameters:
        x : object
            The input object.

    Returns:
        object
            The same as input `x`.

    Examples:
        >>> _identity(5)
        5
        >>> _identity([1, 2, 3])
        [1, 2, 3]
    """
    return x

def _make_2d_probs(prob):
    """
    Convert a 1-dimensional probability tensor into a 2-dimensional one.

    This function is designed to prepare the output of a binary classifier
    to match scikit-learn's convention of expecting two probabilities for each
    instance, one for each class.

    Parameters:
        prob : torch.Tensor
            A 1-dimensional tensor containing probabilities of the positive class.

    Returns:
        torch.Tensor
            A 2-dimensional tensor with the first column containing the probabilities
            of the negative class and the second column the probabilities of the
            positive class.

    Examples:
        >>> _make_2d_probs(torch.tensor([0.2, 0.5, 0.8]))
        tensor([[0.8000, 0.2000],
                [0.5000, 0.5000],
                [0.2000, 0.8000]])
    """
    y_proba = torch.stack((1 - prob, prob), 1)
    return y_proba



def _sigmoid_then_2d(x):
    """
    Apply sigmoid function to logits and format the output as a 2D probability array.

    This function applies the sigmoid function to convert raw logits to probabilities, then formats these probabilities into a 2D array where the sum of probabilities in each row is 1. This is required for compatibility with scikit-learn's expectation for output of predict_proba method in classifiers.

    Parameters:
        x : torch.Tensor
            A 1-dimensional torch tensor of raw logits.

    Returns:
        torch.Tensor
            A 2-dimensional torch tensor with probabilities. Each row corresponds to a sample, and the two columns represent the probabilities of the negative and positive classes respectively.

    """
    prob = torch.sigmoid(x)
    return _make_2d_probs(prob)

# pylint: disable=protected-access
def _infer_predict_nonlinearity(net):
    """
    Infer the appropriate nonlinearity to apply based on the loss criterion of the net.

    This function determines which nonlinearity should be applied to the output of the neural network model before making a prediction or returning probabilities. This is only applied when using the `predict` or `predict_proba` methods of `NeuralNetClassifier`.

    Parameters:
        net : skorch.NeuralNet
            The neural network on which to infer the nonlinearity.

    Returns:
        function
            A function that applies the appropriate nonlinearity to the output of the model based on the loss criterion.

    """
    # At the moment, this function dispatches based on the criterion.
    if isinstance(net.criterion, torch.nn.CrossEntropyLoss):
        return partial(torch.softmax, dim=-1)

    if isinstance(net.criterion, torch.nn.BCEWithLogitsLoss):
        return _sigmoid_then_2d

    if isinstance(net.criterion, torch.nn.BCELoss):
        return _make_2d_probs

    return _identity

class TeeGenerator:
    """
    A wrapper class for a generator that allows multiple iterations over its data.

    `TeeGenerator` stores a single instance of a generator and uses the `itertools.tee` function to create new independent generators upon each iteration. This permits multiple passes over the original generator's data without exhausting it. Care must be taken as `tee` can lead to memory issues if the iterators are not used at a similar pace.

    Parameters:
        gen : generator
            The generator instance from which `tee` will create new generators on each iteration.

    Yields:
        iterator
            An iterator over the underlying generator's data.

    Examples:
        >>> gen = (x for x in range(10))
        >>> tee_gen = TeeGenerator(gen)
        >>> for _ in range(2):  # We can iterate over tee_gen multiple times
        ...     print(list(tee_gen))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    Notes:
        Iterators returned by `tee` do not have the ability to save their position for later use. 
        Once an item has been consumed by a `tee`-generated iterator, it is gone unless stored elsewhere. 
        In addition, as the original generator is advanced, `tee` stores the data until all derived iterators
        have consumed it, which may lead to increased memory usage.

    """
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        """
        Create and yield from a new iterator derived from the stored generator without exhausting it.

        When the `TeeGenerator` is iterated over, it uses `tee` to split the original generator into two: one that is saved back into `TeeGenerator` and another that is used for the current iteration.

        Returns:
            iterator
                An iterator that yields from the tee'd generator.
        """
        self.gen, it = tee(self.gen)
        yield from it


def _check_f_arguments(caller_name, **kwargs):
    """
    Validates keyword arguments for functions dealing with file operations in skorch.

    This function ensures that keyword arguments passed to methods that save or load
    attributes of a skorch net are correctly prefixed with 'f_'. It also checks that
    'f_params' and 'f_module' are not both passed since they refer to the same thing.
    It categorizes the arguments into those intended for module parameters and others.

    Parameters:
        caller_name : str
            The name of the function that calls this utility function, used for generating error messages.

        **kwargs : dict
            Arbitrary keyword arguments that will be checked for adherence to expected naming patterns.

    Returns:
        kwargs_module : dict
            A dictionary with keyword arguments that pertain to module parameters, suitable for saving/loading model parameters.

        kwargs_other : dict
            A dictionary with other file-related keyword arguments, suitable for saving/loading objects other than model parameters.

    Raises:
        TypeError
            Raised if the keyword arguments do not follow the expected 'f_' prefix pattern or if both 'f_params' and 'f_module' are provided.

    Examples:
        >>> _check_f_arguments('save_params', f_params='model.pth', f_history='history.json')
        ({'module_': 'model.pth'}, {'f_history': 'history.json'})

        >>> _check_f_arguments('load_params', f_optimizer='optimizer.pth', f_pickle='model.pkl')
        ({'optimizer_': 'optimizer.pth'}, {'f_pickle': 'model.pkl'})

        >>> _check_f_arguments('save_params', f_params='model.pth', f_module='module.pth')
        TypeError: save_params called with both f_params and f_module, please choose one

    Notes:
        The function will normalize 'f_params' to 'module_' internally, as 'f_params' and 'f_module' refer to the same set of parameters.
    """

    if kwargs.get('f_params') and kwargs.get('f_module'):
        raise TypeError("{} called with both f_params and f_module, please choose one"
                        .format(caller_name))

    kwargs_module = {}
    kwargs_other = {}
    keys_other = {'f_history', 'f_pickle'}
    for key, val in kwargs.items():
        if not key.startswith('f_'):
            raise TypeError(
                "{name} got an unexpected argument '{key}', did you mean 'f_{key}'?"
                .format(name=caller_name, key=key))

        if val is None:
            continue
        if key in keys_other:
            kwargs_other[key] = val
        else:
            # strip 'f_' prefix and attach '_', and normalize 'params' to 'module'
            # e.g. 'f_optimizer' becomes 'optimizer_', 'f_params' becomes 'module_'
            key = 'module_' if key == 'f_params' else key[2:] + '_'
            kwargs_module[key] = val
            
    return kwargs_module, kwargs_other

def get_activation_function(name):
    """
    Retrieves a PyTorch activation function module by name.

    Given the name of an activation function, this utility function returns the corresponding
    PyTorch module from a predefined set of activation functions. If the provided name does
    not match any predefined activation functions, it defaults to returning a ReLU module.

    Parameters:
        name : str
            The name of the activation function to retrieve. Accepted values are 'relu', 'tanh',
            'sigmoid', 'softmax', 'leaky_relu', and 'gelu'.

    Returns:
        nn.Module
            A PyTorch activation function module corresponding to the provided name.

    Examples:
        >>> get_activation_function('relu')
        ReLU()

        >>> get_activation_function('tanh')
        Tanh()

        >>> get_activation_function('unknown')
        ReLU()  # Default case when 'unknown' is not a recognized name

    Notes:
        The 'softmax' activation function is defined with `dim=1`, which applies the Softmax
        operation across the second dimension (commonly the features dimension in many models).
    """

    activation_functions = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'softmax': nn.Softmax(dim=1),
        'leaky_relu': nn.LeakyReLU(),
        'gelu': nn.GELU(),
    }
    
    return activation_functions.get(name, nn.ReLU())

def evaluate(y_test, y_pred, show=False):
    """
    Evaluates the performance of a predictive model by computing various error metrics.

    Given the true values and the predicted values, this function calculates the Mean Squared Error (MSE),
    Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE). Optionally, it can print
    these metrics instead of returning them.

    Parameters:
        y_test : array-like
            True values to compare predictions against.
        y_pred : array-like
            Predicted values generated by the model.
        show : bool, optional
            If True, print the performance metrics instead of returning them.

    Returns:
        tuple of (float, float, float)
            A tuple containing the MSE, RMSE, and MAPE, unless `show` is True.

    Examples:
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> evaluate(y_true, y_pred)
        (0.375, 0.612, 0.225)

        >>> evaluate(y_true, y_pred, show=True)
        Model Performance
        Mean squared error = 0.375
        Root mean squared error = 0.612
        Mean absolute percentage error = 22.5%.

    Notes:
        - The metrics are calculated using scikit-learn's metric functions.
        - DTW (Dynamic Time Warping) distance can also be calculated but is commented out.
        If needed, uncomment the respective line and ensure `fastdtw` and `euclidean` are
        imported correctly.
        - The function assumes that `y_test` and `y_pred` are of compatible shapes and that
        `y_test` does not contain zeros if MAPE is to be calculated (to avoid division by zero errors).
    """

    mse = sklearn.metrics.mean_squared_error(y_test, y_pred, squared=True)
    rmse = sklearn.metrics.mean_squared_error(y_test, y_pred, squared=False)
    mape = sklearn.metrics.mean_absolute_percentage_error(y_test, y_pred)
    # dtw_distance = fastdtw(y_test, y_pred, dist=euclidean)[0] 
    if show:
        print('Model Performance')
        print("Mean squared error = {:0.3f}".format(mse))
        print("Root mean squared error = {:0.3f}".format(rmse))
        print('Mean absolute percentage error = {:0.3f}%.'.format(mape))
        # print('DTW distance = {:0.3f} '.format(dtw_distance))
    else:
        return mse, rmse, mape