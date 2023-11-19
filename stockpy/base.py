import fnmatch
from collections.abc import Mapping
from collections import defaultdict
from functools import partial
from itertools import chain
from contextlib import contextmanager
import re
import random
import os
import tempfile
import warnings

import numpy as np
from sklearn.base import BaseEstimator as SkBaseEstimator 
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from scipy.stats import mode
import torch
from torch.utils.data import DataLoader
import pyro
from pyro.infer.svi import SVI
from pyro.infer import TraceMeanField_ELBO
from safetensors import safe_open
from safetensors.torch import load
from safetensors.torch import save_file, save

from stockpy.callbacks import EpochTimer
from stockpy.callbacks import PrintLog
from stockpy.callbacks import EpochScoring
from stockpy.callbacks import PassthroughScoring
from stockpy.preprocessing import StockDatasetFFNN
from stockpy.preprocessing import StockDatasetRNN
from stockpy.preprocessing import StockDatasetCNN
from stockpy.preprocessing import ValidSplit
from stockpy.preprocessing import get_len
from stockpy.preprocessing import unpack_data
from stockpy.history import History
from stockpy.exceptions import DeviceWarning
from stockpy.exceptions import StockpyAttributeError
from stockpy.exceptions import StockpyTrainingImpossibleError
from stockpy.utils import _check_f_arguments
from stockpy.utils import TeeGenerator
from stockpy.utils import _identity
from stockpy.utils import _infer_predict_nonlinearity
from stockpy.utils import FirstStepAccumulator
from stockpy.utils import check_is_fitted
from stockpy.utils import duplicate_items
from stockpy.utils import get_map_location
from stockpy.utils import is_dataset
from stockpy.utils import params_for
from stockpy.utils import to_device
from stockpy.utils import to_numpy
from stockpy.utils import to_tensor
from stockpy.utils import data_from_dataset, is_dataset, get_dim, to_numpy

from sklearn.utils.validation import (
    _check_y,
    _get_feature_names,
    _num_features,
    check_array,
    check_X_y,
)

from sklearn.utils.multiclass import unique_labels

import re

def _extract_optimizer_param_name_and_group(optimizer_name, param):
    """
    Extract the parameter name and group from a given parameter string based on the optimizer name.

    The function uses a regular expression to parse the parameter string, which should follow the
    pattern 'optimizer_name__[param_groups__<group_number>__]<parameter_name>'. If the group number
    is not specified, it defaults to 'all'.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer to which the parameter belongs.
    param : str
        The parameter string to parse.

    Returns
    -------
    tuple
        A tuple containing the parameter group (str) and parameter name (str).

    Raises
    ------
    AttributeError
        If the parameter string does not match the expected pattern.

    Examples
    --------
    >>> _extract_optimizer_param_name_and_group('adam', 'adam__param_groups__1__lr')
    ('1', 'lr')
    >>> _extract_optimizer_param_name_and_group('sgd', 'sgd__momentum')
    ('all', 'momentum')
    """

    # Combine both patterns into a single one using a non-capturing group for the optional part.
    pattern = rf'{optimizer_name}__(?:param_groups__(?P<group>\d+)__)?(?P<name>.+)'

    # Compile the regular expression pattern.
    compiled_pattern = re.compile(pattern)

    # Match the parameter against the compiled pattern.
    match = compiled_pattern.fullmatch(param)

    # Raise an exception if there is no match.
    if not match:
        raise AttributeError(f'Invalid parameter "{param}" for optimizer "{optimizer_name}"')

    # Extract the group dictionary from the match object.
    groups = match.groupdict()

    # Default to 'all' if 'group' is not found in the match.
    param_group = groups.get('group', 'all')

    # The 'name' is required, so it should be present.
    param_name = groups['name']

    # Return the extracted group and name.
    return param_group, param_name

def _set_optimizer_param(optimizer, param_group, param_name, value):
    """
    Set a specific parameter for an optimizer's parameter group(s).

    This function directly modifies the optimizer's parameter groups by setting a specific 
    parameter (e.g., learning rate) to a new value. If 'all' is passed as the param_group,
    the parameter is set for all parameter groups; otherwise, it sets the parameter for
    the specified group only.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer object (e.g., instance of torch.optim.Optimizer).
    param_group : str or int
        The index of the parameter group as an integer, or 'all'
        to indicate all parameter groups.
    param_name : str
        The name of the parameter to be set (e.g., 'lr' for learning rate).
    value : float
        The new value to set for the parameter.

    Examples
    --------
    >>> optimizer = torch.optim.Adam([...], lr=0.001)  # An example optimizer
    >>> _set_optimizer_param(optimizer, 'all', 'lr', 0.01)
    # This sets the learning rate to 0.01 for all parameter groups

    >>> _set_optimizer_param(optimizer, 0, 'betas', (0.9, 0.999))
    # This sets the 'betas' parameter for the first parameter group only

    Raises
    ------
    IndexError
        If the `param_group` is not 'all' and does not correspond to a valid index in the optimizer's parameter groups.
    KeyError
        If the `param_name` is not a valid parameter for the optimizer's parameter groups.
    """

    # Obtain the appropriate optimizer parameter groups.
    groups = optimizer.param_groups if param_group == 'all' else [optimizer.param_groups[int(param_group)]]

    # Set the parameter value for the chosen groups.
    for group in groups:
        if param_name not in group:
            raise KeyError(f'Parameter "{param_name}" not found in the optimizer parameter group.')
        group[param_name] = value


def optimizer_setter(net, param, value, optimizer_attr='optimizer_', optimizer_name='optimizer'):
    """
    Set the value of a specified parameter in the optimizer or directly in the network.

    Parameters
    ----------
    net : object
        The neural network object which contains the optimizer.
    param : str
        The name of the parameter to update. If it's 'lr', it sets the global
        learning rate; otherwise, it expects a string that includes the optimizer
        name and the parameter group (if applicable).
    value : float or tuple
        The new value to set for the parameter. This could be a single value or a tuple,
        depending on the parameter being set.
    optimizer_attr : str, optional
        The attribute name of the optimizer in the network object.
        Defaults to 'optimizer_'.
    optimizer_name : str, optional
        The name of the optimizer. Defaults to 'optimizer'.

    Examples
    --------
    >>> net = SomeNetworkClass()
    >>> net.optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    >>> optimizer_setter(net, 'lr', 0.01)
    # This will set the global learning rate to 0.01.

    >>> optimizer_setter(net, 'optimizer__param_groups__0__weight_decay', 0.01)
    # This will set the 'weight_decay' for the first parameter group of the optimizer.

    Raises
    ------
    AttributeError
        If the provided param string does not match the expected pattern and
        cannot be processed.
    IndexError
        If the param_group extracted is not ' all' and does not correspond to a valid
        index in the optimizer's parameter groups.
    KeyError
        If the param_name is not a valid parameter within the optimizer's parameter groups.
    """

    # First, determine if the parameter to be set is a global learning rate
    is_global_lr = param == 'lr'

    # Extract or directly set the param_group and param_name
    param_group = 'all' if is_global_lr else None
    param_name = param if is_global_lr else None

    if not is_global_lr:
        param_group, param_name = _extract_optimizer_param_name_and_group(optimizer_name, param)

    # If it's a learning rate, we're setting it directly as well as on the optimizer
    if is_global_lr:
        net.lr = value

    # Regardless of parameter type, set it on the optimizer
    _set_optimizer_param(
        optimizer=getattr(net, optimizer_attr),
        param_group=param_group,
        param_name=param_name,
        value=value
    )

class BaseEstimator:
    """
    Base class for all estimators in the Stockpy framework.

    This class provides common functionality to all estimator objects within the framework,
    including device handling, initialization of components, and parameter management. It
    should be inherited by any new estimator that is introduced.

    Attributes
    ----------
    prefixes_ : list of str
        List of prefixes that are used in the parameter names for special handling.
    cuda_dependent_attributes_ : list of str
        Attributes that depend on the CUDA device.
    init_context_ : NoneType or object
        Context within which the estimator is initialized. This is for advanced usage.
    _modules : list
        A list that holds modules of the estimator.

    Parameters
    ----------
    compile : bool, default=False
        Whether to compile the estimator immediately.
    use_caching : str, default='auto'
        Determines whether to use caching for certain operations. Can be set to 'auto', True, or False.
    **kwargs : dict
        Additional keyword arguments that are specific to the inheriting estimator class.

    Notes
    -----
    - `cuda_dependent_attributes_` list is used to properly transfer attributes to the CUDA device if needed.
    - The `use_caching` parameter can improve performance by caching certain computations.
    - The `init_context_` is typically None but can be set to a context manager in advanced scenarios.
    - The `compile` parameter is for future use and is not implemented in the current version.
    - `_modules`, `_criteria`, and `_optimizers` are meant to be populated by inheriting classes.
    - The class utilizes dynamic attribute setting for kwargs, and as such, any additional kwargs are set as attributes.

    Raises
    ------
    ValueError
        If there are any issues with the passed kwargs, such as name conflicts or deprecated parameters.
    """

    prefixes_ = ['iterator_train', 'iterator_valid', 'callbacks', 'dataset', 'compile']

    cuda_dependent_attributes_ = []

    init_context_ = None

    _modules = []
    _criteria = []
    _optimizers = []

    # pylint: disable=too-many-arguments
    def __init__(
            self,
            compile=False,
            use_caching='auto',
            **kwargs
    ):
        # Instance attributes
        self.compile = compile
        self.use_caching = use_caching
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prob = "probabilistic" in str(self.__class__)

        # It appears there is a duplicate call to _check_deprecated_params here
        self._check_deprecated_params(**kwargs)

        # Extract specific kwargs into attributes, remove them from kwargs to avoid duplication
        self.history_ = kwargs.pop('history', None)
        self.initialized_ = kwargs.pop('initialized_', False)
        self.virtual_params_ = kwargs.pop('virtual_params_', dict())

        # Prepare to validate additional params
        self._params_to_validate = set(kwargs.keys())

        # Set any remaining kwargs as attributes of the instance
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def history(self):
        return self.history_

    @history.setter
    def history(self, value):
        self.history_ = value

    @property
    def _default_callbacks(self):
        """
        Default callbacks used during training and validation processes.

        This property returns a list of callback tuples. Each tuple consists of a unique string 
        identifier and an instance of the callback class, defining specific actions to be taken 
        at various stages of the training process.

        Returns
        -------
        list of tuple
            A list of tuples where the first element is the callback identifier (str)
            and the second element is an instance of the callback (object).

        Examples
        --------
        By accessing this property, you get the default callbacks:

        >>> model = YourModelClass()
        >>> model._default_callbacks
        [('epoch_timer', EpochTimer()), ('train_loss', PassthroughScoring(name='train_loss', on_train=True)), ...]

        Notes
        -----
        This is an internal property meant to be used by the training framework to maintain
        a consistent set of callbacks. Users can typically override or extend the list of callbacks
        by providing their own during the initialization or configuration of their model instance.
        """

        # Your method implementation goes here...
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', PassthroughScoring(
                name='train_loss',
                on_train=True,
            )),
            ('valid_loss', PassthroughScoring(
                name='valid_loss',
            )),
            ('print_log', PrintLog()),
        ]

    def get_default_callbacks(self):
        return self._default_callbacks

    def notify(self, method_name, **cb_kwargs):
        """
        Invoke a method by name on the instance and all registered callbacks.

        This method dynamically calls the method specified by `method_name` on the current instance
        and on each callback registered in `self.callbacks_`. This is typically used to signal
        events such as the beginning or end of a training epoch, a training step, or other 
        significant occurrences during the training process.

        Parameters
        ----------
        method_name : str
            The name of the method to be invoked on the instance and each callback.
        **cb_kwargs : dict
            Arbitrary keyword arguments that will be passed to the method.

        Examples
        --------
        Assuming a callback is registered with a method `on_epoch_end`, it can be notified like so:

        >>> trainer = YourTrainingClass()
        >>> trainer.register_callback(your_callback_instance)
        >>> trainer.notify('on_epoch_end', epoch=5, logs={...})
        # This will call `on_epoch_end` on the instance and all callbacks with the provided arguments.

        Notes
        -----
        All callbacks registered in `self.callbacks_` are expected to have the method specified
        by `method_name`. If a callback does not have this method, an AttributeError will be raised.
        """

        # Call the method on self
        getattr(self, method_name)(self, **cb_kwargs)

        # Call the method on all callbacks
        for _, cb in self.callbacks_:
            getattr(cb, method_name)(self, **cb_kwargs)

    # pylint: disable=unused-argument
    def on_train_begin(self, net, X=None, y=None, **kwargs):
        pass

    # pylint: disable=unused-argument
    def on_train_end(self, net, X=None, y=None, **kwargs):
        pass

    # pylint: disable=unused-argument
    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        self.history.new_epoch()
        self.history.record('epoch', len(self.history))

    # pylint: disable=unused-argument
    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        pass

    # pylint: disable=unused-argument
    def on_batch_begin(self, net, batch=None, training=False, **kwargs):
        self.history.new_batch()

    def on_batch_end(self, net, batch=None, training=False, **kwargs):
        pass

    def on_grad_computed(
            self, net, named_parameters, batch=None, training=False, **kwargs):
        pass

    def _yield_callbacks(self):
        """
        Organize and yield all callbacks set on this instance.

        This generator function iterates through all default and user-set callbacks, categorizes
        them by their names, and yields them. `PrintLog` callbacks are treated specially and are
        yielded last to ensure logging happens after all other callbacks have been processed.

        Callbacks provided by the user in tuple or list format (where the first element is the 
        callback's name and the second is the instance) are distinguished from those that are not.

        Yields
        ------
        tuple
            A 3-element tuple consisting of the callback's name (str), the callback instance (object),
            and a boolean indicating whether the callback was named by the user (bool).

        Examples
        --------
        >>> trainer = YourTrainingClass()
        >>> for name, cb_instance, named_by_user in trainer._yield_callbacks():
        >>>     print(f"Callback Name: {name}, Named by User: {named_by_user}")
        # This will print the name of each callback and whether it was named by the user.

        Notes
        -----
        This is an internal method and is expected to be used by the class internally to
        process callbacks. The order of callbacks, except for `PrintLog`, is not guaranteed.
        """

        # Using defaultdict to automatically handle uninitialized keys.
        callback_dict = defaultdict(list)

        # Process all callbacks.
        for cb in self.get_default_callbacks() + (self.callbacks or []):
            named_by_user = isinstance(cb, (tuple, list))
            if named_by_user:
                name, cb_instance = cb
            else:
                cb_instance = cb
                name = cb.__name__ if isinstance(cb, type) else cb.__class__.__name__

            # We categorize callbacks by type for later processing.
            callback_dict[name].append((cb_instance, named_by_user))

        # Separate PrintLog callbacks to append them last.
        print_logs = callback_dict.pop('PrintLog', [])
        
        # Yield non-PrintLog callbacks first.
        for name, cbs in callback_dict.items():
            for cb_instance, named_by_user in cbs:
                yield name, cb_instance, named_by_user
        
        # Finally, yield PrintLog callbacks.
        for cb_instance, named_by_user in print_logs:
            yield 'PrintLog', cb_instance, named_by_user

    def _callbacks_grouped_by_name(self):
        """
        Group callbacks by their names and identify those set by the user.

        This method organizes callbacks into a dictionary keyed by their names, with each key 
        corresponding to a list of callback instances. It also identifies which callback names have 
        been explicitly set by the user and collects these names into a set.

        Returns
        -------
        tuple
            A 2-element tuple where the first element is a dictionary with callback names as keys
            and lists of callback instances as values, and the second element is a set of 
            callback names that were explicitly set by the user.

        Examples
        --------
        >>> trainer = YourTrainingClass()
        >>> callbacks_grouped, names_set_by user = trainer._callbacks_grouped_by_name()
        >>> print(callbacks_grouped)
        # Output might look like {'TrainLoss': [instance1, instance2], 'PrintLog': [instance3]}
        >>> print(names_set_by_user)
        # Output might be a set of user-set callback names like {'TrainLoss', 'ValidLoss'}

        Notes
        -----
        This method is typically used internally to prepare and manage the state before starting
        a training process or similar routine. It ensures that callbacks can be executed or 
        retrieved efficiently by grouping them by name.
        """

        callbacks_grouped = defaultdict(list)
        names_set_by_user = set()

        for name, cb, is_named_by_user in self._yield_callbacks():
            if is_named_by_user:
                names_set_by_user.add(name)
            callbacks_grouped[name].append(cb)

        return dict(callbacks_grouped), names_set_by_user

    def _uniquely_named_callbacks(self):
        """
        Ensure each callback has a unique name within the callbacks collection.

        This method iterates over the grouped callbacks obtained from `_callbacks_grouped_by_name`.
        It checks for name conflicts and ensures that each callback has a unique name by appending
        an index to the name if necessary. User-defined callback names are preserved and will raise
        an exception if duplicates are found. Additionally, an exception is raised if the process
        of making a name unique inadvertently creates a name that already exists in the collection.

        Yields
        ------
        tuple
            A 2-element tuple consisting of the unique callback name (str) and the callback
            instance (object).

        Raises
        ------
        ValueError
            If duplicate user-set callback names are found or if renaming a callback
            to ensure uniqueness results in a name that already exists.

        Examples
        --------
        >>> trainer = YourTrainingClass()
        >>> for unique_name, cb in trainer._uniquely_named_callbacks():
        >>>     print(f"Callback Name: {unique_name}")
        # This will print the unique name for each callback.

        Notes
        -----
        This is an internal method used to prepare callbacks before a process such as training
        starts. It is not meant to be used directly by the end-user.
        """

        grouped_cbs, names_set_by_user = self._callbacks_grouped_by_name()
        for name, cbs in grouped_cbs.items():
            if len(cbs) > 1 and name in names_set_by_user:
                raise ValueError("Found duplicate user-set callback name "
                                 "'{}'. Use unique names to correct this."
                                 .format(name))

            for i, cb in enumerate(cbs):
                if len(cbs) > 1:
                    unique_name = '{}_{}'.format(name, i+1)
                    if unique_name in grouped_cbs:
                        raise ValueError("Assigning new callback name failed "
                                         "since new name '{}' exists already."
                                         .format(unique_name))
                else:
                    unique_name = name
                yield unique_name, cb
    
    def initialize_callbacks(self):
        """
        Initialize all callbacks and store them in the `callbacks_` attribute.

        This method consolidates callbacks from both `default_callbacks` and user-defined `callbacks`,
        initializing each one and assigning a unique name if not already named. It ensures that the 
        callback names are unique, raising a ValueError if a name conflict is detected. All callbacks 
        are then initialized by calling their `initialize` method.

        The result is stored as a list of tuples within the `callbacks_` attribute, where each tuple 
        consists of a callback's name and the initialized callback object.

        Returns
        -------
        self : object
            The instance with the `callbacks_` attribute set to the list of initialized 
            callbacks.

        Raises
        ------
        ValueError
            If there are parameters set for a callback that does not exist or if there is
            an attempt to set a callback with a non-unique name.

        Examples
        --------
        >>> trainer = YourTrainingClass()
        >>> trainer.initialize_callbacks()
        # After this call, trainer.callbacks_ will have the initialized callbacks

        Notes
        -----
        This method is part of the setup process and should be called before starting the main
        routine (like training in machine learning) that uses callbacks.
        """

        callbacks_ = []

        class Dummy:
            # We cannot use None as dummy value since None is a
            # legitimate value to be set.
            pass

        for name, cb in self._uniquely_named_callbacks():
            # check if callback itself is changed
            param_callback = getattr(self, 'callbacks__' + name, Dummy)
            if param_callback is not Dummy:  # callback itself was set
                cb = param_callback

            # below: check for callback params
            # don't set a parameter for non-existing callback
            params = self.get_params_for('callbacks__{}'.format(name))
            if (cb is None) and params:
                raise ValueError("Trying to set a parameter for callback {} "
                                 "which does not exist.".format(name))
            if cb is None:
                continue

            if isinstance(cb, type):  # uninitialized:
                cb = cb(**params)
            else:
                cb.set_params(**params)
            cb.initialize()
            callbacks_.append((name, cb))

        # pylint: disable=attribute-defined-outside-init
        self.callbacks_ = callbacks_
        return self
    
    def initialized_instance(self, instance_or_cls, kwargs):
        """
        Initialize or re-initialize an instance or class with given parameters.

        This utility method is designed to handle the initialization of components, 
        taking into account several scenarios, such as whether the component is 
        already an instance or a class that needs to be instantiated. It ensures 
        that the component is properly initialized with the given keyword arguments.

        Parameters
        ----------
        instance_or_cls : object or type
            The component to be initialized. It can be an instance, a class, 
            or any callable that requires initialization.
        kwargs : dict
            Keyword arguments for initialization. If `instance_or_cls` is 
            already an instance and `kwargs` is empty, the instance is returned 
            as is.

        Returns
        -------
        object
            The initialized instance of the component.

        Examples
        --------
        >>> module = MyModuleClass
        >>> initialized_mod = trainer.initialized_instance(module, {'input_features': 10})
        # `initialized_mod` is now an instance of MyModuleClass with input_features set to 10

        Notes
        -----
        If `instance_or_cls` is already an instance and `kwargs` is provided, 
        a new instance of the same type is created with the given keyword 
        arguments. If `instance_or_cls` is a class or callable, it is 
        initialized with `kwargs`.
        """

        # If it's an instance, modify its attributes
        is_init = isinstance(instance_or_cls, torch.nn.Module)
        if is_init and not kwargs:
            return instance_or_cls
        if is_init:
            return type(instance_or_cls)(**kwargs)
        return instance_or_cls(**kwargs)

    def initialize_criterion(self):
        """
        Initializes the criterion component of the model.

        This method prepares the criterion (typically a loss function) that will be used
        during the model's training process. If the criterion has already been initialized
        and there are no changes to its parameters, it is left unchanged.

        Returns
        -------
        self : object
            The instance with the `criterion_` attribute set to the initialized criterion.

        Examples
        --------
        >>> model = YourModelClass()
        >>> model.initialize_criterion()
        # After this call, model.criterion_ will be set to the initialized criterion object

        Note
        ----
        This method is typically called during the setup phase before training begins.
        The criterion can be a predefined PyTorch loss function or a user-defined one.
        The necessary parameters for initialization are retrieved using the `get_params_for`
        method with 'criterion' as an argument, which should return a dictionary of parameters.
        """

        kwargs = self.get_params_for('criterion')
        self.criterion_ = self.initialized_instance(self.criterion_, kwargs)

        return self
    
    def _is_virtual_param(self, key):
        """
        Checks if the given key is a virtual parameter.

        A virtual parameter is not actually stored as an attribute but is managed
        through a custom mechanism possibly involving pattern matching and dynamic
        handling.

        Parameters
        ----------
        key : str
            The key to check for virtual parameter status.

        Returns
        -------
        bool
            True if the key matches any of the virtual parameter patterns, False otherwise.

        Examples
        --------
        >>> self._is_virtual_param('learning_rate')
        True  # If 'learning_rate' is a virtual parameter
        """

        return any(fnmatch.fnmatch(key, pat) for pat in self.virtual_params_)
    
    def _virtual_setattr(self, param, val):
        """
        Sets an attribute on the instance as part of handling virtual parameters.

        This function is the default function used to set the value of a virtual
        parameter on the instance.

        Parameters
        ----------
        param : str
            The parameter name to set.
        val : object
            The value to assign to the parameter.
        """

        setattr(self, param, val)

    def _register_virtual_param(self, param_patterns, fn=_virtual_setattr):
        """
        Registers a pattern or a list of patterns as virtual parameters with a
        corresponding handling function.

        Parameters
        ----------
        param_patterns : str or list
            The pattern or list of patterns that define the virtual parameters.
        fn : callable, optional
            The function to handle setting the virtual parameter. Defaults to
            `_virtual_setattr`, which sets the parameter directly on the instance.

        Examples
        --------
        >>> self._register_virtual_param('learning_rate_*', custom_handler_function)
        """

        if not isinstance(param_patterns, list):
            param_patterns = [param_patterns]
        for pattern in param_patterns:
            self.virtual_params_[pattern] = fn

    def _apply_virtual_params(self, virtual_kwargs):
        """
        Applies the virtual parameter logic to the given keyword arguments.

        It uses pattern matching to determine which parameters are virtual and
        applies the registered function to set them.

        Parameters
        ----------
        virtual_kwargs : dict
            A dictionary where keys are parameter names and values are
            the values to be set for those parameters.

        Examples
        --------
        >>> self._apply_virtual_params({'learning_rate_init': 0.01})
        # Applies the virtual parameter handling for 'learning_rate_init'
        """

        for pattern, fn in self.virtual_params_.items():
            for key, val in virtual_kwargs.items():
                if not fnmatch.fnmatch(key, pattern):
                    continue
                fn(self, key, val)

    def initialize_virtual_params(self):
        """
        Initializes the container for virtual parameters.

        This method sets up an empty dictionary to hold virtual parameter patterns and
        their corresponding functions.

        Examples
        --------
        >>> self.initialize_virtual_params()
        # After this call, `self.virtual_params_` will be an empty dictionary ready to store virtual params
        """

        self.virtual_params_ = {}
    
    def initialize_elbo(self):
        """
        Initializes the Evidence Lower Bound (ELBO) criterion.

        If `elbo` is already initialized and no parameter has been changed, it will remain unchanged.
        Otherwise, `elbo` will be re-initialized with the current parameters.

        The initialized `elbo` attribute is set on the instance.

        Returns
        -------
        self : object
            The instance with the initialized `elbo` attribute.

        Examples
        --------
        >>> self.initialize_elbo()
        # This will set the `elbo` attribute on `self` after initializing it with the appropriate parameters.
        """

        kwargs = self.get_params_for('elbo')
        self.elbo = self.initialized_instance(self.elbo, kwargs)
        setattr(self, 'elbo', self.elbo)  # Save the updated criterion

        return self

    def initialize_optimizer(self, triggered_directly=None):
        """
        Initializes the optimizer for the model. If the optimizer's learning rate (`optimizer__lr`) is 
        not explicitly set, it falls back to using the learning rate specified by `self.lr`.

        Parameters
        ----------
        triggered_directly : bool, optional
            This parameter is deprecated and should not be used.
            It is maintained for backward compatibility.

        Raises
        ------
        DeprecationWarning
            If `triggered_directly` is used, a deprecation warning is issued.

        Returns
        -------
        self : object
            The instance with the initialized optimizer.

        Examples
        --------
        >>> self.initialize_optimizer()
        # This will initialize the optimizer with parameters for the model.

        Notes
        -----
        - The optimizer can be customized through `self.optimizer` which is expected to be
        a class or a factory function for creating optimizer instances.
        - The method handles conditional initialization if probabilistic parameters are involved,
        indicated by `self.prob`.
        """

        # Handle deprecated parameter
        if triggered_directly is not None:
            warnings.warn(
                "The 'triggered_directly' argument to 'initialize_optimizer' is "
                "deprecated, please don't use it anymore.", DeprecationWarning)

        # Retrieve all learnable parameters of the model
        named_parameters = self.get_all_learnable_params()
        # Extract arguments for the optimizer initialization
        args, kwargs = self.get_params_for_optimizer('optimizer', named_parameters)

        # Initialize the optimizer conditionally based on probabilistic settings
        if self.prob is False:
            self.optimizer_ = self.optimizer(*args, **kwargs)
        else:
            optim_args = {'lr': kwargs.pop('lr')} if 'lr' in kwargs else {}
            self.optimizer_ = self.optimizer(optim_args)

        return self
    
    def initialize_history(self):
        """
        Initializes the history of the model. If the history has not been created yet, it 
        instantiates a new History object. If the history already exists, it resets it, 
        effectively clearing any previous records.

        Returns
        -------
        self : object
            The instance with its history initialized or reset.

        Examples
        --------
        >>> model.initialize_history()
        # This will set up a fresh history or clear the existing one.

        Notes
        -----
        - The history is used to record metrics and other information during training.
        - It is essential to initialize or clear the history before starting a new training
        to ensure that the training process starts with a clean state.
        """
        # Initialize History object if it doesn't exist
        if self.history_ is None:
            self.history_ = History()
        # If it exists, clear the existing history for a fresh start
        else:
            self.history_.clear()

        return self
        
    def initialize_stochastic_variational_inference(self):
        """
        Initializes the Stochastic Variational Inference (SVI) for the model. This method 
        sets up the SVI with the model, guide, optimizer, and the ELBO loss function. 
        It then applies any additional parameters specific to the SVI configuration.

        Returns
        -------
        self : object
            The instance with its SVI component initialized.

        Examples
        --------
        >>> model.initialize_stochastic_variational_inference()
        # This will prepare the model for variational inference with the SVI approach.

        Notes
        -----
        - This method assumes that `self.model`, `self.guide`, `self.optimizer_`, and `self.elbo`
        have been previously initialized and are available as attributes of the instance.
        - The `triggered_directly` parameter is deprecated and not used within this method.
        """
        # Initialize the SVI object with model, guide, optimizer, and loss function
        svi = SVI(self.model, self.guide, self.optimizer_, loss=self.elbo)
        # Extract any SVI-specific parameters from the instance
        kwargs = self.get_params_for('svi')
        # Update the SVI with extracted parameters
        self.svi_ = svi
        # Save the updated SVI as an attribute of the instance
        setattr(self, 'svi', self.svi_)

        return self
    
    def _format_reinit_msg(self, name, kwargs=None, triggered_directly=True):
        """
        Constructs a message informing the user about the re-initialization of a component.

        When components such as modules or optimizers are re-initialized, this method 
        provides a formatted message detailing which component is re-initialized and which 
        specific parameters, if any, triggered the re-initialization.

        Parameters
        ----------
        name : str
            The name of the component that is being re-initialized.
        kwargs : dict, optional
            The parameters that caused the re-initialization. Defaults to None.
        triggered_directly : bool, optional
            Indicates if the re-initialization was directly triggered by a change in parameters. Defaults to True.

        Returns
        -------
        str
            A message formatted to inform about the component's re-initialization.

        Examples
        --------
        >>> self._format_reinit_msg('optimizer', {'lr': 0.01})
        'Re-initializing optimizer because the following parameters were re-set: lr.'

        Notes
        -----
        - If `kwargs` is None or empty, the message will not include information about parameters.
        - The `triggered_directly` argument is used to include parameters in the message only 
        if their change is the direct reason for re-initialization.
        """
        # Constructing the base message about re-initialization
        msg = "Re-initializing {}".format(name)
        # Adding details about the parameters if any were set and caused direct re-initialization
        if triggered_directly and kwargs:
            msg += (" because the following parameters were re-set: {}"
                    .format(', '.join(sorted(kwargs))))
        # Closing the message with a period
        msg += "."
        return msg
    
    @contextmanager
    def _current_init_context(self, name):
        """
        A context manager to temporarily set the current initialization context.

        This context manager is used to set a temporary state indicating the 
        name of the component that is currently being initialized. It helps in 
        keeping track of which component's initialization code is being executed, 
        especially when initialization methods can be nested or called multiple times.

        Parameters
        ----------
        name : str
            The name of the component that is being initialized.

        Yields
        ------
        None
            This method yields nothing and simply sets the context.

        Examples
        --------
        >>> with self._current_init_context('module'):
        ...     # initialize module here
        ...     pass
        >>> print(self.init_context_)
        None

        Notes
        -----
        - The initialization context is set to `name` when entering the context.
        - The initialization context is cleared (set to None) when exiting the context.
        """
        # Attempt to set the current initialization context to the given name
        try:
            self.init_context_ = name
            yield
        # Ensure that the initialization context is cleared after the block is exited
        finally:
            self.init_context_ = None

    def _initialize_virtual_params(self):
        """
        Initialize virtual parameters within a consistent initialization context.

        This method wraps the initialization of virtual parameters within a context manager that sets
        the current initialization context. Although the context ('virtual_params') is not utilized 
        at the moment, this approach maintains consistency with other initialization methods and
        allows for future expansion where the context might be necessary.

        Returns
        -------
        self : object
            The instance itself after initializing virtual parameters.

        Examples
        --------
        >>> self._initialize_virtual_params()
        <instance>  # The instance with virtual parameters initialized

        Notes
        -----
        - The method calls `initialize_virtual_params` which sets up a dictionary to manage virtual
        parameters.
        - The context manager `_current_init_context` is used here for consistency, although the 
        specific context set is not actively used within the method.
        - Virtual parameters are typically used to represent parameters that do not directly map to
        attributes but are controlled via specialized setter functions.
        """
        # Use the context manager for setting initialization context
        with self._current_init_context('virtual_params'):
            # Initialize the virtual parameters
            self.initialize_virtual_params()
            # Return the instance itself
            return self
        
    def _initialize_callbacks(self):
        """
        Initialize the callbacks for the instance within a consistent initialization context.

        This method handles the initialization of callbacks by checking if the callbacks are 
        explicitly disabled by the user. If not disabled, it proceeds to initialize them normally. 
        The process is wrapped within a context manager that establishes an initialization context, 
        even though it is not directly used by the method currently.

        Returns
        -------
        self : object
            The instance with its callbacks initialized or cleared based on the user's input.

        Examples
        --------
        >>> self._initialize_callbacks()
        <instance>  # The instance with callbacks initialized or disabled based on the configuration

        Notes
        -----
        - The initialization context ('callbacks') is set using `_current_init_context` for 
        consistency with the initialization flow of other components, despite not being used.
        - If `self.callbacks` is set to the string "disable", all callbacks are cleared by setting 
        `self.callbacks_` to an empty list. Otherwise, `initialize_callbacks` is called to set up 
        the callbacks.
        - This method allows for a user-configurable approach to managing callbacks, providing the 
        flexibility to enable or disable them as needed.
        """
        # Use the context manager for setting initialization context
        with self._current_init_context('callbacks'):
            # Check if callbacks are disabled
            if self.callbacks == "disable":
                # Clear all callbacks
                self.callbacks_ = []
            else:
                # Initialize the callbacks normally
                self.initialize_callbacks()
            # Return the instance itself
            return self
        
    def _initialize_criterion(self, reason=None):
        """
        Initialize the criterion within a consistent initialization context.

        This method is responsible for initializing the criterion specified in the `_criteria` attribute. 
        It gathers keyword arguments for all specified criteria and checks if any criterion requires 
        re-initialization based on these arguments or an external reason. If verbose logging is 
        enabled and initialization has already occurred, it will output a message indicating 
        that re-initialization is taking place.

        Parameters
        ----------
        reason : str, optional
            An optional message that explains why the criterion is being re-initialized. This is particularly useful when 
            re-initialization is triggered indirectly by other processes.

        Returns
        -------
        self : object
            The instance with its criterion initialized or re-initialized.

        Examples
        --------
        >>> self._initialize_criterion(reason="New parameter set.")
        <instance>  # The instance with the criterion re-initialized due to a new parameter setting.

        Notes
        -----
        - The initialization context ('criterion') is used to identify which component is 
        currently being initialized.
        - The method updates the criterion based on the current device configuration and compiles 
        it if necessary using `self.torch_compile`.
        - If a criterion is already initialized as a `torch.nn.Module`, it will be moved to the 
        appropriate device.
        - If `reason` is not provided but other conditions for re-initialization are met, 
        a re-initialization message is generated using `_format_reinit_msg`.
        - It is important to provide a `reason` when the re-initialization is part of a larger 
        workflow where the criterion needs to be reset indirectly, to maintain clarity for the user.
        """
        # Context manager for initialization
        with self._current_init_context('criterion'):
            # Gather keyword arguments for criteria
            kwargs = {}
            for criterion_name in self._criteria:
                kwargs.update(self.get_params_for(criterion_name))

            # Check if any criteria are already initialized
            has_init_criterion = any(
                isinstance(getattr(self, criterion_name + '_', None), torch.nn.Module)
                for criterion_name in self._criteria)

            # Determine if a re-init message is needed and print if verbose
            if kwargs or reason or has_init_criterion:
                if self.initialized_ and self.verbose:
                    msg = reason if reason else self._format_reinit_msg("criterion", kwargs)
                    print(msg)

            # Initialize the criterion
            self.initialize_criterion()

            # Set the criterion to the right device and compile
            for name in self._criteria:
                criterion = getattr(self, name + '_')
                if isinstance(criterion, torch.nn.Module):
                    criterion = to_device(criterion, self.device)
                    criterion = self.torch_compile(criterion, name=name)
                    setattr(self, name + '_', criterion)

            # Return the instance
            return self
        
    def _initialize_module(self, reason=None):
        """
        Initialize the modules within a consistent initialization context.

        This method handles the initialization or re-initialization of modules defined in the `_modules` 
        attribute. It aggregates parameters for each module and checks if any module requires re-initialization 
        due to new parameters or external reasons. When the instance is already initialized and verbosity is 
        enabled, it prints a message informing about the re-initialization.

        Parameters
        ----------
        reason : str, optional
            An optional string that describes why the module is being re-initialized. This can be particularly useful 
            when the re-initialization is a result of indirect actions within the model's workflow.

        Returns
        -------
        self : object
            The instance with its modules initialized or re-initialized.

        Examples
        --------
        >>> self._initialize_module(reason="Adjusting model architecture.")
        <instance>  # The instance with the modules re-initialized due to architecture adjustments.

        Notes
        -----
        - The initialization context ('module') specifies the component that is being initialized.
        - This method will move the module to the configured device and compile it if necessary using 
        `self.torch_compile`.
        - If a module has been previously initialized as a `torch.nn.Module`, it will be transferred to 
        the appropriate device.
        - In case `reason` is not specified but re-initialization is triggered, a default message is 
        created using the `_format_reinit_msg` function.
        - Providing a `reason` is advised when module re-initialization is triggered as part of a larger 
        process, to maintain transparency and provide context to the user.
        """
        with self._current_init_context('module'):
            # Compile keyword arguments for all modules
            kwargs = {}
            for module_name in self._modules:
                kwargs.update(self.get_params_for(module_name))

            # Determine if any modules are already initialized
            has_init_module = any(
                isinstance(getattr(self, module_name + '_', None), torch.nn.Module)
                for module_name in self._modules)

            # Log re-initialization message if necessary
            if kwargs or reason or has_init_module:
                if self.initialized_ and self.verbose:
                    msg = reason if reason else self._format_reinit_msg("module", kwargs)
                    print(msg)

            # Proceed with module initialization
            self.initialize_module()

            # Assign modules to the appropriate device and compile them
            for name in self._modules:
                try:
                    module = getattr(self, name + '_')
                except AttributeError:
                    # Fallback if the module is not found with underscore
                    module = getattr(self, name)
                    
                if isinstance(module, torch.nn.Module):
                    module = to_device(module, self.device)
                    module = self.torch_compile(module, name=name)
                    
                    # Set the module attribute correctly based on existing naming convention
                    if hasattr(self, name + '_'):
                        setattr(self, name + '_', module)
                    else:
                        setattr(self, name, module)

            # Return the instance for chaining
            return self
        
    def torch_compile(self, module, name):
        """
        Compiles a PyTorch module to potentially improve performance using the `torch.compile` API.

        This method is called to compile the PyTorch modules (like `module_` and `criterion_`) 
        when the `compile` attribute of the instance is set to `True`. If the attribute is set but 
        the installed PyTorch version does not support compiling, a `ValueError` is raised.

        Parameters
        ----------
        module : torch.nn.Module
            The PyTorch module to compile.
        name : str
            The name identifier for the module. This parameter is currently not utilized 
            in the method but can be used for conditional compilation based on module names.

        Returns
        -------
        torch.nn.Module or torch._dynamo.OptimizedModule
            The original module if `compile` is set to `False`, 
            or the compiled module if `compile` is `True`.

        Raises
        ------
        ValueError
            If `compile` is `True` but `torch.compile` is not available in the installed 
            PyTorch version.

        Examples
        --------
        >>> compiled_module = self.torch_compile(self.module_, 'module_')
        <compiled_module>  # The compiled module is returned if `compile` is `True`.

        Notes
        -----
        - This feature requires PyTorch version 1.14 or higher. Please ensure that the version of 
        PyTorch installed supports the `torch.compile` function.
        """
        if not self.compile:
            # Return the original module if compile is not enabled
            return module

        # Check if the torch.compile function is available
        torch_compile_available = hasattr(torch, 'compile')
        if not torch_compile_available:
            raise ValueError(
                "compile=True, but torch.compile is not available. Your installed PyTorch version is "
                f"{torch.__version__}, which is not compatible. Compilation requires PyTorch v1.14, v2.0 or higher."
            )

        # Get parameters for the compilation process
        params = self.get_params_for('compile')

        # Compile the module with the provided parameters
        module_compiled = torch.compile(module, **params)

        # Return the compiled module
        return module_compiled

    def get_all_learnable_params(self):
        """
        Generates name and parameter tuples for all learnable parameters across all modules.

        This method iterates over all modules defined within the neural network class and 
        yields their named parameters. This includes parameters of the primary module (`module_`),
        as well as any additional custom modules or parameters that are part of the criterion,
        provided they are learnable. The method ensures that each parameter is only returned once,
        even if it is referenced multiple times due to module re-use.

        Overrides of this method can customize which parameters are considered learnable and thus
        returned by this generator. This can be utilized to pair specific modules' parameters with
        dedicated optimizers during the initialization of optimizers.

        Yields
        ------
        tuple of (str, torch.nn.Parameter)
            Tuples of parameter names and the corresponding 
            parameters that are learnable.

        Examples
        --------
        >>> for name, param in self.get_all_learnable_params():
        >>>     print(name, type(param))
        'module_.weight' <class 'torch.nn.parameter.Parameter'>
        'module_.bias' <class 'torch.nn.parameter.Parameter'>
        # etc.

        Notes
        -----
        Custom modules or criterions with learnable parameters should ensure they implement
        the `named_parameters` method to be compatible with this generator.

        """
        # Use a set to track already seen parameters to avoid duplicates
        seen = set()

        for name in self._modules:
            # Attempt to access the module with a trailing underscore
            module = getattr(self, name + '_', None)

            # If not found, try accessing without the trailing underscore
            if module is None:
                module = getattr(self, name, None)
            
            # If the module is found, retrieve its named parameters if available
            if module is not None:
                named_parameters = getattr(module, 'named_parameters', None)
                if callable(named_parameters):
                    # Iterate through named parameters, filtering out duplicates
                    for param_name, param in named_parameters():
                        if param not in seen:
                            seen.add(param)
                            yield param_name, param

    def _initialize_optimizer(self, reason=None):
        """
        Initializes the optimizer(s) for the neural network.

        This method is called internally to set up the optimizers. If the network is already
        initialized and the verbose flag is set to True, it will print out a message indicating
        the re-initialization of the optimizer. The method also handles the registration of
        virtual parameters for the optimizers, which allows for a dynamic update of parameters
        post-initialization.

        Parameters
        ----------
        reason : str, optional
            A message indicating the reason for re-initialization, which 
            is printed out if provided. Defaults to None.

        Returns
        -------
        self
            Returns an instance of itself to allow for method chaining.
        
        Examples
        --------
        >>> net = NeuralNet(...)
        >>> net._initialize_optimizer()
        This returns the instance of `NeuralNet` after initializing the optimizer.

        Notes
        -----
        This method should not be called directly in most cases; it is intended to be used 
        internally by the network's initialization sequence.

        """
        # Set up the current context for initialization
        with self._current_init_context('optimizer'):
            # If the net is already initialized and verbosity is enabled, print the message
            if self.initialized_ and self.verbose:
                if reason:
                    # Indirect re-initialization, reason provided
                    msg = reason
                else:
                    # Direct re-initialization, format the default message
                    msg = self._format_reinit_msg("optimizer", triggered_directly=False)
                print(msg)

            # Perform the optimizer initialization
            self.initialize_optimizer()

            # Register virtual parameters for each optimizer for dynamic updates
            for name in self._optimizers:
                # Define the pattern for virtual parameter names
                param_pattern = [name + '__param_groups__*__*', name + '__*']
                if name == 'optimizer':  # Special case: 'lr' is short for 'optimizer__lr'
                    param_pattern.append('lr')

                # Set up a partial function for setting optimizer parameters
                setter = partial(
                    optimizer_setter,
                    optimizer_attr=name + '_',
                    optimizer_name=name,
                )

                # Register the virtual parameters with the setter function
                self._register_virtual_param(param_pattern, setter)

            return self
        
    def _initialize_history(self):
        """
        Initializes the history object for the model.

        This method sets up the history tracking for the neural network training process.
        It is called internally during the model's initialization process. The history object
        is responsible for recording the training metrics and validation metrics at each epoch.

        Returns
        -------
        self
            Returns an instance of itself to allow for method chaining.
        
        Examples
        --------
        >>> net = NeuralNet(...)
        >>> net._initialize_history()
        This returns the instance of `NeuralNet` after initializing the history object.

        Notes
        -----
        This method should not be called directly in most cases; it is intended to be used 
        internally by the network's initialization sequence.
        """
        # Set up the current context for initialization, although it's not used currently
        with self._current_init_context('history'):
            # Call the specific history initialization method
            self.initialize_history()
            
            return self

    def initialize(self):
        """
        Sequentially initializes all components of the model necessary for training.

        This method serves as an aggregation point for the initialization of different
        model components, including virtual parameters, callbacks, modules, criteria,
        optimizers, and history. It also performs additional initializations if certain
        conditions are met (e.g., initializing ELBO and SVI for probabilistic models),
        and validates the parameters before marking the model as initialized.

        Returns
        -------
        self
            Returns an instance of itself to allow for method chaining.

        Notes
        -----
        This method should be called before training the model to ensure all components
        are properly set up. It is implicitly called during the fitting process, so
        manual invocation is not typically required unless custom initialization logic
        is needed.

        Raises
        ------
        stockpy.exceptions.StockpyError
            If the model is not ready to be trained (determined
            by `check_training_readiness` method).
        """
        # First, check if the model is ready for training
        self.check_training_readiness()

        # Initialize components in sequence
        self._initialize_virtual_params()
        self._initialize_callbacks()
        self._initialize_module()
        self._initialize_optimizer()
        # Probabilistic models have additional components to initialize
        if self.prob is True:
            self.initialize_elbo()
            self.initialize_stochastic_variational_inference()
        else:
            self._initialize_criterion()
        self._initialize_history()
        
        # Validate all parameters to ensure proper model configuration
        self._validate_params()

        # Mark the model as initialized
        self.initialized_ = True
        
        return self

    def check_training_readiness(self):
        """
        Checks if the network is in a state that allows for training to begin.

        This method ensures that the network has not been altered in a way that would
        prevent training, such as having its parameters trimmed for the purpose of
        prediction. It should be called before training to validate the readiness of
        the network.

        Raises
        ------
        StockpyTrainingImpossibleError
            If the network has been trimmed for prediction
            and cannot be trained.

        Notes
        -----
        This method is generally called internally by the `initialize` method and
        does not need to be invoked directly by the user.
        """
        # Check if the network was trimmed for prediction and raise an error if so
        is_trimmed_for_prediction = getattr(self, '_trimmed_for_prediction', False)
        if is_trimmed_for_prediction:
            msg = (
                "The net's attributes were trimmed for prediction, thus it cannot "
                "be used for training anymore."
            )
            raise StockpyTrainingImpossibleError(msg)

    def check_data(self, X, y=None):
        pass

    def _set_training(self, training=True):
        """
        Sets the mode of all PyTorch modules and criteria to training or evaluation.

        This method controls the training/evaluation mode of all contained PyTorch modules
        by setting them accordingly. This affects layers like dropout and batchnorm which
        behave differently during training vs during evaluation (inference).

        Parameters
        ----------
        training : bool, optional (default=True)
            Determines the mode to set:
            - `True` will set all modules and criteria to training mode.
            - `False` will set all modules and criteria to evaluation mode.

        Examples
        --------
        >>> model = YourModel()
        >>> model._set_training(True)  # Switch to training mode
        >>> model._set_training(False) # Switch to evaluation mode

        Notes
        -----
        - This method should be used to switch modes before training or evaluating the model.
        - It does not return anything but has an in-place effect on the modules and criteria.

        See Also
        --------
        train : Inherited method from `torch.nn.Module` that is used internally to set the mode.
        """
        
        self.train(training)

    @torch.no_grad()
    def validation_step(self, batch, **fit_params):
        """
        Perform a validation step, compute and return the loss, and possibly predictions.

        During validation, the module is set to evaluation mode (`module.eval()`) to deactivate
        specific layers (e.g., dropout, batch normalization) that should behave differently during 
        validation. The method assumes that `batch` contains both features and targets, and does 
        not track gradients to improve performance.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            The batch data to be validated, where the first element is assumed to be the input
            features and the second element the target labels.
        **fit_params : dict, optional
            Arbitrary keyword arguments. These arguments are passed to the `forward` method of
            the module and any functions that override this method.

        Returns
        -------
        dict
            A dictionary containing the following keys:
            - 'loss' (torch.Tensor): The loss computed for the batch, which is a scalar tensor.
            - 'y_pred' (optional, torch.Tensor): The predictions made by the module on the input features.
            This is returned only when `self.prob` is False.

        Examples
        --------
        >>> batch = (X_val_tensor, y_val_tensor)
        >>> validation_output = model.validation_step(batch)
        >>> print(validation_output['loss'])  # Prints the validation loss
        >>> if 'y_pred' in validation_output:
        ...     print(validation_output['y_pred'])  # Prints the predictions if available

        Notes
        -----
        - The `torch.no_grad()` context manager is used to disable gradient calculation
        during the forward pass, reducing memory usage and speeding up computation.
        - If `self.prob` is True, this method is part of a probabilistic modeling approach
        and will not return predictions, only the loss.

        Raises
        ------
        NotImplementedError
            If `self.prob` is True but the necessary methods (e.g., `self.svi_.evaluate_loss`)
            are not implemented.

        """

        self._set_training(False)  # Set the module to evaluation mode.
        
        Xi, yi = unpack_data(batch)  # Unpack the features and labels.
        
        # If we are not in probabilistic mode, predict and calculate the loss normally.
        if not self.prob:
            y_pred = self.infer(Xi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=False)
            return {'loss': loss, 'y_pred': y_pred}
        
        # For probabilistic modeling, compute the loss differently.
        if hasattr(self, 'svi_'):
            loss = self.svi_.evaluate_loss(x=Xi, y=yi) 
            y_pred = self.infer(Xi, **fit_params)
            return {'loss': loss, 'y_pred': y_pred}
        else:
            raise NotImplementedError("Probabilistic validation requires an `svi_` attribute.")

    def train_step_single(self, batch, **fit_params):
        """
        Perform a single training step including forward pass, loss computation, and backpropagation.

        In this training step, the module is set to training mode, which activates layers like
        dropout and batch normalization. A forward pass is conducted to predict outputs based on
        the input batch, and the loss is computed. The gradients are then backpropagated through
        the network's parameters. In probabilistic mode, the method adapts to perform a step of
        stochastic variational inference instead.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch from the training data loader, where the first element is the input features
            and the second element is the corresponding targets.
        **fit_params : dict, optional
            Additional keyword arguments for the `forward` method of the module.

        Returns
        -------
        dict
            A dictionary with the following key-value pairs:
            - 'loss' (torch.Tensor): The computed loss for the batch as a scalar tensor.
            - 'y_pred' (optional, torch.Tensor): The predictions made by the module for the input features. 
            This is not returned if the module is in probabilistic mode (`self.prob` is True).

        Raises
        ------
        NotImplementedError
            If `self.prob` is True but the `self.svi_` attribute is not present or the `step`
            method is not implemented on `self.svi_`.

        Examples
        --------
        >>> batch = (X_train_tensor, y_train_tensor)
        >>> train_step_output = model.train_step_single(batch)
        >>> print(train_step_output['loss'])  # Prints the loss for the training step
        >>> if 'y_pred' in train_step_output:
        ...     print(train_step_output['y_pred'])  # Prints predictions if available

        Notes
        -----
        - If `self.prob` is True, it is assumed that `self.svi_` has been properly initialized and
        contains a `step` method to perform a step of stochastic variational inference. The loss
        returned in this case corresponds to the negative ELBO (Evidence Lower Bound) and is
        typically a positive value.
        - During backpropagation, only the parameters of the module that have `requires_grad` set to
        True will have their gradients updated.

        """

        self._set_training(True)  # Ensure the module is in training mode.
        
        Xi, yi = unpack_data(batch)  # Unpack input features and targets from the batch.
        
        # Handle non-probabilistic mode: forward pass, loss computation, backpropagation.
        if not self.prob:
            y_pred = self.infer(Xi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=True)
            loss.backward()  # Backpropagate to compute gradients.
            return {'loss': loss, 'y_pred': y_pred}
        
        # Handle probabilistic mode: perform an SVI step.
        if hasattr(self, 'svi_'):
            loss = self.svi_.step(x=Xi, y=yi)
            y_pred = self.infer(Xi, **fit_params)
            return {'loss': loss, 'y_pred': y_pred}
        else:
            raise NotImplementedError("Probabilistic training requires an `svi_` attribute with a `step` method.")
        
    def get_train_step_accumulator(self):
        """
        Creates and returns an accumulator for the training step results.

        In the training loop, each step typically involves a call to an optimizer, which
        returns a value (usually the loss). An accumulator is used to collect and store
        these values. The default behavior is to store the first value returned from
        the optimizer's call during each training step, which is suitable for most cases.

        If an optimizer performs multiple evaluations of the loss function within a single
        training step (e.g., as is the case with the LBFGS optimizer), it may be desirable
        to collect more than the first loss value. In such cases, this method can be
        overridden to return a custom accumulator that collects the required data.

        Returns
        -------
        FirstStepAccumulator
            An instance of `FirstStepAccumulator`, the default accumulator class,
            which is tailored to store the first value returned by the optimizer's step
            during a training step.

        Notes
        -----
        - The accumulator pattern used here allows for flexibility in how optimization
        results are collected during training. Users can create custom accumulator
        classes to capture different statistics or behaviors of the optimizer.
        - By default, the `FirstStepAccumulator` is suitable for most optimizers that
        return a single loss value per step. However, optimizers like LBFGS, which
        make multiple loss evaluations per step, may require a different strategy for
        accumulation, warranting a custom implementation.

        """

        return FirstStepAccumulator()

    def _zero_grad_optimizer(self, set_to_none=None):
        """
        Zero out the gradients of all optimizers in the model or optionally set them to `None`.

        This function is responsible for preparing the optimizers before a new optimization
        step begins. It is a wrapper around the `zero_grad()` method provided by PyTorch
        optimizers and allows for the gradients to be explicitly set to `None`, which can be
        beneficial for optimizing memory usage in certain situations.

        Parameters
        ----------
        set_to_none : bool or None, optional
            When `True`, gradients will be set to `None` instead of being zeroed. This can
            prevent unnecessary memory operations, potentially leading to performance
            improvements. If `False`, gradients are zeroed in the conventional manner.
            By default (`None`), the gradients are zeroed, which is the typical and expected
            behavior unless explicitly overridden.

        Returns
        -------
        None

        See Also
        --------
        torch.optim.Optimizer.zero_grad : Underlying PyTorch method used to zero gradients.

        Notes
        -----
        - Setting gradients to `None` was introduced in PyTorch 1.7 and can be a subtle
        optimization. It may result in different behavior if gradients are accessed before
        the next backward pass as they won't be explicitly zero but will be `None` instead.
        - It's important to ensure that all backward passes are complete and that you do not
        expect to accumulate gradients before calling this function with `set_to_none=True`.
        """
        
        # Check if `set_to_none` is specified and act accordingly
        self.optimizer_.zero_grad(set_to_none=bool(set_to_none))

    def _step_optimizer(self, step_fn=None):
        """
        Perform an optimization step using the registered optimizer.

        This function takes a single optimization step for the optimizer linked with the model.
        The step can be a simple gradient descent or a more complex procedure, depending on the
        optimizer and the provided `step_fn` closure, if any. The closure `step_fn` should
        calculate the loss and perform a backward pass, which is typically necessary for
        optimizers that require multiple function evaluations per optimization step, such as LBFGS.

        Parameters
        ----------
        step_fn : callable, optional
            A closure that reevaluates the model and returns the loss. This is necessary for
            certain optimization algorithms that perform multiple evaluations under the hood
            during a single optimization step (e.g., LBFGS). If not provided, the optimizer
            will simply perform a single step based on gradients computed externally.

        Returns
        -------
        None

        Examples
        --------
        >>> # Assuming `model` has been instantiated and contains a registered optimizer
        >>> def closure():
        >>>     model._zero_grad_optimizer()
        >>>     loss = model.compute_loss(data, targets)
        >>>     loss.backward()
        >>>     return loss
        >>> model._step_optimizer(step_fn=closure)

        Notes
        -----
        - The use of a closure is specific to certain optimizers and is not a common practice
        for optimizers like SGD or Adam that don't require it.
        - Overriding this method can be useful for implementing custom optimization procedures
        or integrating complex algorithms that deviate from the typical single-step update.

        See Also
        --------
        torch.optim.Optimizer.step : The underlying method called on the optimizer.
        """
        
        # If a step function is provided, use it; otherwise, call step without arguments
        if step_fn is not None:
            self.optimizer_.step(step_fn)
        else:
            self.optimizer_.step()

    def train_step(self, batch, **fit_params):
        """
        Perform a single optimization step during training.

        This function is responsible for orchestrating the training process for a single batch.
        It sets the model to training mode, computes the loss via `train_step_single`, accumulates
        the step information, and then uses the optimizer to update the model's weights.

        Parameters
        ----------
        batch : iterable
            A single batch of data provided by the data loader, typically a tuple of
            tensors containing input features and the corresponding target labels.
                    
        **fit_params : dict
            Arbitrary keyword arguments that will be passed directly to the `forward`
            method of the model, as well as any other method that accepts `**fit_params`.

        Returns
        -------
        dict
            A dictionary with at least the following two entries:
            - 'loss': The scalar loss value for the given batch.
            - 'y_pred': The predictions made by the model for the input features in the batch.

        Raises
        ------
        ValueError
            If `batch` does not have the expected format (input features, target labels).

        Examples
        --------
        >>> batch = next(iter(train_loader))  # Assume `train_loader` is a DataLoader instance
        >>> fit_params = {'additional_parameter': value}
        >>> training_step_output = model.train_step(batch, **fit_params)
        >>> loss, y_pred = training_step_output['loss'], training_step_output['y_pred']

        Notes
        -----
        - The actual optimization step is conditionally executed depending on the `prob` attribute.
        If `prob` is False, a standard optimization step using backpropagation is performed.
        Otherwise, a custom step function defined within this method is executed.
        - The `step_accumulator` is used to store and return the optimization results.
        This is useful for advanced optimization procedures that may require access to
        these intermediate results.
        - This method may be overridden to customize the training loop for specific requirements.

        """
        
        # Switch model to training mode
        self._set_training(True)
        
        # Initialize an accumulator to collect steps
        step_accumulator = self.get_train_step_accumulator()
        
        # Define the step function to be used for optimization
        def step_fn():
            # Zero the gradients if the model is not probabilistic
            if self.prob is False: 
                self._zero_grad_optimizer()
            
            # Compute loss and potentially other metrics for a single training step
            step = self.train_step_single(batch, **fit_params)
            
            # Store this step's data in the accumulator
            step_accumulator.store_step(step)
            
            # Trigger any callbacks or hooks related to gradient computation
            self.notify(
                'on_grad_computed',
                named_parameters=TeeGenerator(self.get_all_learnable_params()),
                batch=batch,
                training=True,
            )
            
            # The loss value is returned because it may be needed for the optimizer's `step` method
            return step['loss']
        
        # Execute the optimization step
        if self.prob is False:
            self._step_optimizer(step_fn)
        else: 
            # For probabilistic models, call step_fn directly, which might include sampling procedures
            step_fn()
        
        # Return the collected steps from the accumulator
        return step_accumulator.get_step()

    def evaluation_step(self, batch, training=False):
        """
        Perform a forward step to obtain the model's predictions.

        This function conducts a forward pass using the provided data batch to obtain predictions from the model.
        It is used during both evaluation and prediction phases. By default, the model is set to evaluation mode
        by disabling training-specific layers like dropout. The mode can be overridden by setting the `training`
        parameter to True.

        Parameters
        ----------
        batch : iterable
            A single batch of data obtained from a DataLoader, which typically contains the input features and
            the target labels for evaluation purposes.

        training : bool, optional
            A flag that determines the mode of the model during this forward pass. If set to True, the model
            will be in training mode, otherwise it will be in evaluation mode. The default value is False.

        Returns
        -------
        torch.Tensor
            The predictions generated by the model for the input batch.

        Raises
        ------
        NotFittedError
            If the model has not been trained (fitted), this error is raised indicating that the evaluation cannot
            be performed.

        Examples
        --------
        >>> validation_batch = next(iter(validation_loader))  # Assume `validation_loader` is a DataLoader instance
        >>> predictions = model.evaluation_step(validation_batch)
        >>> # Now, `predictions` holds the model's predictions for the validation batch

        Notes
        -----
        - This method should be used when evaluating the model's performance on a validation set or when making
        predictions on a new dataset.
        - The model's state (training or evaluation) is managed internally by this function based on the `training`
        flag, ensuring that layers like BatchNorm and Dropout work in the correct mode.

        """
        # Ensure the model has been trained before attempting to make predictions
        self.check_is_fitted()

        # Unpack the data to separate inputs (features) from targets
        Xi, _ = unpack_data(batch)

        # Use context manager to set gradient computation based on the training mode
        with torch.set_grad_enabled(training):
            # Set the model to the appropriate mode (train/eval) based on the `training` flag
            self._set_training(training)

            # Forward pass to get predictions
            predictions = self.infer(Xi)

            return predictions

    def fit_loop(self, X, y=None, **fit_params):
        """
        Execute the main training loop to fit the model.

        The fit loop is the core of the model's training process. It accepts input data and targets,
        prepares them for training, and iterates through the specified number of epochs, updating the
        model's parameters with each batch processed. It also manages the data splitting into training
        and validation sets, and alternates between training and validation phases within each epoch.

        Parameters
        ----------
        X : various
            Input data, compatible with the formats accepted by stockpy.dataset.Dataset. This can be
            numpy arrays, PyTorch tensors, pandas DataFrame or Series, scipy sparse CSR matrices, and
            more, including lists/tuples or dictionaries of these types or a custom Dataset object.

        y : various, optional
            True targets corresponding to `X`. Should be the same format as `X` and can be omitted if
            `X` is a Dataset that already includes targets.

        **fit_params : dict
            Additional keyword arguments dynamically passed to the model's `forward` method and to the
            train/validation data splitting routine.

        Returns
        -------
        self : object
            The instance of this class, updated to represent the fitted model.

        Raises
        ------
        ValueError
            Raised if `X` or `y` are not in a format compatible with the underlying dataset and model
            requirements.

        Notes
        -----
        - The actual number of epochs for training is determined by the `self.epochs` attribute, which
        can be set directly or passed as an argument in `fit_params`.
        - The `fit_loop` also manages any necessary seeding for reproducibility if the model's `prob`
        attribute is set to True.
        - If a validation dataset is present, the model will alternate between training on the training
        set and evaluating on the validation set each epoch.

        """
        # Check if the input data is compatible with the expected format
        self.check_data(X, y)
        
        # Prepare the model for training, checking initial conditions and setting states
        self.check_training_readiness()
        
        # Split the data into training and validation sets if necessary
        dataset_train, dataset_valid = self.get_split_datasets(X, y, **fit_params)
        
        # Construct arguments for epoch-level callbacks
        on_epoch_kwargs = {
            'dataset_train': dataset_train,
            'dataset_valid': dataset_valid,
        }

        # Setup the iterators for going through the training and validation datasets
        iterator_train = self.get_iterator(dataset_train, training=True)
        iterator_valid = None  # Initialize validation iterator, to be set if validation data exists

        if dataset_valid is not None:
            iterator_valid = self.get_iterator(dataset_valid, training=False)

        # Determine the number of epochs from the model's settings or parameters
        epochs = self.epochs if 'epochs' not in fit_params else fit_params['epochs']
        
        # Setup for probabilistic model training, with appropriate seeding if required
        if self.prob is True:
            pyro.clear_param_store()

        # Iterate over each epoch to train the model
        for _ in range(epochs):

            self.notify('on_epoch_begin', **on_epoch_kwargs)
            
            # Execute training and validation for the current epoch
            self.run_single_epoch(iterator_train, training=True, prefix="train",
                                step_fn=self.train_step, **fit_params)

            self.run_single_epoch(iterator_valid, training=False, prefix="valid",
                                step_fn=self.validation_step, **fit_params)

            # Notify listeners that the epoch has ended
            self.notify("on_epoch_end", **on_epoch_kwargs)

        return self
    
    def run_single_epoch(self, iterator, training, prefix, step_fn, **fit_params):
        """
        Compute and record a single epoch of training or validation.

        During the epoch, the function iterates over batches of data provided by the `iterator`,
        applies the `step_fn` to each batch, and records the results. This function is integral
        to the fitting process, orchestrating the lower-level batch-wise operations and ensuring
        the side-effects (like logging and invoking callbacks) are consistently applied.

        Parameters
        ----------
        iterator : torch.utils.data.DataLoader or None
            The DataLoader that provides batches of data. If this is None, the epoch is skipped.
            This allows for conditional execution of training/validation epochs.

        training : bool
            Indicates whether the model should be in training mode (True) or evaluation mode (False).
            This affects operations like dropout or batch normalization.

        prefix : str
            A prefix string added to the metrics recorded for this epoch in the model's history.
            Typical values are 'train' or 'valid', indicating the phase of the epoch.

        step_fn : callable
            A function to be called on each batch of data. In the case of training, this might be
            a function that computes the loss and performs a backpropagation step. For validation,
            it might simply evaluate the model's performance on the batch.

        **fit_params : dict
            Arbitrary keyword arguments that are passed through to the `step_fn`. This provides
            flexibility for `step_fn` to accept training-specific parameters.

        Notes
        -----
        - The function handles both the recording of batch-specific metrics (like loss) and
        batch sizes to the model's history, ensuring that they can be analyzed after the
        epoch completes.
        - The `notify` method calls are used to invoke any callbacks registered to the
        'on_batch_begin' and 'on_batch_end' events, allowing for custom behavior to be
        executed at these points in the training process.
        - Probabilistic models may not use a concrete `.item()` value for loss, in which
        case the raw tensor is recorded in the history.
        - The `batch_count` recorded at the end of the epoch gives a total count of batches
        processed, which can be useful for understanding the scale of each epoch.

        """
        
        # If no iterator is provided, there is nothing to do for this epoch
        if iterator is None:
            return

        # Initialize a counter for the number of batches processed
        batch_count = 0

        # Iterate over all batches provided by the DataLoader
        for batch in iterator:
            # Notify any listeners that a new batch is starting
            self.notify("on_batch_begin", batch=batch, training=training)
            
            # Apply the step function to the current batch
            step = step_fn(batch, **fit_params)

            # Record the loss; if the model is probabilistic, the loss may not be a scalar
            loss_record = step["loss"].item() if self.prob is False else step["loss"]
            self.history.record_batch(prefix + "_loss", loss_record)
            
            # Determine and record the batch size
            batch_size = (get_len(batch[0]) if isinstance(batch, (tuple, list)) else get_len(batch))
            self.history.record_batch(prefix + "_batch_size", batch_size)
            
            # Notify any listeners that the batch has ended
            self.notify("on_batch_end", batch=batch, training=training, **step)

            # Increment the count of batches processed during this epoch
            batch_count += 1

        # Record the total number of batches processed in this epoch
        self.history.record(prefix + "_batch_count", batch_count)
    
    def partial_fit(self, X, y=None, **fit_params):
        """
        Partially fit the module on the provided data without re-initializing.

        This method allows for incremental training of the model, often referred to as a 'warm start'.
        It's particularly useful when you have new data and want to update the model without
        retraining from scratch. The internal parameters of the model are preserved, and the training
        continues from the current state.

        Parameters
        ----------
        X : Various types
            The input data. It must be compatible with `stockpy.dataset.StockpyDataset`. Supported types include:
            - NumPy arrays
            - PyTorch tensors
            - pandas DataFrame or Series
            - SciPy sparse CSR matrices
            - Dictionaries containing any combination of the above
            - Lists or tuples containing any combination of the above
            - A custom Dataset object capable of handling the input data format

            If the data format is not supported, the method will fail to proceed.

        y : Various types, optional
            The target data corresponding to `X`. It should support the same types as `X`.
            If `X` is already a Dataset containing the target data, then `y` should be None.
            Default is None.

        classes : array-like, shape (n_classes,), optional
            This parameter exists for compatibility with scikit-learn's interface but is not
            used within this method. Defaults to None.

        **fit_params : dict
            Additional keyword arguments that are passed to the `forward` method of the model and to
            the `self.train_split` method during the fitting process. These parameters can control
            training behaviors, such as weighting classes differently during the loss calculation.

        Returns
        -------
        self : object
            This method returns the instance itself, with updated parameters reflecting the
            training that has taken place.

        Notes
        -----
        - Before invoking this method, ensure that the model's architecture and hyperparameters
        are correctly configured, as this method will continue training from the current state
        of the model.
        - The method will automatically initialize the model if it hasn't been initialized already.
        This is useful when you've created a new instance and want to start training it incrementally.
        - Partial fitting is often used in online learning scenarios, where the model is updated as
        new data arrives.
        - If the training process is interrupted, for example by a KeyboardInterrupt, the method
        will catch the interruption and the model will remain in its current state.

        """
        
        # Ensure the model is initialized before proceeding
        if not self.initialized_:
            self.initialize()

        # Inform any listeners that training is about to begin
        self.notify('on_train_begin', X=X, y=y)

        # Attempt the fitting process within a try-except block to handle unexpected interruptions
        try:
            # Continue the fitting process with the new data
            self.fit_loop(X, y, **fit_params)
        except KeyboardInterrupt:
            # Training was interrupted, potentially capture state or take other actions
            pass

        # Inform any listeners that training has finished
        self.notify('on_train_end', X=X, y=y)

        # Return the model instance
        return self

    def fit(self, 
            X, 
            y=None, 
            optimizer=torch.optim.SGD,
            elbo=TraceMeanField_ELBO,
            callbacks=None,
            lr=0.01,
            epochs=10,
            batch_size=32,
            shuffle=False,
            verbose=1,
            model_params=None,
            warm_start=False,
            train_split=ValidSplit(5),
            **fit_params):
        """
        Fit the model to the input data X and target y.

        This method is designed to handle the training process of the neural network.
        It supports a variety of data types for input and can work with or without explicit target labels.
        The method offers customization through a range of parameters, including optimizer choice,
        learning rate, and others. Notably, this method supports variational inference with a
        customizable Evidence Lower Bound (ELBO) for probabilistic models.

        Parameters
        ----------
        X : array-like or Dataset
            The input data for training the model. Acceptable formats include:
            - NumPy arrays
            - PyTorch tensors
            - Pandas DataFrame or Series
            - SciPy sparse CSR matrices
            - Dictionaries containing a combination of the above types
            - Lists or tuples containing a combination of the above types
            - Custom Dataset objects
        y : array-like, optional
            The target values for supervised learning. It can be set to None if targets are included
            within the input data X. The method ensures that y, if provided, is properly formatted
            for the training process.
        optimizer : torch.optim.Optimizer, default=torch.optim.SGD
            The optimizer class to be used for training the model. Defaults to stochastic gradient descent.
        elbo : type, default=TraceMeanField_ELBO
            The ELBO class to be used for probabilistic models during the training. Only used if `self.prob` is True.
        callbacks : list of callback functions, optional
            A list of callback functions that can be used to monitor and influence the training process.
        lr : float, default=0.01
            The learning rate for the optimizer.
        epochs : int, default=10
            The number of epochs to train the model.
        batch_size : int, default=32
            The size of the batches for training.
        shuffle : bool, default=False
            Whether to shuffle the data before each epoch.
        verbose : int, default=1
            Verbosity mode; 0 = silent, 1 = progress bar, 2 = one line per epoch.
        model_params : dict, optional
            A dictionary containing parameters for model initialization.
        warm_start : bool, default=False
            If set to True, the model is not re-initialized and continues training from where it left off.
        train_split : ValidSplit, default=ValidSplit(5)
            The strategy to use for splitting training data from validation data.
        **fit_params : dict
            Additional parameters to pass to the fit method.

        Returns
        -------
        self : object
            This method returns the current object instance after completing the training process, 
            which allows for method chaining.

        Raises
        ------
        ValueError
            If input X is of invalid shape or type, an error is raised.

        Notes
        -----
        - It's crucial to provide data in a supported format to avoid issues during the training process.
        - If using a probabilistic model, ensure that `self.prob` is set to True to make use of the ELBO.
        - The `fit` method supports a warm start mechanism to continue training without resetting the model's parameters.

        """
            
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        if self.prob is True:
            self.elbo = elbo
        self.callbacks = callbacks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.model_params = model_params
        self.warm_start = warm_start
        self.train_split = train_split

        X, y = self._validate_data(
            X, y, accept_sparse=["csr", "csc", "coo"], multi_output=True
        )

        # Ensure y is 2D
        if get_dim(y) == 1:
            y = y.reshape((-1, 1))
            if isinstance(self, Classifier):
                self.n_classes_ = len(unique_labels(y))

        self.n_outputs_ = y.shape[1]
        
        # Check if the model should be re-initialized. If warm_start is True and the
        # model is already initialized, skip the re-initialization.
        if not self.warm_start or not self.initialized_:
            self.initialize()

        # if model_params is not None load the model parameters
        if self.model_params is not None:
            print('Loading model parameters...')
            self.load_params(f_history=self.model_params.get('f_history', None),
                             f_optimizer=self.model_params.get('f_optimizer', None),
                             f_params=self.model_params.get('f_params', None))
                             
        # Perform the partial fit, which is the actual fitting process.
        self.partial_fit(X, y, **fit_params)
        
        return self

    def check_is_fitted(self, attributes=None, *args, **kwargs):
        """
        Ensure that the neural network model is initialized by checking for specific attributes.

        This method determines if the model has been fitted by checking for the existence of
        specific attributes that are set during the initialization process. It serves as a guard
        to prevent the use of model methods that require the model to be initialized first.

        Parameters
        ----------
        attributes : list of str or None, optional
            A list of strings representing the names of attributes that the model is expected to have
            once it is properly fitted. If None (default), the method checks for the presence of
            attributes derived from the names in `self._modules` with an appended underscore, or
            checks for the 'module_' attribute if `self._modules` is empty.
        *args : list
            Additional positional arguments that should be passed to the `check_is_fitted` function
            from scikit-learn. This allows for flexibility in the underlying check.
        **kwargs : dict
            Additional keyword arguments that should be passed to the `check_is_fitted` function
            from scikit-learn. This permits custom checks and messages.

        Raises
        ------
        NotInitializedError
            If any of the specified attributes do not exist, indicating that the model is not
            fitted or initialized, a `NotInitializedError` from the `stockpy.exceptions` module
            is raised.

        Notes
        -----
        - It's generally a good practice to call this method before performing operations that
        require a fitted model, such as predicting or updating parameters.

        - The `check_is_fitted` function from scikit-learn is used internally, so the model
        adheres to scikit-learn's conventions on fitted models.

        """

        if attributes is None:
            if self._modules:
                attributes = [module for module in self._modules.keys()]
            else:
                attributes = ['module_']
        
        check_is_fitted(self, attributes, *args, **kwargs)

    def trim_for_prediction(self):
        """
        Streamline the model for prediction by removing non-essential attributes.

        This method aims to optimize the memory footprint of the model post-training by
        discarding attributes that are solely required during training, such as optimizers
        and callbacks. Post-trimming, the model is constrained to prediction tasks only.

        Raises
        ------
        NotInitializedError
            If the model has not been fitted, an error is raised to prevent trimming of an
            uninitialized model.

        Notes
        -----
        - This method irreversibly alters the model. Post-execution, the model cannot be
        reverted to its pre-trimmed state or be further trained without complete
        re-initialization.
        - It is recommended to deepcopy the model before trimming if there is a potential need for
        further training or to examine training-related attributes later.
        - After trimming, any attempt to retrain the model will require re-initialization or
        re-instantiation of the model object.

        """
        
        # Check if the model is already trimmed for prediction. If yes, do nothing.
        if getattr(self, '_trimmed_for_prediction', False):
            return
        
        # Check if the model is initialized and has been fitted.
        self.check_is_fitted()
        
        # Set internal flag to indicate that the model is trimmed for prediction.
        # pylint: disable=attribute-defined-outside-init
        self._trimmed_for_prediction = True
        
        # Disable the training mode of the model.
        self._set_training(False)

        if isinstance(self.callbacks, list):
            self.callbacks.clear()
        self.callbacks_.clear()

    def forward_iter(self, X, training=False, device='cpu'):
        """
        Iterate over the input data in batches and perform forward calls to the module.

        This method allows for batch-wise generation of model outputs. It is especially useful
        when the dataset is too large to be processed at once, allowing for memory-efficient
        inference.

        Parameters
        ----------
        X : various types
            The input data. It can be one of the following:
            - numpy arrays
            - torch tensors
            - pandas DataFrame or Series
            - scipy sparse CSR matrices
            - a dictionary comprising the above types
            - a list or tuple containing the above types
            - a custom Dataset object that can handle the data
            
            The method is designed to handle a wide range of input types, but custom data types
            may require a corresponding custom Dataset object.

        training : bool, optional (default=False)
            If set to True, it will set the module to training mode before the forward call.
            Otherwise, the module will be set to evaluation mode.

        device : str, optional (default='cpu')
            The device to which the output tensor will be moved before yielding. It can be
            set to 'cpu' or a specific 'cuda' device (e.g., 'cuda:0') depending on the
            availability and requirement for GPU acceleration.

        Yields
        ------
        torch.Tensor
            The output tensor from the module for each batch.

        Notes
        -----
        Remember to convert your data to the appropriate torch Tensor and device before using
        this function if your data is not in one of the supported formats.
        """

        # Get the dataset object from the input data
        dataset = self.get_dataset(X)
        
        # Create an iterator for the dataset, optionally set to training mode
        iterator = self.get_iterator(dataset, training=training)

        # Loop through the dataset one batch at a time
        for batch in iterator:
            # Perform the forward operation for the current batch
            yp = self.evaluation_step(batch, training=training)

            # Move the output tensor to the specified device and yield
            yield to_device(yp, device=device)

    def forward(self, X, training=False, device='cpu'):
        """
        Perform a forward pass over the input data and concatenate the outputs.

        This method runs the forward pass of the model over all batches of the input data
        using `self.forward_iter`, and concatenates the resulting tensors into a single
        tensor (or a tuple of tensors in case of multiple outputs).

        Parameters
        ----------
        X : various types
            The input data. Acceptable formats include:
            - numpy arrays
            - torch tensors
            - pandas DataFrame or Series
            - scipy sparse CSR matrices
            - a dictionary with values being any of the above types
            - a list or tuple of any of the above types
            - a Dataset object
            If X is not in a compatible format, a custom `Dataset` object should be provided.

        training : bool, optional
            Flag indicating whether to set the model in training mode (True) or evaluation mode
            (False). Defaults to False.

        device : str, optional
            The device on which to perform the computation. Defaults to 'cpu' but can be set
            to a CUDA device as well, such as 'cuda:0' for GPU acceleration.

        Returns
        -------
        y_infer : torch.Tensor or tuple of torch.Tensor
            The concatenated output tensor from the model's forward pass. If the model's
            forward method returns multiple outputs, then a tuple of tensors is returned
            with each tensor corresponding to one of the outputs.

        Raises
        ------
        ValueError
            If the input data X is empty or the forward pass returns inconsistent output
            sizes that cannot be concatenated.

        Notes
        -----
        Ensure that the input X is properly preprocessed and converted to a torch Tensor
        before using this method. The data should also be moved to the correct device, if
        necessary.
        """

        # Create a list of output tensors by iterating through batches.
        y_infer = list(self.forward_iter(X, training=training, device=device))

        # Check if there are multiple outputs (tuple) for each batch.
        is_multioutput = len(y_infer) > 0 and isinstance(y_infer[0], tuple)

        # If there are multiple outputs, concatenate each one separately.
        if is_multioutput:
            return tuple(map(torch.cat, zip(*y_infer)))
        
        # For single output, concatenate to form a single tensor.
        return torch.cat(y_infer)
    
    def _merge_x_and_fit_params(self, x, fit_params):
        """
        Merge input data `x` and additional fitting parameters `fit_params`.

        This utility function takes the input data and additional fitting parameters,
        provided as two dictionaries, and merges them into a single dictionary. If
        there are overlapping keys between the two dictionaries, a ValueError is raised
        to prevent unintended overwriting of parameters.

        Parameters
        ----------
        x : dict
            A dictionary containing input data as key-value pairs where the keys are
            data identifiers (e.g., feature names) and the values are the data.

        fit_params : dict
            A dictionary containing additional fitting parameters where keys are parameter
            names and values are the corresponding values for those parameters.

        Returns
        -------
        x_dict : dict
            A new dictionary resulting from the merger of `x` and `fit_params`, containing
            all key-value pairs from both.

        Raises
        ------
        ValueError
            If there are overlapping keys in `x` and `fit_params`, indicating a conflict
            between input data identifiers and fitting parameter names.

        Notes
        -----
        - It's assumed that `x` and `fit_params` do not have overlapping keys, as they
        represent distinct sets of information; `x` is for input data, and `fit_params`
        is for hyperparameters and other training-related settings.
        - The merge is necessary to streamline the process of feeding both data and
        parameters into the fitting process, ensuring all necessary information is
        accessible in one container.
        """
        # Check for duplicate keys between 'x' and 'fit_params'.
        duplicates = duplicate_items(x, fit_params)
        if duplicates:
            # Raise an error if duplicate keys are found.
            msg = "X and fit_params contain duplicate keys: "
            msg += ', '.join(duplicates)
            raise ValueError(msg)

        # Perform a shallow copy of 'x' to ensure original data is not modified.
        x_dict = dict(x)
        
        # Update the copied dictionary with items from 'fit_params'.
        x_dict.update(fit_params)
        
        # Return the merged dictionary.
        return x_dict
            
    def infer(self, x, y=None, **fit_params):
        """
        Perform a single inference step on a batch of data.

        This method processes the input data `x`, and if provided, the target data `y`, 
        along with additional fitting parameters. It ensures the data is in tensor form 
        and conducts a forward pass through the network to generate predictions.

        Parameters
        ----------
        x : Any
            Input data for inference. This can be a batch of data in various forms such as 
            numpy arrays, torch tensors, or others that are convertible to tensors.

        y : Any, optional
            Target data corresponding to `x`. If provided, it is included in the forward 
            pass, which can be useful for certain types of models where the target data 
            might influence the inference outcome (e.g., teacher forcing in RNNs). If the 
            model does not utilize target data during inference, `y` can be omitted.

        **fit_params : dict, optional
            Additional parameters for fine-tuning the inference process, such as dropout 
            rates or custom layers' settings, which are passed directly to the `forward` 
            method of the module.

        Returns
        -------
        torch.Tensor or sequence of torch.Tensor
            The output(s) of the network's forward pass, representing the inference results 
            for the input data.
        """
        # Ensure input and target data are tensors on the correct device
        x_tensor = to_tensor(x, device=self.device)
        y_tensor = to_tensor(y, device=self.device) if y is not None else None

        # Merge input data and fitting parameters if input is a mapping type
        if isinstance(x, Mapping):
            x_dict = self._merge_x_and_fit_params(x, fit_params)
            return self.forward(**x_dict)

        # Call the forward method with the tensor data and additional parameters
        return self.forward(x_tensor, y_tensor, **fit_params) if y_tensor is not None else self.forward(x_tensor, **fit_params)

    def _get_predict_nonlinearity(self):
        """
        Return the nonlinearity to be applied to the prediction.

        In scenarios where the model's output needs to be post-processed before predictions,
        this method provides the necessary transformation function. for example, converting
        logits to probabilities in a classification task.

        The retrieved nonlinearity is intended to be used with the `predict` and `predict_proba`
        methods. It is not involved in loss computation during training, thus has no effect
        on the model's training process.

        If 'auto' is specified, the nonlinearity is inferred based on the model's output layer
        type and task (e.g., softmax for multi-class classification).

        Raises
        ------
        TypeError
            Raised if the returned value is not callable.

        Returns
        -------
        nonlin : callable
            A function that accepts a PyTorch tensor and returns a tensor with the nonlinearity
            applied.

        Examples
        --------
        >>> # Retrieve the nonlinearity function
        >>> nonlin = self._get_predict_nonlinearity()
        >>> # Apply the nonlinearity function to some data
        >>> transformed_data = nonlin(torch.tensor([1.0, 2.0, 3.0]))

        Notes
        -----
        - The `_identity` function is a placeholder that simply returns its input without any change.
        - When 'auto' is specified for `predict_nonlinearity`, the function `_infer_predict_nonlinearity`
        automatically determines a suitable nonlinearity based on the model architecture and the type
        of problem being addressed.
        """
        # Ensure the model has been trained before this method is called
        self.check_is_fitted()

        # Get the nonlinearity set for prediction
        nonlin = self.predict_nonlinearity

        # Use identity function if no nonlinearity has been set
        if nonlin is None:
            nonlin = _identity

        # Infer the nonlinearity based on model architecture and task if set to 'auto'
        elif nonlin == 'auto':
            nonlin = _infer_predict_nonlinearity(self)

        # Ensure that the nonlinearity is a callable function
        if not callable(nonlin):
            raise TypeError("The predict_nonlinearity attribute must be callable, 'auto', or None.")

        # Return the nonlinearity function
        return nonlin
    
    def predict_proba(self, X):
        """
        Compute the probability estimates of the given input data `X`.

        This method processes the input data `X` through the neural network using the `forward_iter`
        method to obtain the raw outputs. It then applies a nonlinearity to these outputs to
        transform them into probability estimates. The nonlinearity is specified by the
        `_get_predict_nonlinearity` method. For models that output multiple values per instance
        (e.g., multi-task learning), only the first output is used to compute probabilities.

        Parameters
        ----------
        X : compatible with stockpy.dataset.StockpyDataset
            The input data for which to predict probabilities. The data can be in various formats,
            including numpy arrays, torch tensors, pandas DataFrame or Series, scipy sparse CSR
            matrices, dictionaries containing any of these types, lists/tuples of any of these types,
            or a custom Dataset object.

        Returns
        -------
        y_proba : numpy.ndarray
            A 2D array of shape (n_samples, n_classes) with the probability estimates of the
            classes for each input sample.

        Notes
        -----
        - If the model's forward method returns multiple outputs as a tuple, it is assumed
        that the first element of the tuple contains the relevant output for probability
        estimation.
        - The nonlinearity transformation is crucial for models like logistic regression or
        networks with a softmax final layer, as raw outputs (logits) are not probabilities.
        - This method is typically used in classification tasks where the output represents
        class membership probabilities.

        """

        # Retrieve the nonlinearity function to transform the model's output to probabilities
        nonlin = self._get_predict_nonlinearity()

        # List to accumulate the probability estimates batch-wise
        y_probas = []

        # Iterate over the batches of data, forward pass, apply nonlinearity, and collect results
        for batch in self.forward_iter(X, training=False):
            # Select the first output if the network returns a tuple
            output = batch[0] if isinstance(batch, tuple) else batch
            
            # Transform the network output to probabilities
            probabilities = nonlin(output)
            
            # Convert PyTorch tensor to numpy array and store
            y_probas.append(to_numpy(probabilities))
        
        # Concatenate the batch-wise probability arrays to form the final result
        y_proba = np.concatenate(y_probas, axis=0)

        return y_proba


    def get_loss(self, y_pred, y_true, X=None, training=False):
        """
        Compute the loss value for a batch of predictions and corresponding true values.

        The function calculates the loss using the model's prediction and the actual target values
        for a batch of data. This computation is a fundamental part of the training and validation
        process as it quantifies the model's performance. The specific loss function used is 
        determined by the criterion set during model configuration (`self.criterion_`).

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted target values obtained from the model's forward pass.

        y_true : torch.Tensor
            The true target values against which to compute the loss.

        X : input data, compatible with stockpy.dataset.StockpyDataset, optional
            The input data corresponding to the target values, used for custom loss functions 
            that require input features along with targets. Default types accepted include numpy 
            arrays, torch tensors, pandas DataFrames or Series, scipy sparse CSR matrices, and 
            combinations thereof within dictionaries or lists/tuples. For other data types, provide 
            a custom `Dataset` capable of handling them.

        training : bool, optional
            Indicates whether the loss calculation is performed during training. Some loss functions 
            behave differently during training and evaluation phases (e.g., dropout or batch normalization).

        Returns
        -------
        loss : torch.Tensor
            The calculated loss tensor. If reduction is applied in the loss criterion (e.g., 'mean' or 'sum'),
            the result will be a scalar tensor; otherwise, it will match the batch size.

        """
        # Ensure that the true target values are on the same device as the predictions before loss computation
        y_true = to_tensor(y_true, device=self.device)

        # Compute the loss using the criterion defined in the model configuration
        return self.criterion_(y_pred, y_true)    

    def get_dataset(self, X, y=None):
        """
        Construct or fetch the appropriate dataset object for the input and target data.

        This method is tasked with setting up a dataset that can be processed by the
        neural network. It either returns the dataset provided in `X`, if it's already
        a dataset object, or creates a new one using the specified `self.dataset`.
        Custom dataset types based on the model being used (RNN, FFNN, CNN, Seq2Seq) 
        are supported through dynamic selection based on `self.model_type`.

        To customize dataset creation, this method can be overridden.

        Parameters
        ----------
        X : input data, compatible with stockpy.dataset.StockpyDataset
            Acceptable data formats include numpy arrays, torch tensors, pandas DataFrame 
            or Series, scipy sparse CSR matrices, and combinations of these in dictionaries 
            or lists/tuples. If `X` is an instance of `Dataset`, it is returned as is.

        y : target data, compatible with stockpy.dataset.StockpyDataset, optional
            Supports the same types as `X`. If `y` is None and `X` is a `Dataset` that 
            contains targets, `y` does not need to be provided.

        Returns
        -------
        dataset : Dataset
            An instance of the dataset containing the input and target data ready for 
            model processing.

        Raises
        ------
        TypeError
            If both an initialized dataset object and dataset arguments are passed, 
            it raises a TypeError to avoid conflicts.

        Notes
        -----
        - The method uses `self.datasets`, a dictionary that maps model types to their
        respective dataset class. It checks if a dataset object is already initialized
        and if so, returns it directly. Otherwise, it initializes a new dataset with
        the parameters obtained from `self.get_params_for('dataset')`.

        """
        if is_dataset(X):
            # Directly return if X is already a dataset instance
            return X

        # Dataset classes based on model type
        self.datasets = {
            'rnn': StockDatasetRNN,
            'ffnn': StockDatasetFFNN,
            'cnn': StockDatasetCNN,
            # 'seq2seq': StockDatasetSeq2Seq,
        }

        # Select and potentially instantiate the dataset
        dataset_cls = self.datasets[self.model_type]
        is_initialized = not callable(dataset_cls)

        # Fetch parameters meant for dataset initialization
        dataset_kwargs = self.get_params_for('dataset')

        # Avoid conflicting initializations
        if is_initialized and dataset_kwargs:
            raise TypeError(f"Cannot pass initialized Dataset with additional arguments: {dataset_kwargs}")

        # Initialize with additional parameters if required
        if not is_initialized:
            if issubclass(dataset_cls, StockDatasetRNN):
                # Initialize with sequence length for RNN and Seq2Seq models
                return dataset_cls(X, y, length=None, seq_len=self.seq_len, **dataset_kwargs)

            # Initialize for other types without sequence length
            return dataset_cls(X, y, **dataset_kwargs)

        # Return the already initialized dataset
        return dataset_cls

    def get_split_datasets(self, X, y=None, **fit_params):
        """
        Obtain the training and validation datasets for use within the net.

        This method is responsible for splitting the input data `X` and targets `y` 
        into datasets for training and validation. If no custom train/validation 
        split is provided (`self.train_split` is None), then no validation will be 
        performed. This method can be overridden to modify the way the net handles 
        data splitting.

        Parameters
        ----------
        X : input data, compatible with stockpy.dataset.StockpyDataset
            Acceptable data formats by default include:

            - numpy arrays
            - torch tensors
            - pandas DataFrame or Series
            - scipy sparse CSR matrices
            - a dictionary of the former three
            - a list/tuple of the former three
            - a Dataset

            If these do not suit your data, a custom `Dataset` that can process
            your data format should be passed.

        y : target data, compatible with stockpy.dataset.StockpyDataset, optional
            This supports the same data formats as `X`. If `X` is a `Dataset` that 
            already includes targets, `y` can be set to None.

        **fit_params : additional parameters
            These are passed to the `self.train_split` method and can be used to
            provide additional information necessary for data splitting.

        Returns
        -------
        dataset_train : Dataset
            The prepared dataset for training.

        dataset_valid : Dataset or None
            The prepared dataset for validation, if applicable. None if `self.train_split`
            is set to None, implying that internal validation should be skipped.

        Notes
        -----
        This method internally calls `self.get_dataset` to create a dataset object from 
        the input data `X` and targets `y`. It then applies `self.train_split` to 
        this dataset to create the training and validation datasets. Custom split 
        criteria can be provided during the net's initialization or by overriding 
        `self.train_split`.
        """

        # Get a dataset object from the input data and targets
        dataset = self.get_dataset(X, y)

        # If no train split method is provided, only return the dataset for training
        if not self.train_split:
            return dataset, None

        # If train split method is provided, split the dataset and return
        # If y is None and the dataset includes the targets, fit_params can still be applied
        return self.train_split(dataset, y, **fit_params)

    def get_iterator(self, dataset, training=False):
        """
        Create an iterator for the given dataset to loop over data batches.

        This method is used to instantiate an iterator that yields batches
        of data from the specified dataset. It determines the type of iterator
        to use based on the 'training' flag and sets the batch size according
        to the model's parameters or defaults to 'self.batch_size'.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset to iterate over. While the default expected type is
            'stockpy.dataset.StockpyDataset', it can be any subclass of 'torch.utils.data.Dataset'.

        training : bool, optional (default=False)
            Flag indicating whether the iterator is for training or validation/test.
            When True, 'iterator_train' parameters are used; otherwise 'iterator_valid'
            parameters are used.

        Returns
        -------
        iterator : torch.utils.data.DataLoader
            The instantiated DataLoader that can iterate over the dataset's mini-batches.
        
        Notes
        -----
        The batch size for the iterator is determined in the following order of precedence:
        1. 'iterator_train__batch_size' or 'iterator_valid__batch_size' if set,
        2. 'self.batch_size' if 'batch_size' is not included in the kwargs,
        3. The size of the dataset if 'batch_size' is set to -1 (meaning use all data at once).
        """
        # Choose the iterator and parameters based on the training flag
        kwargs = self.get_params_for('iterator_train' if training else 'iterator_valid')
        iterator = DataLoader

        # Default to 'self.batch_size' if 'batch_size' is not specified
        kwargs.setdefault('batch_size', self.batch_size)

        # If 'batch_size' is set to -1, use the entire dataset as a single batch
        if kwargs['batch_size'] == -1:
            kwargs['batch_size'] = len(dataset)

        return iterator(dataset, **kwargs)
    
    def save_params(
            self,
            f_params=None,
            f_optimizer=None,
            f_history=None,
            use_safetensors=False,
            **kwargs):
        """
        Save parameters, optimizer state, and history to files.

        This method allows for saving the model parameters, optimizer state,
        and training history to disk, allowing for the resumption of training
        at a later time or model deployment.

        Parameters
        ----------
        f_params : str or file-like object, optional
            Filename or file-like object where the model parameters will be saved.
            If not specified, the model parameters will not be saved.

        f_optimizer : str or file-like object, optional
            Filename or file-like object where the optimizer state will be saved.
            If not specified, the optimizer state will not be saved.

        f_history : str or file-like object, optional
            Filename or file-like object where the training history will be saved.
            If not specified, the training history will not be saved.

        use_safetensors : bool, optional (default=False)
            If set to True, the model parameters will be saved in the safetensors
            format which ensures more robust and safer serialization.

        **kwargs : dict
            Additional keyword arguments that will be passed to the underlying
            save functions. These arguments are dynamically determined based
            on the other inputs.

        Raises
        ------
        ValueError
            If `use_safetensors` is True and a ValueError occurs, possibly due
            to trying to save non-tensor objects with safetensors.

        Notes
        -----
        If the model, optimizer, or history has not been initialized, an error
        message will be printed, indicating that the user should initialize
        or fit the model before attempting to save it.

        """

        # Internal function to save the state dictionary using safetensors or PyTorch's save
        if use_safetensors:
            def _save_state_dict(state_dict, f_name):
                from safetensors.torch import save_file, save
                try:
                    if isinstance(f_name, (str, os.PathLike)):
                        save_file(state_dict, f_name)
                    else:  # file
                        as_bytes = save(state_dict)
                        f_name.write(as_bytes)
                except ValueError as exc:
                    msg = (
                        f"You are trying to store {f_name} using safetensors "
                        "but there was an error. Safetensors can only store "
                        "tensors, not generic Python objects (as e.g. optimizer "
                        "states). If you want to store generic Python objects, "
                        "don't use safetensors."
                    )
                    raise ValueError(msg) from exc
        else:
            def _save_state_dict(state_dict, f_name):
                torch.save(state_dict, f_name)

        kwargs_module, kwargs_other = _check_f_arguments(
            'save_params',
            f_params=f_params,
            f_optimizer=f_optimizer,
            f_history=f_history,
            **kwargs)

        if not kwargs_module and not kwargs_other:
            if self.verbose:
                print("Nothing to save")
            return

        msg_init = (
            "Cannot save state of an un-initialized model. "
            "Please initialize first by calling .initialize() "
            "or by fitting the model with .fit(...).")
        msg_module = (
            "You are trying to save 'f_{name}' but for that to work, the net "
            "needs to have an attribute called 'net.{name}_' that is a PyTorch "
            "Module or Optimizer; make sure that it exists and check for typos.")

        for attr, f_name in kwargs_module.items():
            # valid attrs can be 'module_', 'optimizer_', etc.
            if attr.endswith('_') and not self.initialized_:
                self.check_is_fitted([attr], msg=msg_init)

            _save_state_dict(self.state_dict(), f_name)

        # only valid key in kwargs_other is f_history
        f_history = kwargs_other.get('f_history')
        if f_history is not None:
            self.history.to_file(f_history)

    def load_params(
            self,
            f_params=None,
            f_optimizer=None,
            f_history=None,
            checkpoint=None,
            use_safetensors=False,
            **kwargs):
        """
        Load parameters, optimizer state, and history from files.

        This function is used to load the model parameters, optimizer state,
        and training history. This can be used for continuing training or model
        deployment.

        Parameters
        ----------
        f_params : str or file-like object, optional
            The filename or file-like object from which the model parameters should be loaded.
            If not specified, the model parameters will not be loaded.

        f_optimizer : str or file-like object, optional
            The filename or file-like object from which the optimizer state should be loaded.
            If not specified, the optimizer state will not be loaded.

        f_history : str or file-like object, optional
            The filename or file-like object from which the training history should be loaded.
            If not specified, the training history will not be loaded.

        checkpoint : Checkpoint, optional
            A `Checkpoint` instance that contains the paths to the files which store
            the model parameters, optimizer state, and training history. If specified,
            it will override the individual file paths provided through the other parameters.

        use_safetensors : bool, optional (default=False)
            If set to True, use the `safetensors` format to load the model parameters.
            This format ensures more robust and safer serialization.

        **kwargs : dict
            Additional keyword arguments that will be used to update the file paths for
            loading the components if they are provided.

        Raises
        ------
        FileNotFoundError
            If a file path is specified and the file does not exist.

        Notes
        -----
        If a `Checkpoint` instance is passed, and the network is not yet initialized, it
        will initialize the network before loading the checkpoint. Also, `f_history` will
        be loaded from the checkpoint if it's not explicitly provided but is present in the
        checkpoint.
        """

        if use_safetensors:
            def _get_state_dict(f_name):
                from safetensors import safe_open
                from safetensors.torch import load

                if isinstance(f_name, (str, os.PathLike)):
                    state_dict = {}
                    with safe_open(f_name, framework='pt', device=self.device) as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                else:
                    # file
                    as_bytes = f_name.read()
                    state_dict = load(as_bytes)

                return state_dict
        else:
            def _get_state_dict(f_name):
                map_location = get_map_location(self.device)
                self.device = self._check_device(self.device, map_location)
                return torch.load(f_name, map_location=map_location)

        kwargs_full = {}
        if checkpoint is not None:
            if not self.initialized_:
                self.initialize()
            if f_history is None and checkpoint.f_history is not None:
                self.history = History.from_file(checkpoint.f_history_)
            kwargs_full.update(**checkpoint.get_formatted_files(self))

        # explicit arguments may override checkpoint arguments
        kwargs_full.update(**kwargs)
        for key, val in [('f_params', f_params), ('f_optimizer', f_optimizer),
                         ('f_history', f_history)]:
            if val:
                kwargs_full[key] = val

        kwargs_module, kwargs_other = _check_f_arguments('load_params', **kwargs_full)

        if not kwargs_module and not kwargs_other:
            if self.verbose:
                print("Nothing to load")
            return

        # only valid key in kwargs_other is f_history
        f_history = kwargs_other.get('f_history')
        if f_history is not None:
            self.history = History.from_file(f_history)

    def _get_params_for(self, prefix):
        """
        Retrieve parameters for a specific prefix from the instance's dictionary.

        This private method is used to gather all parameters that belong to a
        certain scope, defined by the prefix. For instance, it can extract all
        parameters related to the 'train' process if the prefix 'train' is passed.
        This is particularly useful for setting up different configurations for
        various stages in the training and evaluation of a neural network.

        Parameters
        ----------
        prefix : str
            The prefix indicating which parameters to retrieve. for example, 'train'
            would retrieve all parameters that control training configuration.

        Returns
        -------
        dict
            A dictionary with keys that start with the given prefix, containing the
            related parameters. The prefix is stripped from the keys in the returned
            dictionary.

        Raises
        ------
        AttributeError
            If the prefix is not found in the instance's attribute dictionary.

        """
        return params_for(prefix, self.__dict__)

    def get_params_for(self, prefix):
        """
        Retrieve and return initialization parameters for a specific attribute.

        This method is intended for internal use within the class to collect the
        initialization parameters for class attributes, such as PyTorch modules,
        optimizers, or data loaders. These parameters are typically used to
        reinitialize components during the setting of parameters or during class
        initialization. As a user, you might use this function directly if you
        are customizing or extending the class functionality.

        Parameters
        ----------
        prefix : str
            The name of the attribute for which to retrieve the initialization
            parameters. For instance, if the prefix is 'module', the method will
            return the parameters that were used to initialize the model's module.

        Returns
        -------
        kwargs : dict
            A dictionary containing the keyword arguments that can be used to
            initialize or update the attribute identified by the prefix.

        Examples
        --------
        Suppose `self` is an instance of a class that has a module attribute initialized with
        certain parameters, and the class defines this method to retrieve them:

        .. code:: python

            module_params = self.get_params_for('module')
            # Now `module_params` contains the initialization parameters for the module.

        """
        # Call the internal method to get the parameters for the given prefix
        return self._get_params_for(prefix)

    def _get_params_for_optimizer(self, prefix, named_parameters):
        """
        Retrieve and return initialization parameters for a specific optimizer.

        This internal method handles the extraction of optimizer parameters
        including any user-defined parameter groups. Additionally, it includes
        the learning rate ('lr') as a default parameter unless overridden by user
        specifications. It prepares both arguments and keyword arguments required
        for the instantiation of a PyTorch optimizer.

        Parameters
        ----------
        prefix : str
            The prefix indicating the specific optimizer to configure, e.g., 'optimizer_adam'
            for an Adam optimizer.

        named_parameters : iterable of (str, torch.nn.Parameter)
            An iterable of named parameters typically retrieved from a PyTorch module using
            the `named_parameters()` method. This iterable contains tuples with the parameter
            name and the parameter itself.

        Returns
        -------
        args : tuple
            A single-element tuple containing a list of parameter groups. Each parameter group
            is a dictionary specifying parameters and their corresponding optimization options.

        kwargs : dict
            A dictionary of keyword arguments for the optimizer initialization. This typically
            includes the learning rate ('lr') and potentially other hyperparameters like weight
            decay.

        """
        # Fetch optimizer parameters using a provided method to extract parameters
        kwargs = self.get_params_for(prefix)
        
        # Initialize list to hold parameter groups
        pgroups = []

        # Convert the named parameters to a list for easier processing
        params = list(named_parameters)

        # Extract and set up parameter groups if specified by the user
        for pattern, group in kwargs.pop('param_groups', []):
            # Filter parameters matching the pattern
            matches = [i for i, (name, _) in enumerate(params) if fnmatch.fnmatch(name, pattern)]
            # Create a parameter group for the matched parameters
            if matches:
                pgroup_params = [params.pop(i)[1] for i in reversed(matches)]
                pgroups.append({'params': pgroup_params, **group})

        # Remaining parameters are grouped separately
        if params:
            pgroups.append({'params': [param for _, param in params]})

        # Tuple of argument is a list of parameter groups
        args = (pgroups,)
        
        # Default learning rate added to kwargs if not provided
        if 'lr' not in kwargs:
            kwargs['lr'] = self.lr

        return args, kwargs

    def get_params_for_optimizer(self, prefix, named_parameters):
        """
        Retrieve initialization parameters for a specified optimizer.

        This public method simplifies the process of collecting initialization
        parameters for optimizers, especially when dealing with complex configurations,
        such as varying learning rates for different layers or parameters within the model.
        It is designed to be called within your custom `initialize_optimizer` method
        to properly set up the optimizer with its necessary arguments and keyword arguments.

        Parameters
        ----------
        prefix : str
            A string that identifies the optimizer configuration within the internal
            parameter storage. for example, 'optimizer' could be a key for the default
            optimizer settings.

        named_parameters : iterator of (str, torch.nn.Parameter)
            An iterator yielding named parameter tuples as obtained from a model's
            `named_parameters()` method. These are the parameters that the optimizer will
            be responsible for updating during the training process.

        Returns
        -------
        args : tuple
            A tuple of positional arguments needed to initialize the optimizer. This typically
            includes a list of parameter groups, each possibly with its own optimization settings.

        kwargs : dict
            A dictionary of keyword arguments for initializing the optimizer, which might include
            the learning rate ('lr') and other hyperparameters such as weight decay.

        Notes
        -----
        This method acts as a convenient wrapper around the internal
        `_get_params_for_optimizer` method, streamlining the optimizer initialization process.
        Direct calls to this method are usually not necessary unless custom initialization
        logic is being implemented.

        """
        # Delegate the task to the internal method to fetch the parameters
        args, kwargs = self._get_params_for_optimizer(prefix, named_parameters)
        
        # Return the parameters ready for optimizer initialization
        return args, kwargs

    def _get_param_names(self):
        """
        Retrieve names of hyperparameters.

        This function retrieves the names of all hyperparameters belonging to the object, 
        excluding those that end with an underscore ('_'). Conventionally, hyperparameters
        ending with an underscore represent parameters that are derived during fitting.

        Returns
        -------
        param_names : list of str
            A list containing the names of all hyperparameters for the model.

        Notes
        -----
        The function excludes any attribute that ends with an underscore ('_'). In machine
        learning libraries like scikit-learn, these attributes are typically set only after
        the model has been fitted and are not initial hyperparameters set by the user.

        """

        # Using list comprehension to filter out attributes
        # that are designated for fitted parameters (ending with '_')
        param_names = [key for key in self.__dict__.keys() if not key.endswith('_')]

        return param_names
    
    def _get_params_callbacks(self, deep=True):
        """
        Extract parameters for callback attributes.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            A dictionary of parameter names mapped to their values. For callbacks,
            the parameter names are prefixed with 'callbacks__', and for parameters
            within each callback, the names are further suffixed with the name of the
            parameter within the callback.

        Notes
        -----
        This function is needed because the default implementation of `get_params()`
        in scikit-learn does not handle attributes that are lists of objects, which
        is typically the case with callbacks.

        """
        # Initialize an empty dictionary to hold parameter names and values
        params = {}
        
        # If not doing a deep retrieval, return the empty params
        if not deep:
            return params

        # Access the 'callbacks_' attribute if it exists, otherwise use an empty list
        callbacks_ = getattr(self, 'callbacks_', [])
        
        # Iterate over the callbacks, if any
        for key, val in chain(callbacks_, self._default_callbacks):
            # Construct the parameter name with a 'callbacks__' prefix
            name = 'callbacks__' + key
            
            # Store the callback object itself
            params[name] = val
            
            # If the callback is deactivated (None), skip it
            if val is None:
                continue
            
            # Retrieve and store parameters for each callback using its `get_params` method
            for subkey, subval in val.get_params().items():
                # Construct the full parameter name with both prefixes
                subname = name + '__' + subkey
                
                # Store each sub-parameter
                params[subname] = subval
        
        # Return the dictionary of parameters
        return params

    def get_params(self, deep=True, **kwargs):
        """
        Get parameters for this estimator.

        This method will obtain the parameters for the estimator and include
        parameters for the callbacks, which require special handling. It extends the
        functionality of `get_params` from scikit-learn's `BaseEstimator`.

        Parameters inherited from scikit-learn's `BaseEstimator` are retrieved, and then
        the parameters specific to the callbacks are added. Certain parameters are
        explicitly excluded from the result.

        Parameters
        ----------
        deep : bool, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        **kwargs
            Additional parameters that will be passed to the `get_params` method of
            the superclass.

        Returns
        -------
        params : dict
            A dict of parameter names mapped to their values, including those for
            callbacks, after excluding specific parameters that are not part of the
            public API.

        Notes
        -----
        The `get_params` method is crucial for model inspection and for cloning
        estimators within scikit-learn pipelines and grid searches. Parameters that are
        deemed to be "internal" (like fitted parameters or temporary objects that are
        not user-specified parameters) are excluded.

        """
        # First, get the parameters as returned by the sklearn's base estimator
        params = SkBaseEstimator.get_params(self, deep=deep, **kwargs)
        
        # Get the callback parameters which require special treatment
        params_cb = self._get_params_callbacks(deep=deep)
        
        # Update the parameters dictionary with callback parameters
        params.update(params_cb)

        # Define attributes that should not be included in the returned parameters
        to_exclude = {'_modules', '_criteria', '_optimizers'}

        # Return the parameters excluding the ones specified in to_exclude
        return {key: val for key, val in params.items() if key not in to_exclude}

    def _validate_params(self):
        """
        Validates hyperparameters passed during the initialization of the class instance.

        This internal utility function is used to ensure that the parameters passed to the
        class constructor are valid. It checks whether the provided hyperparameters match
        the expected ones and raises a ValueError if an unexpected or incorrectly named
        argument is found.

        It goes through each initialization parameter and determines whether it is a known
        parameter based on the established prefixes. If the parameter is not recognized or
        is missing an underscore, the function will generate an error message.

        Raises
        ------
        ValueError
            If any provided initialization arguments are not recognized (i.e., unexpected)
            or if they are named incorrectly (missing a double underscore after the prefix),
            a ValueError is raised with a message detailing the issues.

        Notes
        -----
        This method should not be confused with the `_validate_params` method from scikit-learn's
        `BaseEstimator`. While it serves a similar purpose, this method is specifically tailored
        to the class it resides in and handles the validation without relying on scikit-learn's
        parameter validation machinery.
        """
        # List for storing names of unexpected kwargs
        unexpected_kwargs = []

        # List for storing names of kwargs where double underscores are missing
        missing_dunder_kwargs = []

        # Iterate over each key in the provided parameters for validation
        for key in sorted(self._params_to_validate):

            # Skip attributes that are meant to be set by the class (ending with '_')
            if key.endswith('_'):
                continue

            # Check if the key matches any of the expected prefixes
            for prefix in sorted(self.prefixes_, key=lambda s: (-len(s), s)):
                if key == prefix:
                    break
                if key.startswith(prefix) and not key.startswith(prefix + '__'):
                    # If the key is missing the '__', it's likely a typo
                    missing_dunder_kwargs.append((prefix, key))
                    break
            else:
                # Key didn't match any prefix, it's unexpected
                unexpected_kwargs.append(key)

        # Collect error messages
        msgs = []

        # Generate messages for unexpected kwargs
        if unexpected_kwargs:
            tmpl = ("__init__() got unexpected argument(s) {}. "
                    "Either you made a typo, or you added new arguments "
                    "in a subclass; if that is the case, the subclass "
                    "should deal with the new arguments explicitly.")
            msgs.append(tmpl.format(', '.join(sorted(unexpected_kwargs))))

        # Generate messages for kwargs with missing double underscores
        for prefix, key in sorted(missing_dunder_kwargs, key=lambda tup: tup[1]):
            tmpl = "Got an unexpected argument {}, did you mean {}?"
            suggestion = prefix + '__' + key[len(prefix):].lstrip('_')
            msgs.append(tmpl.format(key, suggestion))

        # Additional checks for specific parameters can be included here
        # for example:
        valid_vals_use_caching = ('auto', False, True)
        if self.use_caching not in valid_vals_use_caching:
            msgs.append(
                f"Incorrect value for 'use_caching' parameter ('{self.use_caching}'), "
                f"expected one of: {', '.join(map(str, valid_vals_use_caching))}.")

        # Raise ValueError if there are any messages
        if msgs:
            raise ValueError('\n'.join(msgs))

    def _check_deprecated_params(self, **kwargs):
        """
        Placeholder for checking deprecated parameters.
        Currently does nothing.
        """
        pass
    
    def _check_n_features(self, X, reset):
        """
        Verify the number of features in X matches the expected number.

        This is an internal method that updates or compares the `n_features_in_` attribute
        based on the actual number of features in the input sample `X`. It is typically
        used to ensure that the number of features for `X` during `fit` matches that of
        data passed to other methods like `predict` or `transform`.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples whose number of features are to be checked.

        reset : bool
            Determines how to use the `n_features_in_` attribute. If True, this method
            sets the `n_features_in_` attribute to the number of features in `X`.
            If False, this method checks that `n_features_in_` matches the number
            of features in `X`.

        Raises
        ------
        ValueError
            If `reset` is False and the number of features in `X` does not match the
            expected number of features (`n_features_in_`), this method raises a
            ValueError with a message indicating the discrepancy.

        Notes
        -----
        This method is typically called from `fit` and `partial_fit` with `reset=True`
        to establish the expected number of features, and from other methods like
        `predict`, `transform`, or `score` with `reset=False` to enforce that subsequent
        input data has the same number of features.
        """
        try:
            n_features = _num_features(X)
        except TypeError as e:
            if not reset and hasattr(self, "n_features_in_"):
                raise ValueError(
                    "X does not contain any features, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{self.n_features_in_} features"
                ) from e
            # If the number of features is not defined and reset=True,
            # then we skip this check
            return

        if reset:
            self.n_features_in_ = n_features
            return

        if not hasattr(self, "n_features_in_"):
            # Skip this check if the expected number of expected input features
            # was not recorded by calling fit first. This is typically the case
            # for stateless transformers.
            return

        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )
        
    def _check_feature_names(self, X, *, reset):
        """
        Validate or update the feature names recorded by the estimator.

        This method is used to set or verify the `feature_names_in_` attribute of the estimator
        based on the feature names present in the input `X`. It's typically used within `fit`
        to set feature names and within other methods to ensure consistency of feature names
        in subsequent calls after the model has been fitted.

        Parameters
        ----------
        X : {ndarray, dataframe} of shape (n_samples, n_features)
            The input samples. Feature names are inferred from `dataframe` and are ignored
            if `X` is a NumPy ndarray without feature names.

        reset : bool
            If True, the method will set the `feature_names_in_` attribute based on the input
            `X`. If False, the method checks the input for consistency with the feature names
            seen when `reset` was last True.

        Raises
        ------
        ValueError
            If the feature names in `X` differ from those seen during the last reset,
            this method raises a ValueError with a message indicating which feature names
            are inconsistent.

        Warns
        -----
        UserWarning
            If `X` has feature names and the estimator was fitted without feature names or
            vice versa, a warning is raised to notify the potential inconsistency.

        Notes
        -----
        This method should be called with `reset=True` during `fit` and during the first call
        to `partial_fit`. All other methods that validate `X` should call this with
        `reset=False` to ensure that the feature names of `X` match those seen during fitting.
        """

        if reset:
            feature_names_in = _get_feature_names(X)
            if feature_names_in is not None:
                self.feature_names_in_ = feature_names_in
            elif hasattr(self, "feature_names_in_"):
                # Delete the attribute when the estimator is fitted on a new dataset
                # that has no feature names.
                delattr(self, "feature_names_in_")
            return

        fitted_feature_names = getattr(self, "feature_names_in_", None)
        X_feature_names = _get_feature_names(X)

        if fitted_feature_names is None and X_feature_names is None:
            # no feature names seen in fit and in X
            return

        if X_feature_names is not None and fitted_feature_names is None:
            warnings.warn(
                f"X has feature names, but {self.__class__.__name__} was fitted without"
                " feature names"
            )
            return

        if X_feature_names is None and fitted_feature_names is not None:
            warnings.warn(
                "X does not have valid feature names, but"
                f" {self.__class__.__name__} was fitted with feature names"
            )
            return

        # validate the feature names against the `feature_names_in_` attribute
        if len(fitted_feature_names) != len(X_feature_names) or np.any(
            fitted_feature_names != X_feature_names
        ):
            message = (
                "The feature names should match those that were passed during fit.\n"
            )
            fitted_feature_names_set = set(fitted_feature_names)
            X_feature_names_set = set(X_feature_names)

            unexpected_names = sorted(X_feature_names_set - fitted_feature_names_set)
            missing_names = sorted(fitted_feature_names_set - X_feature_names_set)

            def add_names(names):
                output = ""
                max_n_names = 5
                for i, name in enumerate(names):
                    if i >= max_n_names:
                        output += "- ...\n"
                        break
                    output += f"- {name}\n"
                return output

            if unexpected_names:
                message += "Feature names unseen at fit time:\n"
                message += add_names(unexpected_names)

            if missing_names:
                message += "Feature names seen at fit time, yet now missing:\n"
                message += add_names(missing_names)

            if not missing_names and not unexpected_names:
                message += (
                    "Feature names must be in the same order as they were in fit.\n"
                )

            raise ValueError(message)
        
    def _set_params_callback(self, **params):
        """
        Set parameters for callbacks.

        Parameters
        ----------
        **params : dict
            Arbitrary keyword arguments. Each key-value pair in `params` corresponds
            to a parameter name and its value.

        Returns
        -------
        self : object
            The instance itself.

        Raises
        ------
        ValueError
            If an attempt is made to set a parameter on a callback that does not exist.

        Notes
        -----
        The method is modeled after `sklearn.utils._BaseComposition._set_params`. It is intended
        for internal use by the `set_params` method of the estimator, not for direct user call.

        The 'callbacks' attribute is set directly if provided in `params`. The method then replaces
        any existing callbacks with the specified ones. If a key in `params` is prefixed with
        'callbacks__', it designates the parameter to be set on a specific callback. The method
        updates the parameters of these specific callbacks.

        """
        # model after sklearn.utils._BaseCompostion._set_params
        # 1. All steps
        if 'callbacks' in params:
            setattr(self, 'callbacks', params.pop('callbacks'))

        # 2. Step replacement
        names, _ = zip(*getattr(self, 'callbacks_'))
        for key in params.copy():
            name = key[11:]  # drop 'callbacks__'
            if '__' not in name and name in names:
                self._replace_callback(name, params.pop(key))

        # 3. Step parameters and other initilisation arguments
        for key in params.copy():
            name = key[11:]
            part0, part1 = name.split('__')
            kwarg = {part1: params.pop(key)}
            callback = dict(self.callbacks_).get(part0)
            if callback is not None:
                callback.set_params(**kwarg)
            else:
                raise ValueError(
                    "Trying to set a parameter for callback {} "
                    "which does not exist.".format(part0))

        return self

    def _replace_callback(self, name, new_val):
        """
        Replace a callback with a new value in the callbacks list.

        Parameters
        ----------
        name : str
            The name of the callback to replace.
        new_val : object
            The new callback object to replace the old one.

        Raises
        ------
        ValueError
            If no existing callback matches the provided name.

        Notes
        -----
        This function assumes the presence of an attribute `callbacks_`, which is a list
        of tuples. Each tuple contains the name of the callback as its first element.
        The method replaces the callback with the specified `name` with `new_val`.
        
        The method is intended for internal use, performing in-place modification of
        the `callbacks_` list and does not return any value. The caller is responsible
        for ensuring the validity of the `name`.
        
        """

        # assumes `name` is a valid callback name
        callbacks_new = self.callbacks_[:]
        for i, (cb_name, _) in enumerate(callbacks_new):
            if cb_name == name:
                callbacks_new[i] = (name, new_val)
                break
        setattr(self, 'callbacks_', callbacks_new)
        
    def _validate_data(
        self,
        X="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        cast_to_ndarray=True,
        **check_params,
    ):
        """
        Validate input data and manage `n_features_in_` attribute.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape (n_samples, n_features), default="no_validation"
            The input samples. No validation if 'no_validation'.
        y : array-like of shape (n_samples,), default="no_validation"
            The target values. No validation if 'no_validation'.
        reset : bool, default=True
            Reset `n_features_in_` based on `X`.
        validate_separately : bool or tuple of dicts, default=False
            Validate `X` and `y` separately if True, else together.
        cast_to_ndarray : bool, default=True
            Cast `X` and `y` to ndarray after validation.
        **check_params : kwargs
            Additional kwargs for validation functions.

        Returns
        -------
        out : ndarray, sparse matrix, or tuple
            The validated input data; tuple of `(X, y)` if both are validated.

        Raises
        ------
        ValueError
            If both `X` and `y` are 'no_validation', or either is None when `requires_y` is True.

        Notes
        -----
        Intended for internal use by estimators for input validation before fit or predict.
        Not a public method.

        See Also
        --------
        check_array : Validate an array, list, sparse matrix or similar.
        check_X_y : Validate `X` and `y`.
        
        """
        self._check_feature_names(X, reset=reset)

        if y is None and self._get_tags()["requires_y"]:
            raise ValueError(
                f"This {self.__class__.__name__} estimator "
                "requires y to be passed, but the target y is None."
            )

        no_val_X = isinstance(X, str) and X == "no_validation"
        no_val_y = y is None or isinstance(y, str) and y == "no_validation"

        if no_val_X and no_val_y:
            raise ValueError("Validation should be done on X, y or both.")

        default_check_params = {"estimator": self}
        check_params = {**default_check_params, **check_params}

        if not cast_to_ndarray:
            if not no_val_X and no_val_y:
                out = X
            elif no_val_X and not no_val_y:
                out = y
            else:
                out = X, y
        elif not no_val_X and no_val_y:
            out = check_array(X, input_name="X", **check_params)
        elif no_val_X and not no_val_y:
            out = _check_y(y, **check_params)
        else:
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling check_array()
                # on X and y isn't equivalent to just calling check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                if "estimator" not in check_X_params:
                    check_X_params = {**default_check_params, **check_X_params}
                X = check_array(X, input_name="X", **check_X_params)
                if "estimator" not in check_y_params:
                    check_y_params = {**default_check_params, **check_y_params}
                y = check_array(y, input_name="y", **check_y_params)
            else:
                X, y = check_X_y(X, y, **check_params)
            out = X, y

        if not no_val_X and check_params.get("ensure_2d", True):
            self._check_n_features(X, reset=reset)

        return out
    
    def __getstate__(self):
        """
        Retrieve the object's state for serialization, handling special attributes.

        This method is called by `pickle` when serializing an object. If the object
        contains attributes that are not serializable (e.g., CUDA tensors), it
        handles their serialization separately.

        Returns
        -------
        state : dict
            The object's state as a dictionary. Non-picklable attributes
            are stored as serialized byte streams.

        Notes
        -----
        Should be complemented with `__setstate__` for deserialization.

        CUDA-dependent attributes are serialized using `torch.save`.
        This method must recognize these attributes by a specific list or prefix.

        Examples
        --------
        >>> serialized_obj = pickle.dumps(obj)  # obj uses this __getstate__

        See Also
        --------
        __setstate__ : Method to deserialize the object's state.
        pickle.dump : Serialize the object to a file.
        pickle.dumps : Serialize the object to a byte stream.
        
        """

        # Make a copy of the current state
        state = self.__dict__.copy()

        # Initialize an empty dictionary to hold CUDA-dependent attributes
        cuda_attrs = {}

        # Search for CUDA-dependent attributes and add them to cuda_attrs
        for prefix in self.cuda_dependent_attributes_:
            for key in state:
                if isinstance(key, str) and key.startswith(prefix):
                    cuda_attrs[key] = state[key]

        # Remove CUDA-dependent attributes from the main state
        for k in cuda_attrs:
            state.pop(k)

        # Serialize the CUDA-dependent attributes using PyTorch's serialization mechanism
        # and store it in a temporary SpooledTemporaryFile
        with tempfile.SpooledTemporaryFile() as f:
            torch.save(cuda_attrs, f)
            f.seek(0)
            
            # Add the serialized CUDA-dependent attributes back to the state
            state['__cuda_dependent_attributes__'] = f.read()

        return state

    def __setstate__(self, state):
        """
        Restore the object's state from a pickled representation, with handling for special attributes.

        This method is invoked by `pickle` when deserializing an object, to restore its internal
        state. It is specifically designed to handle the restoration of CUDA-dependent attributes
        which have been serialized separately.

        Parameters
        ----------
        state : dict
            The state dictionary into which the object's state was serialized. This should
            include the `__cuda_dependent_attributes__` key for any serialized CUDA-dependent
            attributes.

        Notes
        -----
        Counterpart to `__getstate__`.
        Assumes that 'device' in the state dictionary specifies the device for CUDA-dependent attributes.

        Examples
        --------
        >>> obj = pickle.loads(serialized_obj)  # obj has this __setstate__ method

        See Also
        --------
        __getstate__ : For serialization.
        pickle.load : To deserialize an object from a file.
        pickle.loads : To deserialize an object from a byte stream.

        """

        # Get the appropriate device map location, which is useful
        # when restoring on a machine where CUDA is not available.
        map_location = get_map_location(state['device'])
        
        # Keyword arguments for PyTorch's loading mechanism
        load_kwargs = {'map_location': map_location}

        # Check and update the device information in the state
        state['device'] = self._check_device(state['device'], map_location)

        # Load CUDA-dependent attributes from the temporary SpooledTemporaryFile
        with tempfile.SpooledTemporaryFile() as f:
            f.write(state['__cuda_dependent_attributes__'])
            f.seek(0)
            cuda_attrs = torch.load(f, **load_kwargs)

        # Update the state with the loaded CUDA-dependent attributes
        state.update(cuda_attrs)

        # Remove the temporary CUDA-dependent attributes key from the state
        state.pop('__cuda_dependent_attributes__')

        # Finally, update the object's __dict__ to restore its state
        self.__dict__.update(state)

    def _register_attribute(
            self,
            name,
            attr,
            prefixes=True,
            cuda_dependent_attributes=True,
    ):
        """
        Register an attribute for special handling.

        This internal method is used to maintain a record of certain types of attributes,
        particularly those that require special handling such as CUDA-dependent attributes
        or those acknowledged by `set_params`.

        Parameters
        ----------
        name : str
            The name of the attribute to be registered, with trailing underscores removed.
        attr : object
            The attribute object to register, which can be any type but often is a module
            or optimizer in PyTorch contexts.
        prefixes : bool, optional
            If True, the attribute's base name is added to a list indicating it should be
            recognized by `set_params`. Defaults to True.
        cuda_dependent_attributes : bool, optional
            If True, the attribute's name (with an underscore) is added to a list for
            CUDA-dependent attributes. Defaults to True.

        Notes
        -----
        The `prefixes_` list is utilized by `set_params` to identify valid attributes
        for setting, usually reflecting hyperparameters.
        The `cuda_dependent_attributes_` list is essential for attributes that require
        special handling when the object is transferred between CPU and GPU.

        Examples
        --------
        >>> self._register_attribute('model', my_model)

        See Also
        --------
        set_params : Uses `prefixes_` to determine attributes that can be set.
        """

        # Remove trailing underscores if any, e.g., "module_" becomes "module"
        name = name.rstrip('_')

        # Create a copy of prefixes_ list to avoid mutating the original list
        if prefixes:
            self.prefixes_ = self.prefixes_[:] + [name]

        # Create a copy of cuda_dependent_attributes_ list to avoid mutating the original list
        if cuda_dependent_attributes:
            self.cuda_dependent_attributes_ = (
                self.cuda_dependent_attributes_[:] + [name + '_'])

    def _unregister_attribute(
            self,
            name,
            prefixes=True,
            cuda_dependent_attributes=True,
    ):
        """
        Remove an attribute from the object's tracking lists.

        This method is used to remove an attribute from the tracking lists that have been
        established for managing certain attributes within the library, such as those for
        `set_params` handling and CUDA-dependent attributes.

        Parameters
        ----------
        name : str
            The name of the attribute to unregister, with trailing underscores stripped for
            the purpose of this operation.
        prefixes : bool, optional
            If True, the attribute's base name is removed from the `prefixes_` list. Defaults to True.
        cuda_dependent_attributes : bool, optional
            If True, the attribute's full name (with an underscore) is removed from the
            `cuda_dependent_attributes_` list. Defaults to True.

        Notes
        -----
        If an attribute is not found within a list, the list is left unchanged. New lists are
        created without the unwanted attribute to avoid mutating the original lists.
        The removal of attributes might be necessary during cleanup or reconfiguration procedures,
        or before object serialization.

        Examples
        --------
        >>> self._unregister_attribute('model')

        See Also
        --------
        _register_attribute : The method used for registering attributes.
        """

        # Remove trailing underscores if any, e.g., "module_" becomes "module"
        name = name.rstrip('_')

        # Create a copy of the prefixes_ list to avoid mutating the original list,
        # then remove the attribute name if present.
        if prefixes:
            self.prefixes_ = [p for p in self.prefixes_ if p != name]

        # Create a copy of the cuda_dependent_attributes_ list to avoid mutating the original,
        # then remove the attribute name if present.
        if cuda_dependent_attributes:
            self.cuda_dependent_attributes_ = [
                a for a in self.cuda_dependent_attributes_ if a != name + '_']

    def _check_settable_attr(self, name, attr):
        """
        Check if the attribute is settable in the current context.

        Ensures that the attribute being set is done so within an appropriate context,
        specifically when it's a `torch.nn.Module` or `torch.optim.Optimizer`. It must be
        set during initialization, and the name must end with an underscore.

        Parameters
        ----------
        name : str
            The name of the attribute to be checked.
        attr : object
            The attribute to check, expected to be a PyTorch module or optimizer.

        Raises
        ------
        StockpyAttributeError
            If the attribute is a `torch.nn.Module` or `torch.optim.Optimizer` and is not
            set within an initialization context, or if the name does not end with an underscore.

        Notes
        -----
        This function uses an internal flag `init_context_` to verify the correct setting context.
        This flag should be properly set prior to invoking this method.
        """
        # Check if attribute is a PyTorch Module and if it is set outside of an initialization context
        if (self.init_context_ is None) and isinstance(attr, torch.nn.Module):
            msg = ("Trying to set torch component '{}' outside of an initialize method."
                  " Consider defining it inside 'initialize_module'".format(name))
            raise StockpyAttributeError(msg)

        # Check if attribute is a PyTorch Optimizer and if it is set outside of an initialization context
        if (self.init_context_ is None) and isinstance(attr, torch.optim.Optimizer):
            msg = ("Trying to set torch component '{}' outside of an initialize method."
                  " Consider defining it inside 'initialize_optimizer'".format(name))
            raise StockpyAttributeError(msg)

        # Check if the attribute name ends with an underscore
        if not name.endswith('_'):
            msg = ("Names of initialized modules or optimizers should end "
                  "with an underscore (e.g. '{}_')".format(name))
            raise StockpyAttributeError(msg)

    def __setattr__(self, name, attr):
        """
        Set an attribute on the instance with custom handling for certain types.

        This method is overridden to provide custom behavior for setting attributes
        on neural network class instances. It handles the registration of PyTorch modules
        and optimizers into internal tracking lists and manages custom logic for certain
        types of attributes.

        Parameters
        ----------
        name : str
            The name of the attribute to set.
        value : Any
            The value to assign to the attribute.

        Notes
        -----
        - Regular attribute assignment is performed for known attributes or those already
        in the `prefixes_` list.
        - Attributes with names prefixed by double underscores '__' are considered private and
        are set normally without registration.
        - If the object is within its initial initialization phase (`__init__`), attributes
        are set normally.
        - Non-PyTorch attributes are assigned without any additional handling.

        The actual registration of attributes for tracking and the validation of settable
        attributes are deferred to the `_register_attribute()` and `_check_settable_attr()`
        methods, respectively.
        """
        # Check if attribute is already known or is a special attribute
        is_known = name in self.prefixes_ or name.rstrip('_') in self.prefixes_
        
        # Check if attribute is a special param (contains '__')
        is_special_param = '__' in name
        
        # Check if this is the first initialization of the object
        first_init = not hasattr(self, 'initialized_')
        
        # Check if attribute is a PyTorch component (Module or Optimizer)
        is_torch_component = isinstance(attr, (torch.nn.Module, torch.optim.Optimizer))

        # Conditional checks to determine whether to register the attribute or just set it as usual
        if not (is_known or is_special_param or first_init) and is_torch_component:
            # Validate if attribute can be set in the current context
            # self._check_settable_attr(name, attr)
            
            # Register the attribute into internal tracking lists
            # self._register_attribute(name, attr)
            pass
        # Perform the actual setting of the attribute
        super().__setattr__(name, attr)

    def __delattr__(self, name):
        """
        Remove an attribute from the instance with custom handling for certain types.

        This method is overridden to provide custom behavior for deleting attributes
        from neural network class instances. It unregisters PyTorch modules and optimizers
        from internal tracking lists before actually deleting the attribute to ensure
        proper cleanup.

        Parameters
        ----------
        name : str
            The name of the attribute to delete.

        Notes
        -----
        - This method utilizes `_unregister_attribute` to handle the deregistration of
        the attribute from tracking lists like `prefixes_` and `cuda_dependent_attributes_`.
        - Actual deletion of the attribute is performed after deregistration.

        Examples
        --------
        # Assuming an attribute named 'my_attribute' is registered:
        del obj.my_attribute
        """
        # Call internal method to unregister attribute from internal tracking lists
        self._unregister_attribute(name)
        
        # Perform the actual deletion of the attribute
        super().__delattr__(name)
        
    def _check_device(self, requested_device, map_device):
        """
        Check and resolve the device for neural network operations.

        This function compares a requested device with a device determined by mapping logic (e.g., PyTorch's default device).
        If there is a mismatch, it issues a warning and returns the mapped device.

        Parameters
        ----------
        requested_device : str or None
            The device specified by the user, such as 'cuda:0' or 'cpu'. If None, the choice is deferred to the system's logic.
        map_device : str
            The device determined by the system's mapping logic or configuration, typically the default device PyTorch would use.

        Returns
        -------
        str
            The resolved device to be used, based on system logic and user request.

        Warns
        -----
        DeviceWarning
            If there is a discrepancy between the `requested_device` and `map_device`, or if `requested_device` is None.

        Notes
        -----
        - This function is intended to ensure that operations are carried out on the correct computational device, especially 
        when dealing with GPU-enabled environments where device-specific actions are critical.
        - The comparison is made using the resolved PyTorch device objects to account for device types and indices (e.g., 'cuda:0').

        """

        if requested_device is None:
            # If the user didn't specify a device, use the mapped device and warn.
            msg = (
                f"Setting self.device = {map_device} since the requested device "
                f"was not specified"
            )
            warnings.warn(msg, DeviceWarning)
            return map_device

        # Convert to PyTorch device types for comparison
        type_1 = torch.device(requested_device)
        type_2 = torch.device(map_device)

        # Check if the types differ and warn if so, then return the mapped device
        if type_1 != type_2:
            warnings.warn(
                f'Setting self.device = {map_device} since the requested device ({requested_device}) '
                'is not available.',
                DeviceWarning
            )
            return map_device
        
        # If the types match, return the requested device (could be 'cuda:0' vs 'cuda:1')
        return requested_device
    
    def __repr__(self):
        """
        Compute the official string representation of the instance.

        Provides a representation that includes information about the instance's initialization state
        and lists attributes pertinent to the initialized state.

        Returns
        -------
        str
            A string representation of the instance that varies depending on whether the instance
            is initialized or not. For an uninitialized instance, attributes starting with 'module' 
            are included. For an initialized instance, attributes starting with 'module_' are included, 
            but those starting with 'module__' (indicating internal use) are excluded.

        Notes
        -----
        - The representation is designed to give quick insight into the instance's state and is 
        particularly useful for interactive work where instances are frequently printed and inspected.
        - This method may be overridden by subclasses to provide more detailed or specific representations
        based on additional attributes or states specific to the subclass.

        Examples
        --------
        >>> repr(my_neural_network)
        'NeuralNetwork(module_conv1=Conv2d(...), module_conv2=Conv2d(...), initialized=True)'
        
        In the above Examples, `my_neural_network` is an instance that has been initialized, so the 
        representation includes initialized module attributes with their corresponding values.
        """
        # Initial list of attribute keys to include and exclude in the representation
        to_include = ['module']
        to_exclude = []
        # If the network is uninitialized, specify that in the representation
        parts = [str(self.__class__) + '[uninitialized](']
        
        if self.initialized_:
            # If initialized, update the list of keys to include and exclude
            parts = [str(self.__class__) + '[initialized](']
            to_include = ['module_']
            to_exclude = ['module__']

        # Iterate through sorted dictionary items of the object
        for key, val in sorted(self.__dict__.items()):
            # Skip keys that don't match any prefix in to_include
            if not any(key.startswith(prefix) for prefix in to_include):
                continue
            # Skip keys that match any prefix in to_exclude
            if any(key.startswith(prefix) for prefix in to_exclude):
                continue

            # Convert value to string and handle multi-line strings
            val = str(val)
            if '\n' in val:
                val = '\n  '.join(val.split('\n'))
            
            # Append each key-value pair to the parts list
            parts.append('  {}={},'.format(key, val))

        # Close the representation and join all parts
        parts.append(')')
        return '\n'.join(parts)
    
class Classifier(BaseEstimator, ClassifierMixin):
    """
    A classifier built upon base estimators, enhanced with classification-specific capabilities.

    Inherits from `BaseEstimator` and `ClassifierMixin` to conform to the common interface
    for estimators in a machine learning framework, while providing added functionalities for
    classification tasks.

    Parameters
    ----------
    *args : list, optional
        Variable length argument list for parent `BaseEstimator` initializer.
    classes : array-like, optional
        Represents possible classes. Inferred from data during `fit` if not provided.
    **kwargs : dict, optional
        Arbitrary keyword arguments for parent `BaseEstimator` initializer.

    Attributes
    ----------
    classes_ : array-like
        Labels for classes, determined post-fit.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> classifier = Classifier(classes=np.unique(y))
    >>> classifier.fit(X_train, y_train)
    >>> predictions = classifier.predict(X_test)
    >>> print(predictions)
    [1 2 0 ...]

    The above Examples demonstrates creating an instance of `Classifier` with specified classes,
    fitting it to training data, and making predictions on test data.

    See Also
    --------
    BaseEstimator : Parent class providing foundational estimator functionality.
    ClassifierMixin : Mixin providing standardized classification functionalities.
    """

    def __init__(
            self,
            *args,
            classes=None,
            **kwargs
    ):
        """
        Constructs the Classifier instance with optional class labels and additional arguments.
        """

        super(Classifier, self).__init__(
            *args,
            **kwargs
        )

        self.classes = classes  

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', PassthroughScoring(
                name='train_loss',
                on_train=True,
            )),
            ('valid_loss', PassthroughScoring(
                name='valid_loss',
            )),
            ('valid_acc', EpochScoring(
                'accuracy',
                name='valid_acc',
                lower_is_better=False,
            )),
            ('print_log', PrintLog()),
        ]
    
    @property
    def classes_(self):
        """
        Accessor for the class labels used by the classifier.

        Provides the class labels if specified during initialization, otherwise, it
        attempts to retrieve the inferred class labels from the training data.

        Returns
        -------
        array-like
            The class labels.

        Raises
        ------
        AttributeError
            If class labels were neither provided nor inferred, or if the classifier
            was not trained with `y`.

        Notes
        -----
        To ensure class labels can be inferred, training must include the `y` target data.

        Examples
        --------
        >>> classifier = Classifier()
        >>> classifier.fit(X_train, y_train)
        >>> print(classifier.classes_)
        [0 1 2]

        The above Examples assumes that `classifier` has been fitted on the training data,
        showing how to access the inferred class labels.

        See Also
        --------
        Classifier : This property is associated with the Classifier class.
        """
        if self.classes is not None:
            if not len(self.classes):
                raise AttributeError("{} has no attribute 'classes_'".format(
                    self.__class__.__name__))
            return self.classes

        try:
            return self.classes_inferred_
        except AttributeError as exc:
            # It's not easily possible to track exactly what circumstances led
            # to this, so try to make an educated guess and provide a possible
            # solution.
            msg = (
                f"{self.__class__.__name__} could not infer the classes from y; "
                "this error probably occurred because the net was trained without y "
                "and some function tried to access the '.classes_' attribute; "
                "a possible solution is to provide the 'classes' argument when "
                f"initializing {self.__class__.__name__}"
            )
            raise AttributeError(msg) from exc

    def check_data(self, X, y):
        """
        Validates training data and labels before model fitting.

        Ensures `y` is not `None`, validates compatibility of `X` with expected formats,
        and infers unique class labels where applicable.

        Parameters
        ----------
        X : array-like or Dataset
            Training data.
        y : array-like, optional
            Labels for training data. Can be `None`.

        Raises
        ------
        ValueError
            If `y` is `None` and `X` is not a `Dataset` with labels or a custom `DataLoader`.

        Notes
        -----
        - Sets `classes_inferred_` attribute based on unique labels in `y` or extracted from `X`.
        - Suppressed `AttributeError` may occur during dataset-specific label extraction.

        Examples
        --------
        >>> net = NeuralNetwork(...)
        >>> X, y = load_some_data()
        >>> net.check_data(X, y)  # Performs validation on X and y

        This function is typically called internally before fitting a model to ensure
        data validity and compatibility with the training process.

        """
        # Check if y is None, X is not a Dataset, and the iterator for training is DataLoader
        if (
                (y is None) and
                (not is_dataset(X)) and
                (self.iterator_train is DataLoader)
        ):
            # Raise an error, suggesting the user to supply a DataLoader or Dataset
            msg = ("No y-values are given (y=None). You must either supply a "
                  "Dataset as X or implement your own DataLoader for "
                  "training (and your validation) and supply it using the "
                  "``iterator_train`` and ``iterator_valid`` parameters "
                  "respectively.")
            raise ValueError(msg)

        # If y is None but X is a Dataset, try to extract y from X
        if (y is None) and is_dataset(X):
            try:
                # Extract data and labels from the dataset
                _, y_ds = data_from_dataset(X)
                # Infer unique classes from the extracted labels and store it
                self.classes_inferred_ = np.unique(to_numpy(y_ds))
            except AttributeError:
                # If extraction fails, continue without raising an error
                pass

        # If y is provided, find the unique classes in y and store them
        if y is not None:
            self.classes_inferred_ = np.unique(to_numpy(y))

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        """
        Calculates loss using the model's loss criterion.

        If `NLLLoss` is used, applies a log transformation to `y_pred` before calculation.
        This method is designed to be compatible with custom loss criteria provided they conform 
        to the expected signature and usage pattern of PyTorch loss functions.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted outputs from the model.
        y_true : torch.Tensor
            True labels for the data.
        *args
            Variable length argument list for loss criterion.
        **kwargs
            Arbitrary keyword arguments for loss criterion.

        Returns
        -------
        torch.Tensor
            Computed loss between `y_pred` and `y_true`.

        Notes
        -----
        Assumes `criterion_` is set. Override if using custom criteria.

        Examples
        --------
        >>> net = Classifier(criterion=torch.nn.NLLLoss())
        >>> y_pred = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        >>> y_true = torch.tensor([0, 1])
        >>> loss = net.get_loss(y_pred, y_true)
        >>> print(loss)

        See Also
        --------
        torch.nn.NLLLoss: Loss criterion requiring log probabilities.

        """
        # Check if the criterion is NLLLoss (Negative Log Likelihood Loss)
        if isinstance(self.criterion_, torch.nn.NLLLoss):
            # Small value to avoid log(0)
            eps = torch.finfo(y_pred.dtype).eps
            # Apply log transformation to y_pred
            y_pred = torch.log(y_pred + eps)
        
        y_true = y_true.squeeze()
        # Call the superclass' get_loss method to compute the loss
        return super().get_loss(y_pred, y_true, *args, **kwargs)

    def fit(self, 
            X, 
            y=None, 
            optimizer=torch.optim.SGD,
            elbo=TraceMeanField_ELBO,
            callbacks=None,
            lr=0.01,
            epochs=10,
            batch_size=32,
            shuffle=False,
            verbose=1,
            model_params=None,
            warm_start=False,
            train_split=ValidSplit(5),
            **fit_params):
        """
        Fit the model to the input data.

        The method initializes and trains the model on the provided dataset. It handles the
        initialization of the optimizer and training parameters and orchestrates the training
        process over the specified number of epochs and batches. If `warm_start` is set to True
        and the model has been previously fitted, the training will resume from the last state.

        Parameters
        ----------
        X : Various types
            Input data, compatible with stockpy.dataset.StockpyDataset. See Notes for types.
        y : Various types, optional
            Target data, same types as `X`. If included in `X` as Dataset, can be None.
        optimizer : torch.optim.Optimizer, optional
            Optimizer for model training. Default: `torch.optim.SGD`.
        elbo : Callable, optional
            ELBO function for variational Bayesian methods. Default: `TraceMeanField_ELBO`.
        callbacks : list, optional
            List of stockpy.callbacks.Callback instances. Default: None.
        lr : float, optional
            Learning rate for the optimizer. Default: 0.01.
        epochs : int, optional
            Number of epochs for training. Default: 10.
        batch_size : int, optional
            Batch size for gradient computation. Default: 32.
        shuffle : bool, optional
            If True, shuffle data each epoch. Default: False.
        verbose : int, optional
            Verbosity level. More detail at higher levels. Default: 1.
        model_params : dict, optional
            Model-specific parameters. Default: None.
        warm_start : bool, optional
            If True, continue training from last state. Default: False.
        train_split : stockpy.helper.ValidSplit, optional
            `ValidSplit` instance for validation splits. Default: `ValidSplit(5)`.
        **fit_params
            Extra parameters for `forward` method and `train_split` method.

        Returns
        -------
        self : object
            Classifier instance after fitting.

        Examples
        --------
        >>> X, y = load_data()  # Placeholder for actual data loading.
        >>> model = Classifier()
        >>> model.fit(X, y)

        Notes
        -----
        Actual types for `X` and `y` are dependent on model and dataset implementation.
        Ensure data is in the expected format.

        See Also
        --------
        partial_fit : Incremental fit without re-initializing module.
        """
        # Call to the actual fitting logic remains unchanged
        return super(Classifier, self).fit(X,
                                           y, 
                                           optimizer,
                                           elbo,
                                           callbacks,
                                           lr,
                                           epochs,
                                           batch_size,
                                           shuffle,
                                           verbose,
                                           model_params,
                                           warm_start,
                                           train_split,
                                           **fit_params)

    def predict_proba(self, X):
        """
        Return probability estimates for samples.

        The method computes probability estimates for each class on the provided data. It is
        assumed that the `forward` method of the neural network returns a tuple where the first 
        element is a tensor of raw probabilities, which are then processed to form the output of 
        this method.

        Parameters
        ----------
        X : {array-like, torch.Tensor, pandas.DataFrame/Series, scipy.sparse.csr_matrix, dict, list/tuple, Dataset}
            The input data to predict probabilities for, which should be compatible with `stockpy.dataset.StockpyDataset`.
            The supported types are as follows:
            - numpy arrays
            - torch tensors
            - pandas DataFrame or Series
            - scipy sparse CSR matrices
            - dictionaries containing any of the above
            - lists or tuples containing any of the above
            - Dataset objects
            For other data types, a custom Dataset should be provided.

        Returns
        -------
        y_proba : numpy.ndarray
            An array of shape (n_samples, n_classes) with the probability estimates that the samples belong to each class.

        Notes
        -----
        - This method assumes that the `forward` method of the neural network returns the raw probabilities as its first output.
        - If `forward` returns additional outputs that are required, this method should not be used; instead, one should use the `NeuralNet.forward` method to get all outputs.
        - The implementation uses the `predict_proba` method from the superclass for the actual prediction.

        Examples
        --------
        >>> X = load_data()  # Placeholder for actual data loading.
        >>> model = Classifier()
        >>> probabilities = model.predict_proba(X)
        """
        # Call to superclass to get probability predictions
        return super().predict_proba(X)

    def predict(self, X, predict_nonlinearity='auto'):
        """
        Return class labels for samples in X.

        The method predicts class labels for the input samples by first obtaining the probability
        estimates and then selecting the class with the highest probability as the predicted class.

        Parameters
        ----------
        X : {array-like, torch.Tensor, pandas.DataFrame/Series, scipy.sparse.csr_matrix, dict, list/tuple, Dataset}
            The input data for which to predict class labels, which should be compatible with `stockpy.dataset.StockpyDataset`.
            Supported data types include:
            - numpy arrays
            - torch tensors
            - pandas DataFrame or Series
            - scipy sparse CSR matrices
            - dictionaries containing any of the above types
            - lists or tuples containing any of the above types
            - Dataset objects
            Custom Dataset implementations should be used for data types not listed here.

        predict_nonlinearity : {'auto', callable, None}, default 'auto'
            The nonlinearity function to be applied to the network's output before determining the predicted class.
            Can be 'auto' to use the default from stockpy, a callable for a custom nonlinearity, or None to apply no nonlinearity.

        Returns
        -------
        y_pred : numpy.ndarray
            Predicted class labels for the samples, with shape (n_samples,).

        Notes
        -----
        - The network's `forward` method is expected to return the class scores or probabilities as the first element of a tuple.
        - The prediction process applies a nonlinearity (typically softmax) to the network's output, followed by an argmax operation to derive the class labels.
        - If a `predict_nonlinearity` is specified, it is applied before the argmax step.

        Examples
        --------
        >>> X = load_data()  # Placeholder for actual data loading.
        >>> model = Classifier()
        >>> predictions = model.predict(X)
        """
        # Assuming 'X' and 'predict_nonlinearity' are already defined above this snippet
        if not isinstance(X, torch.utils.data.dataset.Subset) and X.ndim == 1:
            X = X.reshape(1, -1)

        self.predict_nonlinearity = predict_nonlinearity
        
        return self.predict_proba(X).argmax(axis=1)

class Regressor(BaseEstimator, RegressorMixin):
    """
    A regressor that conforms to scikit-learn's estimator interface.

    This class is designed to work with PyTorch modules for regression tasks,
    wrapping a neural network model and providing an sklearn-like API.

    The class is derived from `BaseEstimator` and `RegressorMixin`, which provide
    base functionality for all scikit-learn estimators and regression-specific
    functionality, respectively.

    Parameters
    ----------
    *args
        Variable length argument list passed to the `BaseEstimator` constructor.

    **kwargs
        Arbitrary keyword arguments passed to the `BaseEstimator` constructor.
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
    
        super(Regressor, self).__init__(
            *args,
            **kwargs
        )

    def check_data(self, X, y):
        """
        Validate the input data's format for compatibility with the neural network's training process.

        This method ensures that the input data `X` and targets `y` are in a suitable format for the model's
        training routine. If `y` is not provided, it further checks whether `X` is compatible with the model's
        dataset requirements. An error is raised if the input data is not a dataset or DataLoader when `y` is `None`.

        Parameters
        ----------
        X : various types
            The input data to be validated. Acceptable formats include:
            - numpy arrays
            - torch tensors
            - pandas DataFrame or Series
            - scipy sparse CSR matrices
            - dictionaries containing any of the above
            - lists/tuples containing any of the above
            - Dataset instances

        y : array-like, optional
            The target values corresponding to `X`. If `X` already includes targets or a custom DataLoader
            is being used, `y` may be None. Defaults to None.

        Raises
        ------
        ValueError
            If `y` is None and `X` is not a dataset or DataLoader, and a DataLoader is expected for the
            training process.

        Returns
        -------
        None

        Notes
        -----
        Custom DataLoader users should ensure that `y` is None and `X` is appropriately structured to
        provide the necessary data for training and validation through their DataLoader implementation.
        """

        # Check if y is None and if X is not a Dataset and the training iterator is DataLoader
        if (y is None) and (not is_dataset(X)) and (self.iterator_train is DataLoader):
            raise ValueError("No y-values are given (y=None). You must "
                            "implement your own DataLoader for training "
                            "(and your validation) and supply it using the "
                            "``iterator_train`` and ``iterator_valid`` "
                            "parameters respectively.")
                            
        
        # If y is None, the user has their own mechanism for generating y-values.
        if y is None:
            return

    def fit(self, 
            X, 
            y=None, 
            optimizer=torch.optim.SGD,
            elbo=TraceMeanField_ELBO,
            callbacks=None,
            lr=0.01,
            epochs=10,
            batch_size=32,
            shuffle=False,
            verbose=1,
            model_params=None,
            warm_start=False,
            train_split=ValidSplit(5),
            **fit_params):
        """
        Fit the model to the training data.

        Extends the fit method from the parent `NeuralNet` class, this method initializes
        or updates the model's weights using the provided training data `X` and targets `y`.

        Parameters
        ----------
        X : array-like, DataFrame, sparse matrix, or Dataset
            The input data for training the model. It can be in various formats like numpy arrays,
            torch tensors, pandas DataFrame or Series, scipy sparse CSR matrices, or any structure
            of the above in a dictionary or list/tuple, and also a custom Dataset object.

        y : array-like or None, optional
            The target values (i.e., labels or ground truth) to train the model with. If `X` includes
            the target values, or they are generated within a custom Dataset, `y` can be set to None.

        optimizer : torch.optim.Optimizer, optional
            The optimization algorithm to use for updating the model's parameters. Default is SGD.

        elbo : pyro.infer.ELBO subclass, optional
            The Evidence Lower Bound (ELBO) loss from Pyro's inference library to use in case of
            probabilistic models. The default is `TraceMeanField_ELBO`.

        callbacks : list of Callback instances, optional
            List of callback functions or instances of `stockpy.callbacks.Callback`. These will be called
            at specific points during training (e.g., on epoch start/end).

        lr : float, optional
            Learning rate for the optimizer. Default is 0.01.

        epochs : int, optional
            Number of complete passes through the training data. Default is 10.

        batch_size : int, optional
            Number of samples per batch to load. Default is 32.

        shuffle : bool, optional
            Whether to shuffle the training data before each epoch. Default is False.

        verbose : int, optional
            Verbosity level; the higher, the more messages. For `verbose=0`, no messages are printed.
            Default is 1.

        model_params : dict, optional
            Additional parameters to pass to the model upon construction. This can be useful to set
            parameters of the model that are not optimized.

        warm_start : bool, optional
            If set to `True`, the model is not re-initialized and training continues from where it left
            off. Default is False.

        train_split : stockpy.helper.ValidSplit or None, optional
            Strategy to split the training data into training and validation sets. A `ValidSplit`
            instance or any other valid sklearn splitter can be used. The default `ValidSplit(5)`
            performs a 5-fold cross-validation.

        **fit_params : dict
            Additional parameters passed to the `forward` method of the module and to the
            `train_split` callable.

        Returns
        -------
        self : object
            Returns the instance itself.

        Notes
        -----
        You should override this method if your workflow demands a pre-fit or post-fit processing.
        """
            
        return super(Regressor, self).fit(X,
                                          y, 
                                          optimizer,
                                          elbo,
                                          callbacks,
                                          lr,
                                          epochs,
                                          batch_size,
                                          shuffle,
                                          verbose,
                                          model_params,
                                          warm_start,
                                          train_split,
                                          **fit_params)
    
    def predict(self, 
                X,
                predict_nonlinearity='auto'):
        
        """
        Predict continuous target values for samples in X.

        The `predict` method is designed for regression tasks where the output
        is a continuous variable. If the module's `forward` method returns multiple
        outputs as a tuple, it is assumed that the first output contains the
        prediction values.

        Parameters
        ----------
        X : array-like, DataFrame, sparse matrix, or Dataset
            The input data for making predictions. The data must be compatible with
            the format expected by the `Dataset` used within the `NeuralNet` class.
            This includes:

            - numpy arrays
            - torch tensors
            - pandas DataFrame or Series
            - scipy sparse CSR matrices
            - a dictionary containing any of the above
            - a list/tuple containing any of the above
            - a Dataset object

            If your data doesn't fit into these categories, you should pass a custom
            `Dataset` that can handle your data format.

        predict_nonlinearity : callable or None, optional
            A callable that applies a nonlinearity to the output of the model's
            forward method. This can be used to apply a final activation function
            to the predictions. If set to None, no nonlinearity is applied.

        Returns
        -------
        y_pred : numpy ndarray
            The predicted values as a one-dimensional array.

        """
        # Assuming 'X' and 'predict_nonlinearity' are already defined above this snippet
        if not isinstance(X, torch.utils.data.dataset.Subset) and X.ndim == 1:
            X = X.reshape(1, -1)
            
        # initialize non linearity
        self.predict_nonlinearity = predict_nonlinearity

        return super().predict_proba(X)
    
# class NumericalGenerator(BaseEstimator, RegressorMixin):

#     def __init__(
#             self,
#             *args,
#             **kwargs
#     ):
#         super(NumericalGenerator, self).__init__(
#             *args,
#             **kwargs
#         )

#     def check_data(self, X, y):
#         """
#         Validate that the input data is appropriate for training the neural network.

#         This function checks if both `X` and `y` are provided in an appropriate format
#         for the model's training iterator. If `y` is not provided, and the input data `X`
#         is not a dataset compatible with the training iterator, then a `ValueError` is raised.

#         Parameters
#         ----------
#         X : various types
#             Input data, compatible with stockpy.dataset.StockpyDataset. You should be able to pass:
#             - numpy arrays
#             - torch tensors
#             - pandas DataFrame or Series
#             - scipy sparse CSR matrices
#             - a dictionary containing any of the above types
#             - a list/tuple containing any of the above types
#             - a Dataset

#         y : array-like, optional
#             Labels for input data `X`. It is optional if you implement your own DataLoader.
#             Default is None.

#         Raises
#         ------
#         ValueError
#             If `y` is None and the input data `X` is neither a dataset nor a DataLoader.

#         Returns
#         -------
#         None
#             This function doesn't return anything; it only validates the input data.

#         Examples
#         --------
#         >>> net = NeuralNetClassifier(MyModule)
#         >>> X = np.random.rand(100, 20)
#         >>> y = np.random.randint(0, 2, 100)
#         >>> net.check_data(X, y)  # Should not raise any errors

#         Notes
#         -----
#         If you're providing a custom DataLoader, ensure that `y` is set to `None`.

#         """
#         # Check if y is None and if X is not a Dataset and the training iterator is DataLoader
#         if (y is None) and (not is_dataset(X)) and (self.iterator_train is DataLoader):
#             raise ValueError("No y-values are given (y=None). You must "
#                             "implement your own DataLoader for training "
#                             "(and your validation) and supply it using the "
#                             "``iterator_train`` and ``iterator_valid`` "
#                             "parameters respectively.")
                            
        
#         # If y is None, the user has their own mechanism for generating y-values.
#         if y is None:
#             return

#     # pylint: disable=signature-differs
#     def fit(self, 
#             X, 
#             y=None, 
#             optimizer=torch.optim.SGD,
#             lr=0.01,
#             epochs=10,
#             batch_size=32,
#             shuffle=False,
#             verbose=1,
#             warm_start=False,
#             model_params=False,
#             train_split=ValidSplit(5),
#             **fit_params):
#         """
#         Fit the model to the given data.

#         This method is an override of the ``NeuralNet.fit`` method. In contrast
#         to the parent method, the ``y`` parameter is non-optional to ensure that
#         the user doesn't forget to include labels. However, if the labels ``y`` 
#         are derived dynamically from the input data ``X``, then ``y`` can be set 
#         to ``None``.

#         Parameters
#         ----------
#         X : array-like or Dataset
#             Training data. You should be able to pass:
#             - numpy arrays
#             - torch tensors
#             - pandas DataFrame or Series
#             - scipy sparse CSR matrices
#             - a dictionary of the above types
#             - a list/tuple of the above types
#             - a Dataset
            
#         y : array-like, optional
#             Target values. While this parameter is non-optional, you can set it 
#             to ``None`` if labels are derived from ``X`` dynamically.
            
#         **fit_params : dict
#             Additional fitting parameters that will be passed to the base
#             ``NeuralNet.fit`` method.

#         Returns
#         -------
#         self : object
#             Returns self for method chaining.

#         Examples
#         --------
#         >>> net = Regressor(MyModule)
#         >>> X = np.random.rand(100, 20)
#         >>> y = np.random.rand(100)
#         >>> net.fit(X, y)  # Should fit the model to the data

#         Notes
#         -----
#         If you encounter a pylint bug saying "useless-super-delegation,"
#         you can safely ignore it as it is a known pylint issue.

#         """
#         # pylint: disable=useless-super-delegation
#         # this is actually a pylint bug:
#         # https://github.com/PyCQA/pylint/issues/1085
            
#         self.optimizer = optimizer
#         self.lr = lr
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.verbose = verbose
#         self.warm_start = warm_start
#         self.train_split = train_split

#         X, y = self._validate_data(
#             X, y, accept_sparse=["csr", "csc", "coo"], multi_output=True
#         )

#         # Ensure y is 2D
#         if y.ndim == 1:
#             y = y.reshape((-1, 1))

#         self.n_outputs_ = y.shape[1]
        
#         # Check if the model should be re-initialized. If warm_start is True and the
#         # model is already initialized, skip the re-initialization.
#         if not self.warm_start or not self.initialized_:
#             self.initialize()

#         # Perform the partial fit, which is the actual fitting process.
#         self.partial_fit(X, y, **fit_params)
        
#         return self
    
#     def sample(self, X, num_samples):
#         """
#         Sample a sequence from the trained model given an initial input.
        
#         Parameters
#         ----------
#         model : nn.Module
#             The trained sequence-to-sequence model.
#         initial_input : torch.Tensor
#             The initial input tensor, shaped [batch_size, 1, input_dim].
#         num_samples : int
#             The number of samples to generate.

#         Returns
#         -------
#         samples : torch.Tensor
#             The generated samples, shaped [batch_size, num_samples, input_dim].
#         """
        
#         input = X
#         all_samples = []

#         # Get the dataset object from the input data
#         dataset = self.get_dataset(X)
        
#         # Create an iterator for the dataset, optionally set to training mode
#         iterator = self.get_iterator(dataset, training=False)

#         for i, batch in enumerate(iterator):
#             # Initialize hidden and cell states using the encoder
#             encoder_outputs, hidden, cell = self.encoder(batch[0])
            
#             # Initialize the first input to the Decoder
#             input = to_device(torch.zeros(batch[0].size(0), 1, self.decoder.output_dim), self.device)

#             samples = []  # Initialize an empty list for samples for this batch

#             # Autoregressive sampling
#             for t in range(num_samples):
#                 # Use the model's forward function to predict the next output
#                 output, hidden, cell = self.decoder(input, hidden, cell)
                
#                 # Append the output to the list of samples
#                 samples.append(output.squeeze(1))
                
#                 # Update the next input to be the output
#                 input = output

#             # Convert the list of samples to a tensor and append to all_samples
#             samples = torch.stack(samples, dim=1)
#             all_samples.append(samples)
        
#         # Concatenate all_samples to form a single tensor
#         all_samples = torch.cat(all_samples, dim=0)

#         return all_samples
    
#     def predict(self, 
#                 X,
#                 predict_nonlinearity='auto'):
#         """Where applicable, return class labels for samples in X.

#         If the module's forward method returns multiple outputs as a
#         tuple, it is assumed that the first output contains the
#         relevant information and the other values are ignored. If all
#         values are relevant, consider using
#         :func:`~stockpy.NeuralNet.forward` instead.

#         Parameters
#         ----------
#         X : input data, compatible with stockpy.dataset.StockpyDataset
#           By default, you should be able to pass:

#             * numpy arrays
#             * torch tensors
#             * pandas DataFrame or Series
#             * scipy sparse CSR matrices
#             * a dictionary of the former three
#             * a list/tuple of the former three
#             * a Dataset

#           If this doesn't work with your data, you have to pass a
#           ``Dataset`` that can deal with the data.

#         Returns
#         -------
#         y_pred : numpy ndarray

#         """
#         if X.ndim == 1:
#             X = X.reshape(1, -1)

#         # initialize non linearity
#         self.predict_nonlinearity = predict_nonlinearity

#         return super().predict_proba(X)
    
# class CategoricalGenerator(BaseEstimator, ClassifierMixin):

#     def __init__(
#             self,
#             *args,
#             **kwargs
#     ):
#         super(CategoricalGenerator, self).__init__(
#             *args,
#             **kwargs
#         )

#     def check_data(self, X, y):
#         """
#         Validate that the input data is appropriate for training the neural network.

#         This function checks if both `X` and `y` are provided in an appropriate format
#         for the model's training iterator. If `y` is not provided, and the input data `X`
#         is not a dataset compatible with the training iterator, then a `ValueError` is raised.

#         Parameters
#         ----------
#         X : various types
#             Input data, compatible with stockpy.dataset.StockpyDataset. You should be able to pass:
#             - numpy arrays
#             - torch tensors
#             - pandas DataFrame or Series
#             - scipy sparse CSR matrices
#             - a dictionary containing any of the above types
#             - a list/tuple containing any of the above types
#             - a Dataset

#         y : array-like, optional
#             Labels for input data `X`. It is optional if you implement your own DataLoader.
#             Default is None.

#         Raises
#         ------
#         ValueError
#             If `y` is None and the input data `X` is neither a dataset nor a DataLoader.

#         Returns
#         -------
#         None
#             This function doesn't return anything; it only validates the input data.

#         Examples
#         --------
#         >>> net = NeuralNetClassifier(MyModule)
#         >>> X = np.random.rand(100, 20)
#         >>> y = np.random.randint(0, 2, 100)
#         >>> net.check_data(X, y)  # Should not raise any errors

#         Notes
#         -----
#         If you're providing a custom DataLoader, ensure that `y` is set to `None`.

#         """
#         # Check if y is None and if X is not a Dataset and the training iterator is DataLoader
#         if (y is None) and (not is_dataset(X)) and (self.iterator_train is DataLoader):
#             raise ValueError("No y-values are given (y=None). You must "
#                             "implement your own DataLoader for training "
#                             "(and your validation) and supply it using the "
#                             "``iterator_train`` and ``iterator_valid`` "
#                             "parameters respectively.")
                            
        
#         # If y is None, the user has their own mechanism for generating y-values.
#         if y is None:
#             return

#     # pylint: disable=signature-differs
#     def fit(self, 
#             X, 
#             y=None, 
#             optimizer=torch.optim.SGD,
#             lr=0.01,
#             epochs=10,
#             batch_size=32,
#             shuffle=False,
#             verbose=1,
#             model_params=None,
#             warm_start=False,
#             train_split=ValidSplit(5),
#             **fit_params):
#         """
#         Fit the model to the given data.

#         This method is an override of the ``NeuralNet.fit`` method. In contrast
#         to the parent method, the ``y`` parameter is non-optional to ensure that
#         the user doesn't forget to include labels. However, if the labels ``y`` 
#         are derived dynamically from the input data ``X``, then ``y`` can be set 
#         to ``None``.

#         Parameters
#         ----------
#         X : array-like or Dataset
#             Training data. You should be able to pass:
#             - numpy arrays
#             - torch tensors
#             - pandas DataFrame or Series
#             - scipy sparse CSR matrices
#             - a dictionary of the above types
#             - a list/tuple of the above types
#             - a Dataset
            
#         y : array-like, optional
#             Target values. While this parameter is non-optional, you can set it 
#             to ``None`` if labels are derived from ``X`` dynamically.
            
#         **fit_params : dict
#             Additional fitting parameters that will be passed to the base
#             ``NeuralNet.fit`` method.

#         Returns
#         -------
#         self : object
#             Returns self for method chaining.

#         Examples
#         --------
#         >>> net = Regressor(MyModule)
#         >>> X = np.random.rand(100, 20)
#         >>> y = np.random.rand(100)
#         >>> net.fit(X, y)  # Should fit the model to the data

#         Notes
#         -----
#         If you encounter a pylint bug saying "useless-super-delegation,"
#         you can safely ignore it as it is a known pylint issue.

#         """
#         # import pandas as pd

#         #    data = pd.read_pickle('../test/data.pickle')
#         #    X = data.drop(['scenario'], axis=1)
#         #    y = data['scenario']

#         #    y = y.replace({1: 0, 2: 1, 3: 2, 4: 3})
#         # pylint: disable=useless-super-delegation
#         # this is actually a pylint bug:
#         # https://github.com/PyCQA/pylint/issues/1085
            
#         self.optimizer = optimizer
#         self.lr = lr
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.verbose = verbose
#         self.warm_start = warm_start
#         self.train_split = train_split

#         X, y = self._validate_data(
#             X, y, accept_sparse=["csr", "csc", "coo"], multi_output=True
#         )

#         # Ensure y is 2D
#         if y.ndim == 1:
#             y = y.reshape((-1, 1))

#         self.n_outputs_ = y.shape[1]
        
#         # Check if the model should be re-initialized. If warm_start is True and the
#         # model is already initialized, skip the re-initialization.
#         if not self.warm_start or not self.initialized_:
#             self.initialize()

#         # Perform the partial fit, which is the actual fitting process.
#         self.partial_fit(X, y, **fit_params)
        
#         return self
