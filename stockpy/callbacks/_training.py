""" Callbacks related to training progress. """

import os
import pickle
import warnings
from fnmatch import fnmatch
from copy import deepcopy

import numpy as np
from stockpy.callbacks import Callback
from stockpy.exceptions import StockpyException
from stockpy.utils import _check_f_arguments
from stockpy.utils import noop
from stockpy.utils import open_file_like


__all__ = ['Checkpoint', 'EarlyStopping']


class Checkpoint(Callback):
    """
    Save the model and additional states during training if a specified metric improved.

    This callback typically interacts with the validation scoring callback,
    which updates the history with metrics such as ``valid_loss_best``.
    This history update allows the `Checkpoint` callback to identify the
    optimal point to save a snapshot of the model's state.

    Parameters
    ----------
    monitor : str, function, None
        The metric from the history to monitor or a function that returns a
        boolean indicating whether the model's current state should be saved.
        If `None`, the model is saved every epoch.
    f_pickle : file-like object, str, None
        Path or object for pickling the entire model. Set to `None` to skip.
    fn_prefix : str
        Prefix added to all file names if they are strings.
    dirname : str
        Directory where checkpoint files are stored.
    event_name : str
        History column name for checkpointing events. Set to `None` to skip.
    sink : callable
        Function to process messages regarding checkpoint operations.
    load_best : bool
        If `True`, loads the best checkpoint once training is finished.
    use_safetensors : bool
        If `True`, uses `safetensors` library for saving state, restricting
        persistence to PyTorch tensors only.

    Notes
    -----
    - The monitor function must be a callable that takes a single argument,
      the neural network instance, and returns a boolean.
    - Format specifiers in file paths allow dynamic naming based on the model's
      training progress.
    - Restoring the best model requires `monitor` to be specified and not `None`.
    - `safetensors` option is useful for ensuring compatibility with non-Python
      environments but limits the types of objects that can be saved.

    Examples
    --------
    Using the `Checkpoint` callback with default settings:

    >>> net = MyNet(callbacks=[Checkpoint()])
    >>> net.fit(X, y)

    Using a custom `monitor` function to determine checkpointing:

    >>> monitor = lambda net: net.history[-1]['valid_loss'] < 0.2
    >>> net = MyNet(callbacks=[Checkpoint(monitor=monitor)])
    >>> net.fit(X, y)

    Including the epoch number in the saved parameters file name:

    >>> cb = Checkpoint(f_params="params_{last_epoch[epoch]}.pt")
    >>> net = MyNet(callbacks=[cb])
    >>> net.fit(X, y)
    """
    def __init__(
            self,
            monitor='valid_loss_best',
            f_pickle=None,
            fn_prefix='',
            dirname='',
            event_name='event_cp',
            sink=noop,
            load_best=False,
            use_safetensors=False,
            **kwargs
    ):
        self.monitor = monitor
        self.f_pickle = f_pickle
        self.fn_prefix = fn_prefix
        self.dirname = dirname
        self.event_name = event_name
        self.sink = sink
        self.load_best = load_best
        self.use_safetensors = use_safetensors
        self._check_kwargs(kwargs)
        vars(self).update(**kwargs)
        self._validate_filenames()

    def _check_kwargs(self, kwargs):
        """
        Checks if additional keyword arguments follow the expected naming pattern and are compatible with safetensors.

        This method ensures that all keyword arguments that are not predefined parameters of the `Checkpoint` class
        start with ``f_`` (assuming they are intended to specify file-like objects for saving different states).

        It also verifies that if `use_safetensors` is enabled, the `f_optimizer` attribute must not be set because
        `safetensors` is not capable of saving the optimizer state.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments passed to the `Checkpoint` class.

        Raises
        ------
        TypeError
            If any keyword arguments do not follow the ``f_`` naming convention.
        ValueError
            If `use_safetensors` is True and `f_optimizer` is not None, indicating an incompatible configuration.

        Notes
        -----
        - The method is intended to be called internally by the `Checkpoint` class's `__init__` method.
        - Users of the `Checkpoint` class are not expected to call this method directly.

        Examples
        --------
        The following example demonstrates the creation of a `Checkpoint` object with an incorrect keyword argument,
        which would raise a TypeError:

        >>> net = MyNet(callbacks=[Checkpoint(foo='bar')])  # This would raise a TypeError.
        """

        for key in kwargs:
            if not key.startswith('f_'):
                raise TypeError(
                    "{cls_name} got an unexpected argument '{key}', did you mean "
                    "'f_{key}'?".format(cls_name=self.__class__.__name__, key=key))
        if self.use_safetensors and self.f_optimizer is not None:
            raise ValueError(
                "Cannot save optimizer state when using safetensors, "
                "please set f_optimizer=None or don't use safetensors.")

    def initialize(self):
        """
        Prepares the checkpoint callback for use.

        This method performs initialization steps for the checkpoint callback. It calls
        `_validate_filenames()` to ensure that the filenames for saving states are in a proper format.
        Additionally, it checks if the specified `dirname` exists where checkpoint files will be saved.
        If the directory does not exist, it is created.

        Returns
        -------
        self : Checkpoint
            The instance of the Checkpoint class is returned to allow for method chaining.

        Notes
        -----
        The `initialize` method is typically called when setting up callbacks for the neural network
        before training begins.
        The method ensures that the directory structure required for saving checkpoints is in place
        before the training process attempts to save any checkpoints.

        Examples
        --------
        The following example demonstrates the instantiation of a Checkpoint callback and its initialization:

        >>> checkpoint_callback = Checkpoint(dirname='checkpoints')
        >>> checkpoint_callback.initialize()
        """

        self._validate_filenames()
        if self.dirname and not os.path.exists(self.dirname):
            os.makedirs(self.dirname, exist_ok=True)
        return self

    def on_train_end(self, net, **kwargs):
        """
        Load the best model checkpoint after training has finished.

        This method is part of the training loop and is executed when the training ends.
        If the `load_best` attribute is `True` and the `monitor` attribute is not `None`,
        the best model checkpoint is loaded into the neural network. If these conditions are not met,
        the method exits without loading any checkpoints.

        Parameters
        ----------
        net : NeuralNet
            The neural network instance that is being trained.
        **kwargs : dict
            Additional keyword arguments not used by this callback.

        Examples
        --------
        >>> net = NeuralNet(
        ...     module=MyModule,
        ...     callbacks=[Checkpoint(load_best=True)]
        ... )
        >>> net.fit(X, y)
        >>> # After training, best model is automatically loaded

        Notes
        -----
        - This method is automatically called by the training loop and is not intended to be
          called manually.
        
        - Loading of the best checkpoint is only performed if `load_best` is `True`, which
          is helpful in conjunction with callbacks like early stopping.

        """

        if not self.load_best or self.monitor is None:
            return
        self._sink("Loading best checkpoint after training.", net.verbose)
        net.load_params(checkpoint=self, use_safetensors=self.use_safetensors)

    def on_epoch_end(self, net, **kwargs):
        """
        Determine if a checkpoint should be saved at the end of an epoch.

        This method checks the monitor parameter at the end of each epoch to decide if 
        the state of the network should be checkpointed. If the monitor parameter is 
        present in the history and suffixed with '_best', a warning is issued to check if 
        this was intentional. It records the checkpoint event and saves the model if 
        the conditions are met.

        Parameters
        ----------
        net : NeuralNet
            The neural network instance that is being trained.
        **kwargs : dict
            Additional keyword arguments not used by this callback.

        Raises
        ------
        StockpyException
            If the monitoring parameter is not found in the history, and it is expected 
            to be there (e.g., when using validation scores for checkpointing).

        Examples
        --------
        >>> net = NeuralNet(
        ...     module=MyModule,
        ...     callbacks=[Checkpoint(monitor='valid_loss')]
        ... )
        >>> net.fit(X, y)
        >>> # Checkpoints might be saved based on the validation loss.

        Notes
        -----
        - This method is called automatically at the end of each epoch and is not meant 
          for manual invocation.

        - The decision to checkpoint can be based on a pre-defined history key, a 
          custom callable, or every epoch if the monitor is set to `None`.

        """

        if "{}_best".format(self.monitor) in net.history[-1]:
            warnings.warn(
                "Checkpoint monitor parameter is set to '{0}' and the history "
                "contains '{0}_best'. Perhaps you meant to set the parameter "
                "to '{0}_best'".format(self.monitor), UserWarning)

        if self.monitor is None:
            do_checkpoint = True
        elif callable(self.monitor):
            do_checkpoint = self.monitor(net)
        else:
            try:
                do_checkpoint = net.history[-1, self.monitor]
            except KeyError as e:
                msg = (
                    f"{e.args[0]} Make sure you have validation data if you use "
                    "validation scores for checkpointing.")
                raise StockpyException(msg)

        if self.event_name is not None:
            net.history.record(self.event_name, bool(do_checkpoint))

        if do_checkpoint:
            self.save_model(net)
            self._sink("A checkpoint was triggered in epoch {}.".format(
                len(net.history) + 1
            ), net.verbose)

    def _f_kwargs(self):
        """
        Collect file-related attributes from the instance.

        This method retrieves all attributes of the instance whose names start with ``f_`` 
        and returns them as a dictionary, with the exception of 'f_history_'. This 
        dictionary is likely used to dynamically handle file operations for checkpointing 
        different components of the training process.

        Returns
        -------
        dict
            A dictionary where keys are attribute names that begin with ``f_``, and values 
            are their corresponding values from the instance.

        Examples
        --------
        >>> checkpoint = Checkpoint(f_params="params.pt", f_optimizer="optimizer.pt")
        >>> checkpoint._f_kwargs()
        {'f_params': 'params.pt', 'f_optimizer': 'optimizer.pt'}

        Notes
        -----
        - This method is intended for internal use to abstract the collection of file 
        paths and is not part of the public API of the callback.
        - The method explicitly excludes the 'f_history_' attribute, which may have 
        separate handling logic.
        """

        return {key: getattr(self, key) for key in dir(self)
                if key.startswith('f_') and (key != 'f_history_')}

    def save_model(self, net):
        """
        Save the model to files.

        This method saves various components of the model based on the current 
        state of the `Checkpoint` instance. It can handle saving model parameters, 
        optimizer state, criterion state, training history, custom modules, and the 
        entire model object depending on the attributes set in the `Checkpoint` instance.

        The method determines which components to save based on the presence of
        corresponding file path attributes starting with ``f_``.

        Parameters
        ----------
        net : NeuralNet
            The neural network instance from which state will be saved.

        Raises
        ------
        StockpyException
            If any of the specified file paths are not writable or if any other
            issues are encountered during the save process.

        Examples
        --------
        >>> checkpoint = Checkpoint(f_params='params.pt', f_optimizer='optimizer.pt')
        >>> net = NeuralNet(...)
        >>> checkpoint.save_model(net)

        Notes
        -----
        - The actual file paths are determined using the `_format_target` method
          and support dynamic naming based on the current training state, such as 
          epoch number.
        - If a file path attribute is set to `None`, the corresponding component 
          will not be saved.
        - The history is saved in a JSON format, and the entire model object is 
          pickled.
        - This method is a part of the `Checkpoint` callback's internal logic and
          is not intended to be invoked directly by users.

        """

        kwargs_module, kwargs_other = _check_f_arguments(
            self.__class__.__name__, **self._f_kwargs())

        for key, val in kwargs_module.items():
            if val is None:
                continue

            f = self._format_target(net, val, -1)
            key = key[:-1]  # remove trailing '_'
            self._save_params(f, net, 'f_' + key, key + " state")

        f_history = kwargs_other.get('f_history')
        if f_history is not None:
            f = self.f_history_
            self._save_params(f, net, "f_history", "history")

        f_pickle = kwargs_other.get('f_pickle')
        if f_pickle:
            f_pickle = self._format_target(net, f_pickle, -1)
            with open_file_like(f_pickle, 'wb') as f:
                pickle.dump(net, f)

    @property
    def f_history_(self):
        """
        Constructs the full path for the training history file.

        This property appends the directory name and filename prefix to the base
        training history filename provided by the user (`self.f_history`). If 
        `self.f_history` is `None`, it simply returns `None`, indicating that the 
        history will not be saved to a file.

        Returns
        -------
        str or None
            The fully-qualified path to the training history file, or `None` if
            `self.f_history` is `None`.

        Examples
        --------
        >>> checkpoint = Checkpoint(dirname='checkpoints', fn_prefix='run1_', f_history='history.json')
        >>> checkpoint.f_history_
        'checkpoints/run1_history.json'

        Notes
        -----
        - This property is used internally to determine where to save the training 
          history when the `save_model` method is called.
        - The directory and prefix are set when the `Checkpoint` instance is 
          initialized. If they are not set, the training history filename will 
          just be `self.f_history`.
        - If the checkpoint has not been initialized (i.e., `self.dirname` is not 
          set), calling this property will still return the correct path assuming 
          `self.f_history` is not `None`.

        """
        if self.f_history is None:
            return None
        return os.path.join(
            self.dirname, self.fn_prefix + self.f_history)

    def get_formatted_files(self, net):
        """
        Generates a dictionary with keys representing different checkpoint file types
        and values being the fully-qualified file paths for each type.

        This method takes the last checkpoint index from the training history where
        the event specified by `event_name` occurred. If `event_name` is not in the
        history or has no recorded event, it defaults to the last entry (`idx = -1`).

        Parameters
        ----------
        net : nn.Module or PyroModule
            The neural network instance from which the history will be retrieved
            and used for formatting the file names.

        Returns
        -------
        dict
            A dictionary where the keys are the file type identifiers (e.g., 'f_params',
            'f_optimizer') and the values are the corresponding formatted file paths.

        Examples
        --------
        >>> net = NeuralNet(...)
        >>> checkpoint = Checkpoint(dirname='checkpoints', fn_prefix='run1_')
        >>> checkpoint.get_formatted_files(net)
        {
            'f_params': 'checkpoints/run1_params.pt',
            'f_optimizer': 'checkpoints/run1_optimizer.pt',
            'f_history': 'checkpoints/run1_history.json',
            ...
        }

        Notes
        -----
        - The method uses `self._format_target` to create the file paths, which 
          applies any formatting rules such as including the epoch number or other 
          placeholders.

        - The returned dictionary only includes entries for file types that have
          been specified in the checkpoint configuration. If a file type is set to
          `None`, it will not appear in the returned dictionary.

        """

        idx = -1  # Default index to the last element if no event is recorded
        if self.event_name is not None and net.history:
            # Iterate backwards through history to find the last true `event_name`
            for i, v in enumerate(reversed(net.history[:, self.event_name])):
                if v:
                    idx = len(net.history) - i - 1
                    break

        return {key: self._format_target(net, val, idx) for key, val
                in self._f_kwargs().items()}

    def _save_params(self, f, net, f_name, log_name):
        """
        Save the parameters of the neural network to a file.

        This function delegates to `net.save_params` to actually save the parameters
        to the file system. It handles any exceptions that might occur during the
        save operation and logs an appropriate message.

        Parameters
        ----------
        f : str
            The path to the file where parameters are to be saved.
        net : nn.Module or PyroModule
            The neural network instance whose parameters are being saved.
        f_name : str
            The attribute name in `net` that holds the file handle or path. It is
            passed to `net.save_params` as a keyword argument.
        log_name : str
            Human-readable identifier for the type of parameters being saved, e.g.,
            'model parameters', 'optimizer state'. This is used in the error message
            if saving fails.

        Raises
        ------
        Exception
            A broad exception is caught if anything goes wrong during the save process.
            The exception details are logged, and the program continues.

        Examples
        --------
        >>> checkpoint = Checkpoint(dirname='checkpoints', f_params='model.pt')
        >>> checkpoint._save_params('checkpoints/model.pt', net, 'f_params', 'model parameters')
        # Saves the model parameters to 'checkpoints/model.pt'

        Notes
        -----
        - It is important to note that this method is intended to be private and should
        only be used within the context of the class that defines it.
        - The 'use_safetensors' flag is also included when calling `net.save_params`
        which is an attribute of the Checkpoint instance.
        """
        try:
            net.save_params(**{f_name: f, 'use_safetensors': self.use_safetensors})
        except Exception as e:  # pylint: disable=broad-except
            self._sink(
                "Unable to save {} to {}, {}: {}".format(
                    log_name, f, type(e).__name__, e), net.verbose)

    def _format_target(self, net, f, idx):
        """
        Apply formatting to the target filename template.

        The method takes a filename template with placeholders and replaces them
        with actual runtime values such as the last epoch and last batch indices.
        If `f` is a string, it will prepend `self.fn_prefix` to `f` and apply
        formatting. The formatted string is then joined with `self.dirname` to
        create a full file path. If `f` is not a string, it is returned as is.

        Parameters
        ----------
        net : nn.Module or PyroModule
            The neural network object, which provides the history to fill the
            placeholders in the filename template.
        f : str or file-like object
            The filename template as a string with placeholders or a file-like
            object. If it is a string, formatting is applied.
        idx : int
            The index to use for fetching the last epoch and batch information from
            `net.history`. If it's -1, it fetches the latest epoch and batch data.

        Returns
        -------
        str or file-like object
            The formatted filename as a string if `f` was a string, or the
            original `f` if it was a file-like object.

        Examples
        --------
        >>> checkpoint = Checkpoint(dirname='checkpoints', fn_prefix='run_')
        >>> formatted_filename = checkpoint._format_target(net, 'epoch_{last_epoch[epoch]}.pt', -1)
        # If the last entry in net.history for 'epoch' is 10, the result will be
        # 'checkpoints/run_epoch_10.pt'

        Notes
        -----
        - The placeholders in the template string `f` should match the keys in
        `net.history`.
        - This method is typically used internally within the Checkpoint class to
        generate filenames when saving model states.
        """
        if f is None:
            return None
        if isinstance(f, str):
            f = self.fn_prefix + f.format(
                net=net,
                last_epoch=net.history[idx],
                last_batch=net.history[idx, 'batches', -1],
            )
            return os.path.join(self.dirname, f)
        return f

    def _validate_filenames(self):
        """
        Validates that the filenames provided as `f_*` parameters are compatible with `dirname`.

        This method is part of the initialization process of a checkpointing system,
        ensuring that the `f_*` parameters (which denote file paths for different checkpointing elements)
        are not in conflict with the `dirname` parameter. Specifically, if `dirname` is provided,
        all `f_*` parameters must be strings, as they will be appended to `dirname` to form the full
        file paths. If any `f_*` parameter is set and is not a string while `dirname` is also set,
        a `StockpyException` is raised, indicating that `dirname` should only be used with string
        `f_*` parameters.

        Raises
        ------
        StockpyException
            If `dirname` is provided and any of the `f_*` parameters are truthy (indicating they are set)
            and not strings.

        Examples
        --------
        >>> checkpoint = Checkpoint(dirname='checkpoints', f_params='model.pt', f_optimizer='optimizer.pt')
        >>> checkpoint._validate_filenames()
        # No exception is raised since `f_params` and `f_optimizer` are strings.

        >>> checkpoint = Checkpoint(dirname='checkpoints', f_params=123)
        >>> checkpoint._validate_filenames()
        # Raises StockpyException because `f_params` is not a string.

        Notes
        -----
        - This method is typically called during the initialization of a checkpoint object to
        ensure that the filenames for saving model states are properly configured.
        - The `f_*` parameters represent different file-naming options for the elements of the
        checkpoint such as model parameters (`f_params`), optimizer state (`f_optimizer`), etc.
        """

        _check_f_arguments(self.__class__.__name__, **self._f_kwargs())

        if not self.dirname:
            return

        def _is_truthy_and_not_str(f):
            return f and not isinstance(f, str)

        if any(_is_truthy_and_not_str(val) for val in self._f_kwargs().values()):
            raise StockpyException(
                'dirname can only be used when f_* are strings')

    def _sink(self, text, verbose):
        #  We do not want to be affected by verbosity if sink is not print
        if (self.sink is not print) or verbose:
            self.sink(text)


class EarlyStopping(Callback):
    """
    Callback for stopping the training process early if a specified metric does not improve.

    This callback monitors a given metric each epoch and if the metric does not improve
    by at least the defined threshold within a specified number of epochs, it will stop the training process.
    This technique is often used to prevent overfitting.

    Parameters
    ----------
    monitor : str, optional
        The metric name to be monitored. Typically, this could be a loss or a score.
        The default value is 'valid_loss', which refers to the validation loss.
    patience : int, optional
        The number of epochs with no improvement after which training will be stopped.
        The default is 5, meaning training will continue for 5 more epochs after the last
        observed improvement.
    threshold : float, optional
        The minimum change in the monitored metric to qualify as an improvement.
        The default is 1e-4.
    threshold_mode : str, optional
        The mode for the threshold, either 'rel' for relative change or 'abs' for absolute change.
        The default is 'rel'.
    lower_is_better : bool, optional
        A flag indicating if a lower metric value is better. Default is True.
    sink : callable, optional
        A callable that handles information about early stopping events, such as `print` or a logger.
        The default is the `print` function, which outputs the information to the standard output.
    load_best : bool, optional
        A flag indicating if the weights from the epoch with the best monitored metric should be loaded
        at the end of training. Default is False.

    Attributes
    ----------
    misses_ : int
        The number of epochs since the last improvement.
    dynamic_threshold_ : NoneType or float
        The dynamic threshold calculated based on the best score and the threshold mode. Initially None.

    Notes
    -----
    - The `sink` function is designed to allow greater flexibility in logging the early stopping event.
      It can be set to a logging function to integrate with an existing logging system.
    - To restore the full training state, not just the module weights, use a `Checkpoint` callback in
      conjunction with `load_best=True`.

    Examples
    --------
    >>> from stockpy.callbacks import EarlyStopping
    >>> early_stopping = EarlyStopping(patience=10, threshold=1e-2, threshold_mode='abs')
    >>> net = NeuralNet(classifier, criterion, callbacks=[early_stopping])
    >>> net.fit(X, y)
    # Stops if the validation loss does not improve by at least 0.01 within 10 epochs.
    """

    def __init__(
            self,
            monitor='valid_loss',
            patience=5,
            threshold=1e-4,
            threshold_mode='rel',
            lower_is_better=True,
            sink=print,
            load_best=False,
    ):
        self.monitor = monitor
        self.lower_is_better = lower_is_better
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.misses_ = 0
        self.dynamic_threshold_ = None
        self.sink = sink
        self.load_best = load_best

    def __getstate__(self):
        """
        Customize the object state to be pickled.

        When an instance of `EarlyStopping` is serialized using `pickle`,
        this method is called to determine what state should be pickled.
        The `best_model_weights_` attribute is excluded from serialization to
        avoid duplicating the weights in the pickle file, as they are typically
        saved separately and restored when unpickling.

        Returns
        -------
        state : dict
            The state of the object without the `best_model_weights_` attribute.

        Notes
        -----
        The `__getstate__` method is particularly useful when there is a need to
        reduce the size of the pickle file or to customize what gets serialized.
        For example, model weights can be quite large and are often saved
        separately from the model configuration.

        Examples
        --------
        >>> import pickle
        >>> early_stopping = EarlyStopping()
        >>> serialized = pickle.dumps(early_stopping)  # `best_model_weights_` not included
        >>> deserialized = pickle.loads(serialized)    # Restores without `best_model_weights_`
        """
        
        # Avoids to save the module_ weights twice when pickling
        state = self.__dict__.copy()
        state['best_model_weights_'] = None
        return state

    def on_train_begin(self, net, **kwargs):
        """
        Set up the initial state at the beginning of training.

        This method initializes the conditions for early stopping before the training loop starts.
        It verifies that the threshold mode is valid, resets the count of epochs without improvement
        (`misses_`), sets the initial dynamic threshold for improvement, and initializes the storage
        for the best model weights and the best epoch.

        Parameters
        ----------
        net : nn.Module or PyroModule
            The neural network instance that is being trained.
        **kwargs : dict
            Additional arguments that might be passed to the callback.

        Raises
        ------
        ValueError
            If the `threshold_mode` attribute is not one of the accepted values ('rel' or 'abs').

        Notes
        -----
        This method is automatically called by stockpy during the training process and is not meant to
        be invoked directly by the user.

        The dynamic threshold is set to positive infinity if `lower_is_better` is True, meaning that
        any decrease in the monitored score will be considered an improvement, or negative infinity
        if `lower_is_better` is False, so any increase in the monitored score will be considered
        an improvement.

        Examples
        --------
        >>> early_stopping = EarlyStopping()
        >>> neural_net = stockpy.NeuralNet(
        ...     module=MyModule,
        ...     callbacks=[early_stopping]
        ... )
        >>> neural_net.fit(X, y)  # `on_train_begin` is called internally
        """

        if self.threshold_mode not in ['rel', 'abs']:
            raise ValueError("Invalid threshold mode: '{}'"
                             .format(self.threshold_mode))
        self.misses_ = 0
        self.dynamic_threshold_ = np.inf if self.lower_is_better else -np.inf
        self.best_model_weights_ = None
        self.best_epoch_ = 0

    def on_epoch_end(self, net, **kwargs):
        """
        Check whether the monitored score has improved at the end of an epoch and stop training if it has not improved for a given number of epochs.

        At the end of each epoch, this method compares the current score with the best score so far based on the `monitor` attribute. If the score does not improve for a consecutive number of epochs specified by the `patience` attribute, the training process is stopped.

        Parameters
        ----------
        net : nn.Module or PyroModule
            The neural network instance that is being trained.
        **kwargs : dict
            Additional arguments that might be passed to the callback.

        Raises
        ------
        KeyboardInterrupt
            If the monitored score has not improved for the number of epochs specified by `patience`.

        Notes
        -----
        This method is automatically called by stockpy at the end of each epoch during the training process.

        If `load_best` is True and the score has improved, the current state of the network's parameters 
        is saved so it can potentially be restored later.

        Examples
        --------
        >>> early_stopping = EarlyStopping(patience=3)
        >>> neural_net = stockpy.NeuralNet(
        ...     module=MyModule,
        ...     callbacks=[early_stopping]
        ... )
        >>> try:
        ...     neural_net.fit(X, y)  # `on_epoch_end` is called internally
        ... except KeyboardInterrupt:
        ...     print("Early stopping triggered")
        """
        
        current_score = net.history[-1, self.monitor]
        if not self._is_score_improved(current_score):
            self.misses_ += 1
        else:
            self.misses_ = 0
            self.dynamic_threshold_ = self._calc_new_threshold(current_score)
            self.best_epoch_ = net.history[-1, "epoch"]
            if self.load_best:
                self.best_model_weights_ = deepcopy(net.state_dict())
        if self.misses_ == self.patience:
            if net.verbose:
                self._sink("Stopping since {} has not improved in the last "
                           "{} epochs.".format(self.monitor, self.patience),
                           verbose=net.verbose)
            raise KeyboardInterrupt

    def on_train_end(self, net, **kwargs):
        """
        Optionally restore the state of the model to the best epoch upon training completion.

        This method is called at the end of the training process. If the `load_best` attribute is set to True and the best model's weights were saved during training, this method will restore the model's state to those weights.

        Parameters
        ----------
        net : nn.Module or PyroModule
            The neural network instance that has been trained.
        **kwargs : dict
            Additional arguments that might be passed to the callback.

        Notes
        -----
        This method is automatically called by stockpy at the end of the training process. 
        It is not intended to be called manually.

        Examples
        --------
        >>> early_stopping = EarlyStopping(patience=3, load_best=True)
        >>> neural_net = stockpy.NeuralNet(
        ...     module=MyModule,
        ...     callbacks=[early_stopping]
        ... )
        >>> neural_net.fit(X, y)  # `on_train_end` will be called internally
        """

        if (
            self.load_best and (self.best_epoch_ != net.history[-1, "epoch"])
            and (self.best_model_weights_ is not None)
        ):
            net.load_state_dict(self.best_model_weights_)
            self._sink("Restoring best model from epoch {}.".format(
                self.best_epoch_
            ), verbose=net.verbose)

    def _is_score_improved(self, score):
        """
        Check if the current score is an improvement over the threshold.

        This method compares the current score to the dynamic threshold,
        which is set to the best score observed so far during training. The
        definition of "improvement" depends on whether lower scores are
        considered better (`lower_is_better=True`) or not.

        Parameters
        ----------
        score : float
            The current score to compare against the dynamic threshold.

        Returns
        -------
        bool
            True if the score is considered an improvement, False otherwise.

        Notes
        -----
        This method is intended for internal use within the EarlyStopping
        callback to decide whether to continue training or stop early. It is
        not part of the public API.

        Examples
        --------
        >>> early_stopping = EarlyStopping(lower_is_better=True)
        >>> improvement = early_stopping._is_score_improved(0.1)
        >>> print(improvement)
        True
        """

        if self.lower_is_better:
            return score < self.dynamic_threshold_
        return score > self.dynamic_threshold_

    def _calc_new_threshold(self, score):
        """
        Calculate a new threshold for improvement based on the current score.

        If the `threshold_mode` is 'rel' (relative), the new threshold is
        calculated as a fraction of the current score defined by the `threshold`
        attribute. If 'abs' (absolute), the `threshold` itself is used as the
        change required for improvement.

        Parameters
        ----------
        score : float
            The score for the current epoch.

        Returns
        -------
        float
            The newly calculated threshold that the next epoch's score must
            surpass for it to be considered as an improvement.

        Examples
        --------
        >>> early_stopping = EarlyStopping(threshold=0.01, threshold_mode='rel', lower_is_better=True)
        >>> new_threshold = early_stopping._calc_new_threshold(0.1)
        >>> print(new_threshold)
        0.099
        """

        if self.threshold_mode == 'rel':
            abs_threshold_change = self.threshold * score
        else:
            abs_threshold_change = self.threshold

        if self.lower_is_better:
            new_threshold = score - abs_threshold_change
        else:
            new_threshold = score + abs_threshold_change
        return new_threshold

    def _sink(self, text, verbose):
        """
        Send the given text to the sink function if verbosity allows.

        This method checks if the sink is the print function or not. If it is not,
        verbosity is ignored, and the text is always sent to the sink. If the sink
        is print, then the text is only sent to the sink if verbose is True.

        Parameters
        ----------
        text : str
            The text to be sent to the sink.
        verbose : bool
            If True and if the sink is the print function, the text will be printed.
            If False, nothing happens unless the sink is not the print function.

        Notes
        -----
        The `_sink` method is intended for internal use within the EarlyStopping
        class to handle conditional output based on the verbosity level. Users
        should not need to call this method directly.

        Examples
        --------
        >>> early_stopping = EarlyStopping()
        >>> early_stopping._sink("Training will stop if the score does not improve.", verbose=True)
        Training will stop if the score does not improve.
        """

        #  We do not want to be affected by verbosity if sink is not print
        if (self.sink is not print) or verbose:
            self.sink(text)