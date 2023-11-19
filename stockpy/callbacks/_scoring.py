""" Callbacks for calculating scores."""

from contextlib import contextmanager
from contextlib import suppress
from functools import partial
import warnings

import numpy as np
import sklearn
from sklearn.metrics import make_scorer, check_scoring

from stockpy.callbacks import Callback
from stockpy.preprocessing import unpack_data
from sklearn.metrics._scorer import _BaseScorer
from stockpy.utils import data_from_dataset
from stockpy.utils import is_stockpy_dataset
from stockpy.utils import to_numpy
from stockpy.utils import check_indexing
from stockpy.utils import to_device


__all__ = ['PassthroughScoring', 'EpochScoring', 'BatchScoring']


@contextmanager
def cache_net_infer(net, use_caching, y_preds):
    """
    Context manager for caching predictions in stockpy.BaseEstimator.

    This function is deprecated and is scheduled to be removed in a future release.
    When caching is enabled, it modifies the net's `infer` method to return
    cached predictions instead of performing actual inference. Once the context
    is exited, the original `infer` method is restored.

    Parameters
    ----------
    net : stockpy.BaseEstimator
        The neural network instance whose `infer` method is to be temporarily overridden.
    use_caching : bool
        If True, caching is used and the `infer` method is overridden. If False,
        the neural net is yielded unchanged.
    y_preds : iterator
        An iterator over cached predictions that will be returned by the temporarily
        overridden `infer` method.

    Yields
    ------
    stockpy.BaseEstimator
        The neural net with the `infer` method overridden if caching is enabled,
        or the unchanged net if caching is not enabled.


    Notes
    -----
    - The purpose of caching is to avoid redundant computations during inference,
    which can speed up the training process when predictions are needed multiple times.
    - It is crucial to restore the original `infer` method to maintain the expected
    behavior of the neural net outside of the context where caching is relevant.
    - Users are warned about the deprecation through a `DeprecationWarning`.

    """

    # Issue deprecation warning
    warnings.warn(
        "cache_net_infer is no longer used to provide caching for "
        "the scoring callbacks and will hence be removed in a "
        "future release.",
        DeprecationWarning,
    )

    # Proceed without caching if not enabled
    if not use_caching:
        yield net
        return

    # Replace the `infer` method to return cached predictions
    y_preds = iter(y_preds)
    net.infer = lambda *a, **kw: next(y_preds)

    try:
        yield net  # Yield the net with modified infer for use within the context
    finally:
        # Restore the original `infer` method
        # This step is crucial for cleaning up after using the context
        del net.__dict__['infer']  # Remove the temporary attribute


@contextmanager
def _cache_net_forward_iter(net, use_caching, y_preds):
    """
    Context manager for caching the output of forward passes.

    Inside the context, the `forward_iter` method of the net is overridden to yield
    cached predictions. This can be beneficial when the forward pass needs to be
    computed multiple times with the same input. When exiting the context, the
    original `forward_iter` method is reinstated.

    Parameters
    ----------
    net : nn.Module or PyroModule
        The neural network instance whose `forward_iter` method is to be temporarily overridden.
    use_caching : bool
        Flag to enable or disable caching. If 'auto', it is determined by `net.use_caching`.
    y_preds : iterable
        An iterable of cached predictions to be yielded by the overridden `forward_iter` method.

    Yields
    ------
    nn.Module or PyroModule
        The neural net with the `forward_iter` method overridden to provide cached outputs,
        or the original net if caching is not enabled.

    Notes
    -----
    - The caching behavior can be controlled by the `use_caching` parameter of the stockpy.BaseEstimator
      if it is set to 'auto', the decision is made based on the net's own caching policy.
    - This context manager should be used carefully, as it assumes that the cached predictions
      correspond to the inputs that would have been processed by `forward_iter`.
    """

    # Determine caching policy based on net's settings if 'auto'
    if net.use_caching != 'auto':
        use_caching = net.use_caching

    # If not using caching, yield the net unchanged
    if not use_caching:
        yield net
        return

    # Create an iterator from the cached predictions
    y_preds = iter(y_preds)

    # Define a cached version of the forward_iter method
    def cached_forward_iter(*args, device=net.device, **kwargs):
        # Yield cached predictions, ensuring they are moved to the appropriate device
        for yp in y_preds:
            yield to_device(yp, device=device)

    # Override the forward_iter method with the cached version
    net.forward_iter = cached_forward_iter

    try:
        yield net  # Yield the net with the overridden method for use within the context
    finally:
        # Restore the original forward_iter method after exiting the context
        del net.__dict__['forward_iter']  # Remove the temporary attribute


def convert_sklearn_metric_function(scoring):
    """
    Converts a scikit-learn metric function to a scorer object if applicable.

    This function checks if the provided `scoring` argument is a callable metric function from scikit-learn
    (not already a scorer object). If it is, the function is converted to a scorer object using `make_scorer`.
    If `scoring` is already a scorer or does not meet the criteria for conversion, it is returned as is.

    Parameters
    ----------
    scoring : callable or any
        A scikit-learn metric function or any other valid scorer object/definition.

    Returns
    -------
    callable
        A scikit-learn scorer object if `scoring` was a metric function, otherwise `scoring` unchanged.

    Examples
    --------
    >>> from sklearn.metrics import accuracy_score
    >>> scorer = convert_sklearn_metric_function(accuracy_score)
    >>> print(type(scorer))
    <class 'sklearn.metrics._scorer._PredictScorer'>

    Notes
    -----
    - The function uses the `scoring` object's module attribute to check whether it is from `sklearn.metrics`.
    - It ensures that `scoring` is not already a scorer by excluding classes that are known scorer names.
    """

    # Check if scoring is a callable, i.e., a function
    if callable(scoring):
        # Retrieve the module where the function is defined
        module = getattr(scoring, '__module__', None)

        # Define known scorer class names to exclude
        scorer_names = ('_PredictScorer', '_ProbaScorer', '_ThresholdScorer')

        # Check if the scoring function is from sklearn.metrics and not already a known scorer
        if (
            hasattr(module, 'startswith') and
            module.startswith('sklearn.metrics.') and
            not module.startswith('sklearn.metrics.scorer') and
            not module.startswith('sklearn.metrics.tests.') and
            not scoring.__class__.__name__ in scorer_names
        ):
            # Convert to a scorer object
            return make_scorer(scoring)

    # Return the scoring unchanged if it does not meet conversion criteria
    return scoring


class ScoringBase(Callback):
    """
    Abstract base class for implementing scoring callbacks.

    This class serves as a template for creating custom scoring callbacks. To use it,
    one should subclass it and implement the appropriate `on_*` method that corresponds
    to the desired callback trigger (e.g., `on_epoch_end`).

    Parameters
    ----------
    scoring : callable or str
        A callable that takes `y_true` and `y_pred` as inputs and returns a float score;
        or a string representing a predefined score name.
    lower_is_better : bool, optional
        Whether a lower score value represents a better score. If `True`, the best score
        is the minimum score; if `False`, the best score is the maximum score.
        Default is `True`.
    on_train : bool, optional
        Whether to compute the score on the training set. If `False`, the score is computed
        on the validation set. Default is `False`.
    name : str, optional
        The name of the scoring callback. If `None`, a name will be inferred from the `scoring`
        argument. Default is `None`.
    target_extractor : callable, optional
        A function that extracts the target (`y`) from the dataset passed to the scoring callback.
        Default is `to_numpy`, which attempts to convert the target to a NumPy array.
    use_caching : bool, optional
        Indicates whether to use caching for the predictions. Caching can be beneficial when
        the same predictions are used multiple times. Default is `True`.

    Attributes
    ----------
    scoring_ : callable
        The callable scoring function after potentially being wrapped or processed from the
        original `scoring` argument.
    lower_is_better_ : bool
        A definitive boolean indicating whether a lower score indicates a better result.
    on_train_ : bool
        A definitive boolean indicating whether scoring is conducted on the training set.
    name_ : str
        The definitive name of the scoring callback.
    target_extractor_ : callable
        The callable function for extracting targets from the dataset.
    use_caching_ : bool
        A definitive boolean indicating whether caching is used for predictions.

    Examples
    --------
    >>> class AccuracyScoring(ScoringBase):
    ...     def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
    ...         # Assuming self.on_train is False
    ...         y_true = [y for _, y in dataset_valid]
    ...         y_pred = net.predict(dataset_valid)
    ...         score = accuracy_score(y_true, y_pred)
    ...         print(f'Validation Accuracy: {score}')
    ...
    >>> net = NeuralNetClassifier(
    ...     module=MyModule,
    ...     callbacks=[AccuracyScoring(scoring='accuracy')],
    ... )

    Notes
    -----
    - This base class does not implement any `on_*` methods and therefore should not be
    instantiated directly.
    - The scoring function can be a predefined string name as accepted by scikit-learn's
    scoring parameter or a callable with the signature `scorer(y_true, y_pred)`.
    """

    def __init__(
            self,
            scoring,
            lower_is_better=True,
            on_train=False,
            name=None,
            target_extractor=to_numpy,
            use_caching=True,
    ):
        self.scoring = scoring
        self.lower_is_better = lower_is_better
        self.on_train = on_train
        self.name = name
        self.target_extractor = target_extractor
        self.use_caching = use_caching

    def _get_name(self):
        """
        Determine the name of the scoring function to be used for display purposes.

        This method tries to infer the name of the scoring function based on its type.
        If the scoring function was given as a string or has an attribute `__name__`,
        it will use that. For partially applied functions, it will try to retrieve
        the function's original name. If a custom scorer object from scikit-learn is used,
        it will delve into its properties to find the underlying function's name.

        Returns
        -------
        str
            The name of the scoring function. If the name cannot be determined, it defaults to 'score'.

        Raises
        ------
        ValueError
            If the scoring attribute is a dictionary, which is not supported for single-metric scoring.

        Notes
        -----
        - If `self.name` is explicitly provided, it will be used regardless of the actual scoring function.
        - The method does not support dictionaries as it expects a single scoring metric, not multi-metric scoring.
        """

        if self.name is not None:
            return self.name
        if self.scoring_ is None:
            return 'score'
        if isinstance(self.scoring_, str):
            return self.scoring_
        if isinstance(self.scoring_, partial):
            return self.scoring_.func.__name__
        if isinstance(self.scoring_, _BaseScorer):
            if hasattr(self.scoring_._score_func, '__name__'):
                # sklearn < 0.22
                return self.scoring_._score_func.__name__
            # sklearn >= 0.22
            return self.scoring_._score_func._score_func.__name__
        if isinstance(self.scoring_, dict):
            raise ValueError("Dict not supported as scorer for multi-metric scoring."
                             " Register multiple scoring callbacks instead.")
        return self.scoring_.__name__

    def initialize(self):
        """
        Initialize the scoring callback.

        This method sets up the initial best score, converts the scoring metric function
        to a scikit-learn scorer if necessary, and determines the name of the scoring function.

        The initial best score is set to positive infinity if a lower score is better, or
        negative infinity if a higher score is better. This allows for comparison with the
        first computed score during training.

        The scoring function provided is also converted using the `convert_sklearn_metric_function`
        to ensure compatibility with scikit-learn's scorer API.

        Returns
        -------
        self
            Returns an instance of itself.

        Notes
        -----
        - The initialization should be called before using the scoring callback within
        a training loop or evaluation process.
        """
        # Initialize the best score based on the preferred direction (higher or lower).
        self.best_score_ = np.inf if self.lower_is_better else -np.inf

        # Convert the scoring function to a scikit-learn scorer object if it is not already.
        self.scoring_ = convert_sklearn_metric_function(self.scoring)

        # Determine the name of the scoring function for display purposes.
        self.name_ = self._get_name()

        # Return the instance itself to allow for method chaining or simple initialization.
        return self

    def on_train_begin(self, net, X, y, **kwargs):
        """
        Prepare the scoring callback at the beginning of training.

        This method is called at the start of training to prepare the indexing for input
        data `X` and target data `y`. It also initializes the best score from the history
        if such a score exists.

        Parameters
        ----------
        net : nn.Module or PyroModule
            The neural network model.
        X : array-like
            The input data used for training the model.
        y : array-like
            The true labels/targets for the input data.
        **kwargs : dict
            Additional arguments passed to the callback.

        Notes
        -----
        - This method is internally called by the net during the `fit` process.
        - Users should not need to call this method directly.
        """
        # Determine if the input `X` and target `y` support indexing
        # which is required for scoring.
        self.X_indexing_ = check_indexing(X)
        self.y_indexing_ = check_indexing(y)

        # Attempt to find the index of the best score from the net's history.
        # The best score is indicated by a `*_best` suffix in the history keys.
        try:
            # Extract the history column that corresponds to the best scores.
            best_name_history = net.history[:, '{}_best'.format(self.name_)]
            
            # Find the last (rightmost) index where the best score flag is True.
            idx_best_reverse = best_name_history[::-1].index(True)
            
            # Calculate the actual index in normal order.
            idx_best = len(best_name_history) - idx_best_reverse - 1
            
            # Update the best score with the value from the history at the best index.
            self.best_score_ = net.history[idx_best, self.name_]
        except (ValueError, IndexError, KeyError):
            # Suppress any errors encountered during the search for the best score.
            # This allows the training to proceed even if a best score is not found.
            pass

    def _scoring(self, net, X_test, y_test):
        """
        Apply the scoring function to the test data.

        This method uses the provided scoring function to evaluate the performance
        of the neural network on the test data. If predictions have been cached,
        it will use these instead of running the inference again.

        Parameters
        ----------
        net : nn.Module or PyroModule
            The neural network model.
        X_test : array-like
            The features of the test dataset.
        y_test : array-like
            The true labels of the test dataset.

        Returns
        -------
        float
            The score calculated using the scoring function.

        Notes
        -----
        - This method should not be called directly by users. It is intended to be
        called internally by other methods within the scoring callback class.
        """

        # Check and get the scoring function compatible with the Stockpy framework
        # and the user's defined scoring attribute.
        scorer = check_scoring(net, self.scoring_)

        # Execute the scoring function using the network and test dataset.
        return scorer(net, X_test, y_test)

    def _is_best_score(self, current_score):
        """
        Determine if the current score is the best score so far.

        Depending on whether lower or higher scores are better, this method
        compares the current score with the best score observed in previous
        epochs.

        Parameters
        ----------
        current_score : float
            The score obtained in the current epoch.

        Returns
        -------
        bool or None
            Returns True if the current score is the best score, False otherwise.
            If `lower_is_better` is None, returns None, meaning the comparison
            is not applicable.

        Notes
        -----
        - This method is used internally to determine if the current epoch's
        performance is the best across all epochs so far.
        """

        # If `lower_is_better` is not defined, then comparison is not applicable.
        if self.lower_is_better is None:
            return None

        # Compare the current score with the best score according to the
        # `lower_is_better` flag and return the result.
        if self.lower_is_better:
            return current_score < self.best_score_
        return current_score > self.best_score_
    
class BatchScoring(ScoringBase):
    """
    Callback for computing scores after each batch during training or validation.

    This callback computes a score after each batch and stores the results in the history of the
    neural network. Scores for individual batches as well as the average score over an epoch are recorded.
    It also keeps track of the best average score across epochs.

    Compared to `EpochScoring`, which computes scores after each epoch, `BatchScoring` does this for each batch.
    Note that certain metrics (e.g., AUC) may not be accurate when computed on small batches and `EpochScoring`
    may be preferred in those cases.

    Parameters
    ----------
    scoring : None, str, or callable
        The scoring method to use.
        - If None, the model's own `score` method is used.
        - If a string, it must correspond to a valid scikit-learn metric, e.g., "f1_score" or "accuracy_score".
        - If a callable, it must accept three arguments (model, X, y) and return a scalar score.
    lower_is_better : bool, optional (default=True)
        Indicates if a lower score is better (True) or if a higher score is better (False).
    on_train : bool, optional (default=False)
        If True, the scoring is performed on training data, otherwise on validation data.
    name : str or None, optional (default=None)
        The name to be used for the score in the net's history. If None, the name is inferred from the `scoring` argument.
    target_extractor : callable, optional (default=to_numpy)
        A function that extracts the target variable (y) from the dataset. The default is `to_numpy`, which converts y to a NumPy array.
    use_caching : bool, optional (default=True)
        Determines whether to use caching of the model's predictions to compute the score. If False, an additional inference step
        is performed for each batch. The net may override this setting.

    Notes
    -----
    - For metrics that are not well-defined on a per-batch basis (e.g., AUC), prefer using `EpochScoring`.
    - The batch size can affect the computed score; it is recommended to consider the characteristics of the metric when using `BatchScoring`.
    """

    def on_batch_end(self, net, batch, training, **kwargs):
        """
        Compute and record the score at the end of each batch.

        This method is executed at the end of each batch during the training or validation phase, depending
        on the `on_train` attribute of the instance. If the current phase (training or validation) matches
        the `on_train` setting, it proceeds to compute the score using the specified scoring function.

        The method handles the caching of the forward step, if enabled, to avoid redundant computations.

        Parameters
        ----------
        net : nn.Module or PyroModule
            The neural network instance.
        batch : tuple of torch.Tensor
            A tuple containing the data and target tensors. The structure is (data_tensor, target_tensor).
        training : bool
            Flag indicating whether the callback is being called during the training phase.
        **kwargs : dict
            Additional keyword arguments. It is expected to contain the key 'y_pred' which holds the prediction
            tensor. Additional items may be included by stockpy or other callbacks.

        Raises
        ------
        Exception
            Any exception raised by the scoring function will be raised, except for KeyError which is silently ignored.

        Notes
        -----
        - If `training` is not equal to `on_train`, the method returns immediately without computing the score.
        - The method unpacks the batch into data (`X`) and targets (`y`).
        - If `use_caching` is True and caching is available, predictions are retrieved from the cache.
        - If `y` is None, it assumes that the scoring function can handle `y=None`.
        - If the scoring function raises a KeyError (e.g., due to a missing metric), it passes silently.
        """
        
        if training != self.on_train:
            return

        X, y = unpack_data(batch)
        y_preds = [kwargs['y_pred']]
        with _cache_net_forward_iter(net, self.use_caching, y_preds) as cached_net:
            # In case of y=None we will not have gathered any samples.
            # We expect the scoring function to deal with y=None.
            y = None if y is None else self.target_extractor(y)
            try:
                score = self._scoring(cached_net, X, y)
                cached_net.history.record_batch(self.name_, score)
            except KeyError:
                pass

    def get_avg_score(self, history):
        """
        Calculate the weighted average of scores for the latest epoch.

        This method computes the average score across all batches in the most recent epoch.
        It uses the batch sizes as weights to calculate a weighted average, ensuring that
        the average score accurately reflects the distribution of batch sizes.

        Parameters
        ----------
        history : stockpy.history.History
            The history object from a stockpy estimator containing the recorded scores and batch sizes.

        Returns
        -------
        float
            The weighted average score across all batches for the latest epoch.

        Raises
        ------
        ValueError
            If the `history` object does not contain the required information (batch sizes and scores), 
            a `ValueError` may be raised due to invalid indexing or missing keys.

        See Also
        --------
        stockpy.history.History : The history object containing training and validation scores.

        Notes
        -----
        - The method determines the type of batch size (training or validation) based on the `on_train` attribute.
        - It collects the scores and corresponding batch sizes for each batch in the last epoch.
        - The method calculates the weighted average using NumPy's `np.average` function with batch sizes as weights.
        """
        if self.on_train:
            bs_key = 'train_batch_size'
        else:
            bs_key = 'valid_batch_size'

        weights, scores = list(zip(
            *history[-1, 'batches', :, [bs_key, self.name_]]))
        score_avg = np.average(scores, weights=weights)
        return score_avg

    def on_epoch_end(self, net, **kwargs):
        """
        Callback to be executed at the end of each epoch.

        This method calculates the average score for the epoch and updates the history with the
        average score and a flag indicating if this is the best score so far. If no valid score is
        present, the method exits without raising an error.

        Parameters
        ----------
        net : stockpy.base.BaseEstimator
            The neural network model instance.
        **kwargs : dict
            Additional keyword arguments not used in this callback.

        Raises
        ------
        KeyError
            If the batches for the last epoch do not contain the scores identified by `self.name_`,
            the method catches a `KeyError` and exits without updating the history.

        See Also
        --------
        stockpy.callbacks.Callback : Base class for stockpy callbacks.

        Notes
        -----
        - The method attempts to calculate the average score using the `get_avg_score` method.
        - The `_is_best_score` method is used to determine if the current average score is better
          than all previous scores for the corresponding task (training or validation).
        - Updates the model's history with the average score for the current epoch.
        - If the current score is the best, updates the history with this fact.
        - The method is designed to handle the absence of valid data gracefully by using a try-except block.
        - It is annotated with `# pylint: disable=unused-argument` to prevent linting issues with unused arguments.

        """
        history = net.history
        try:  # don't raise if there is no valid data
            history[-1, 'batches', :, self.name_]
        except KeyError:
            return

        score_avg = self.get_avg_score(history)
        is_best = self._is_best_score(score_avg)
        if is_best:
            self.best_score_ = score_avg

        history.record(self.name_, score_avg)
        if is_best is not None:
            history.record(self.name_ + '_best', bool(is_best))

class EpochScoring(ScoringBase):
    """
    Callback for performing scoring after each epoch.

    This callback predicts on train or validation data at the end of each epoch,
    calculates the score, checks if it's the best score so far, and stores the
    results in the history of the neural network.

    Parameters
    ----------
    scoring : None, str, or callable, optional
        The metric for evaluating the predictions:
        - If None (default), the `score` method of the model is used.
        - If a string, it should be a scorer name known to scikit-learn (e.g., "f1", "accuracy").
        - If a callable, should conform to the signature `(model, X, y)` and return a scalar score.
    lower_is_better : bool, optional
        Indicates if a lower score is better (default is True).
    on_train : bool, optional
        Determines whether scoring is done on training data instead of validation (default is False).
    name : str, optional
        The name for the scoring callback. If None, the name is inferred from the `scoring` argument.
    target_extractor : callable, optional
        A function that extracts targets from the `y` argument (default is `to_numpy`).
    use_caching : bool, optional
        Whether to cache predictions and targets over an epoch (default is True). Disabling caching
        necessitates additional inference steps each epoch and restricts input dataset types.

    Examples
    --------
    Using a predefined scoring function:

    >>> net = NeuralNetClassifier(callbacks=[EpochScoring('f1')])

    Using a custom scoring function:

    >>> def custom_score(net, X, y):
    ...     # Your custom scoring logic here
    ...     return score_value
    >>> net = NeuralNetClassifier(callbacks=[EpochScoring(custom_score)])

    Notes
    -----
    - The callback supports custom datasets provided they are compatible with caching if enabled.
    - Caching collects `y_true` and `y_pred` during an epoch to avoid additional inference steps.
    - Custom scoring functions must handle datasets directly if `use_caching` is disabled.

    See Also
    --------
    ScoringBase : The base class for scoring callbacks.
    """

    def _initialize_cache(self):
        """
        Initializes the cache for storing true labels and predicted labels.

        This method resets the `y_trues_` and `y_preds_` lists to empty, preparing
        the cache for the upcoming epoch.
        """
        self.y_trues_ = []
        self.y_preds_ = []

    def initialize(self):
        """
        Initializes the callback before starting the training.

        This method calls the `initialize` method of the superclass to set up the
        scoring callback and then initializes the cache for storing the predictions
        and true labels.

        """
        super().initialize()
        self._initialize_cache()
        return self

    def on_epoch_begin(self, net, dataset_train, dataset_valid, **kwargs):
        """
        Callback method at the beginning of each epoch.

        This method is called before each epoch begins. It resets the cache by
        initializing it, ensuring that the predictions and true labels are stored
        for the current epoch only.

        Parameters
        ----------
        net : nn.Module or PyroModule
            The neural network instance.
        dataset_train : torch.utils.data.Dataset
            The training dataset used for the epoch.
        dataset_valid : torch.utils.data.Dataset
            The validation dataset used for the epoch.
        **kwargs : dict
            Additional arguments passed to the callback.
        """
        self._initialize_cache()

    def on_batch_end(self, net, batch, y_pred, training, **kwargs):
        """
        Callback method at the end of each batch processing.

        This method is called after processing a batch. If caching is enabled and the
        batch corresponds to the appropriate training or validation phase as indicated
        by the `on_train` attribute, it collects references to the prediction and target
        data without copying them, which is efficient in terms of memory.

        This shared reference approach ensures that all instances of Scoring callbacks
        use the same data, which is why the target data is not immediately extracted
        here; this avoids keeping unnecessary copies of data during training and is
        done at epoch end instead.

        Parameters
        ----------
        net : nn.Module or PyroModule
            The neural network instance.
        batch : tuple of torch.Tensor
            The input data batch. It is a tuple containing the data and targets.
        y_pred : torch.Tensor
            The predictions made by the net for the current batch.
        training : bool
            Flag indicating whether the callback is called during training or validation.
        **kwargs : dict
            Additional arguments passed to the callback.
        """
        use_caching = self.use_caching
        if net.use_caching !=  'auto':
            use_caching = net.use_caching

        if (not use_caching) or (training != self.on_train):
            return

        _X, y = unpack_data(batch)
        if y is not None:
            self.y_trues_.append(y)
        self.y_preds_.append(y_pred)

    def get_test_data(self, dataset_train, dataset_valid, use_caching):
        """
        Acquire the data required for evaluating the scoring function.

        This method simplifies the process of obtaining the appropriate
        dataset for scoring by handling the selection between training and
        validation datasets, processing different types of input data,
        and managing the caching of predictions.

        Parameters
        ----------
        dataset_train : stockpy.StockpyDataset or similar
            The training dataset or an object that can be used to generate
            training data batches. This is not necessarily a stockpy Dataset
            but must be compatible with stockpy's expectations of a dataset.
        dataset_valid : stockpy.Dataset or similar
            The validation dataset or an object that can be used to generate
            validation data batches, with requirements similar to `dataset_train`.
        use_caching : bool
            A flag indicating whether the caching of inference predictions
            is being utilized.

        Returns
        -------
        X_test : array-like, Dataset, or None
            The input data on which the predictions were or will be made. If
            `use_caching` is True, `X_test` will be the dataset provided.
        y_test : array-like or None
            The true labels against which the predictions will be scored. If
            `use_caching` is True, `y_test` will be an array of cached true labels.
        y_pred : list of array-like
            A list containing batches of predictions. If `use_caching` is False,
            `y_pred` will be an empty list. When `use_caching` is True, it will
            be necessary to concatenate these batches before using them further,
            which can be done via `y_pred = np.concatenate(y_pred)`.

        Notes
        -----
        - The method internally checks if the provided dataset is a custom dataset
          based on Stockpy and handles it accordingly.
        - If caching is not used, `X_test` and `y_test` are extracted directly
          from the provided dataset.
        - It is expected that the scoring function can handle a `y_test` value of None.
        """
        dataset = dataset_train if self.on_train else dataset_valid

        if use_caching:
            X_test = dataset
            y_pred = self.y_preds_
            y_test = [self.target_extractor(y) for y in self.y_trues_]
            # In case of y=None we will not have gathered any samples.
            # We expect the scoring function to deal with y_test=None.
            y_test = np.concatenate(y_test) if y_test else None
            return X_test, y_test, y_pred

        if is_stockpy_dataset(dataset):
            X_test, y_test = data_from_dataset(
                dataset,
                X_indexing=self.X_indexing_,
                y_indexing=self.y_indexing_,
            )
        else:
            X_test, y_test = dataset, None

        if y_test is not None:
            # We allow y_test to be None but the scoring function has
            # to be able to deal with it (i.e. called without y_test).
            y_test = self.target_extractor(y_test)
        return X_test, y_test, []

    def _record_score(self, history, current_score):
        """
        Log the current score and determine if it is the best score achieved.

        This method updates the history with the current score and a flag indicating
        if the current score is the best one according to the scoring rules defined
        (e.g., higher is better or lower is better). If the current score is the best,
        it updates the best score tracking attribute.

        Parameters
        ----------
        history : stockpy.History
            The history object that logs all the events and scores during training.
        current_score : float
            The score obtained from the scoring callback function for the current epoch or batch.

        Notes
        -----
        - The method relies on the '_is_best_score' helper method to determine if the
        current score is the best.
        - The history is expected to have a record method that logs key-value pairs.
        - If the current score is the best, it is saved in the 'best_score_' attribute.
        - The method handles the scenario where '_is_best_score' returns None, which
        signifies an indeterminate state where the concept of "best" does not apply.
        """

        history.record(self.name_, current_score)

        is_best = self._is_best_score(current_score)
        if is_best is None:
            return

        history.record(self.name_ + '_best', bool(is_best))
        if is_best:
            self.best_score_ = current_score

    def on_epoch_end(
            self,
            net,
            dataset_train,
            dataset_valid,
            **kwargs):
        """
        Compute and record the scoring metric at the end of each epoch.

        This method retrieves the appropriate test data, applies the scoring function,
        and records the resulting score. If caching is enabled and applicable, it uses
        the cached predictions; otherwise, it triggers a new prediction pass.

        Parameters
        ----------
        net : nn.Module or PyroModule
            The neural network instance.
        dataset_train : torch.utils.data.Dataset
            The dataset used for training.
        dataset_valid : torch.utils.data.Dataset
            The dataset used for validation.
        **kwargs : dict
            Additional keyword arguments.

        Notes
        -----
        - The method will not execute if `X_test` is None, indicating that there is no
          data to score against.
        - Caching behavior is determined by the `use_caching` attribute and the `net`'s
          caching policy.
        - The method updates the history with the current score by invoking the
          `_record_score` method.
        - It relies on the `_scoring` method to compute the score and uses a context manager
          `_cache_net_forward_iter` to handle caching if necessary.

        """

        use_caching = self.use_caching
        if net.use_caching !=  'auto':
            use_caching = net.use_caching

        X_test, y_test, y_pred = self.get_test_data(
            dataset_train,
            dataset_valid,
            use_caching=use_caching,
        )
        if X_test is None:
            return

        with _cache_net_forward_iter(net, self.use_caching, y_pred) as cached_net:
            current_score = self._scoring(cached_net, X_test, y_test)

        self._record_score(net.history, current_score)

    def on_train_end(self, *args, **kwargs):
        """
        Clear the prediction and target caches at the end of training.

        This method is a cleanup hook that is called when training finishes,
        ensuring that any data collected during the scoring process is cleared,
        ready for the next training run.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        self._initialize_cache()


class PassthroughScoring(Callback):
    """
    This callback facilitates passing through batch-level scores to the epoch level.

    Rather than computing new scores, it averages the precomputed batch-level scores
    and logs them into the model's history at the epoch level. This is particularly
    useful when batch-level scores are already available and an epoch-level summary
    is desired.

    It's important to use this callback when the score is already available at the
    batch level from previous calculations. If batch-level scores need to be computed,
    consider using the `BatchScoring` callback.

    Parameters
    ----------
    name : str
        The name under which the batch-level score is recorded in the history object.
        This name is used to identify and extract the relevant scores for processing.
    lower_is_better : bool, default=True
        Indicates the desired direction of the score. A `True` value means that lower
        scores indicate better performance (e.g., log loss), while `False` implies that
        higher scores are better (e.g., accuracy).
    on_train : bool, default=False
        Determines whether the averaging should be performed on the training data.
        If `True`, the scores from the training batches are passed through. Otherwise,
        the scores from the validation batches are used.

    Examples
    --------
    >>> net = NeuralNet(
    ...     module=MyModule,
    ...     callbacks=[
    ...         ('batch_acc', BatchScorer(name='accuracy')),
    ...         ('epoch_acc', PassthroughScoring(name='accuracy'))
    ...     ],
    ... )
    >>> net.fit(X, y)
        
    In the example above, `BatchScorer` computes 'accuracy' at the batch level,
    and `PassthroughScoring` passes this score through to the epoch level.
    """

    def __init__(
            self,
            name,
            lower_is_better=True,
            on_train=False,
    ):
        self.name = name
        self.lower_is_better = lower_is_better
        self.on_train = on_train

    def initialize(self):
        """
        Initialize the best score tracking.

        Sets the best score to positive or negative infinity based on the
        direction indicated by `lower_is_better` attribute. This method should
        be called at the beginning of training or evaluation to set the initial
        state for best score tracking.

        Returns
        -------
        self
            Returns an instance of `self`.

        """
        self.best_score_ = np.inf if self.lower_is_better else -np.inf
        return self

    def _is_best_score(self, current_score):
        """
        Check if the current score is the best one so far.

        This method is used internally to determine whether the current score
        obtained from the scoring callback is better than the previously recorded
        best score. The direction of comparison is determined by the `lower_is_better`
        attribute.

        Parameters
        ----------
        current_score : float
            The current score to be compared with the best score.

        Returns
        -------
        bool or None
            Returns `True` if the current score is the best, `False` if not, or
            `None` if `lower_is_better` is not defined.

        Examples
        --------
        >>> scoring = PassthroughScoring(name='accuracy', lower_is_better=False)
        >>> scoring.initialize()
        >>> scoring._is_best_score(0.95)
        True

        Notes
        -----
        This method should be considered private and used internally within the
        class's logic. Users of the class should not call this method directly.
        """

        if self.lower_is_better is None:
            return None
        if self.lower_is_better:
            return current_score < self.best_score_
        return current_score > self.best_score_

    def get_avg_score(self, history):
        """
        Compute the average score from the history over the batches of an epoch.

        This method retrieves the scores and batch sizes from the history for the
        last epoch and calculates the weighted average score across all batches.
        The average is weighted by the batch sizes to account for the potential
        variability in batch sizes throughout the epoch.

        Parameters
        ----------
        history : History
            A History object that records all the events during the training or
            evaluation of the neural network, including batch scores and sizes.

        Returns
        -------
        score_avg : float
            The computed weighted average score for the last epoch.

        Examples
        --------
        >>> history = History()
        >>> # Assume history has been updated during training with batch scores
        >>> scoring = PassthroughScoring(name='accuracy', on_train=True)
        >>> avg_accuracy = scoring.get_avg_score(history)
        >>> print(f"Average training accuracy: {avg_accuracy:.2f}")
        Average training accuracy: 0.95

        Notes
        -----
        This method assumes that the scores are already calculated and stored in
        the history. It only computes the average of these scores and does not
        perform any scoring by itself.
        """
        # Determine the batch size key based on whether we're looking at training or validation data
        bs_key = 'train_batch_size' if self.on_train else 'valid_batch_size'

        # Unpack weights (batch sizes) and scores from the history for the last epoch
        weights, scores = list(zip(*history[-1, 'batches', :, [bs_key, self.name]]))

        # Calculate the weighted average of the scores
        score_avg = np.average(scores, weights=weights)

        return score_avg

    def on_epoch_end(self, net, **kwargs):
        """
        At the end of each epoch, compute and record the average score.

        This method is automatically called at the end of each epoch. It
        retrieves the individual batch scores from the history, computes the
        average score across all batches, and updates the history with the
        average score. It also checks if the computed average score is the best
        one seen so far and updates the history and the `best_score_` attribute
        accordingly.

        Parameters
        ----------
        net : NeuralNet
            The neural network instance.
        **kwargs : dict
            Additional keyword arguments not used by this callback.

        Examples
        --------
        >>> net = NeuralNet(
        ...     module=MyModule,
        ...     callbacks=[PassthroughScoring(name='accuracy')],
        ... )
        >>> net.fit(X_train, y_train)
        # After fitting, the history will have the average accuracy recorded

        Notes
        -----
        - This method does not raise an exception if there are no scores available
          in the history for the current epoch, as this might be the case if no
          valid data is present.
        """

        history = net.history
        try:  # don't raise an exception if there is no score under self.name
            history[-1, 'batches', :, self.name]
        except KeyError:
            return  # exit if the expected score is not found in history

        # Compute the average score for the current epoch
        score_avg = self.get_avg_score(history)

        # Determine if the current score is the best so far
        is_best = self._is_best_score(score_avg)
        if is_best:
            # Update the best score
            self.best_score_ = score_avg

        # Record the average score for the current epoch in the history
        history.record(self.name, score_avg)

        # If we are able to determine whether it's the best score or not (is_best is not None)
        if is_best is not None:
            # Record in the history whether this is the best score so far
            history.record(self.name + '_best', bool(is_best))