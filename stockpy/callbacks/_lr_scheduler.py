import sys


import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer
from stockpy.callbacks import Callback

__all__ = ['LRScheduler', 'WarmRestartLR']


def _check_lr(name, optimizer, lr):
    """
    Ensure a learning rate is provided for each parameter group in the optimizer.

    This function checks the `lr` argument and returns an array of learning rates
    corresponding to each parameter group of the optimizer. If a single learning rate
    is provided, it is replicated for each parameter group. If a sequence of learning
    rates is provided, it verifies that the sequence length matches the number of
    parameter groups.

    Parameters
    ----------
    name : str
        Name of the learning rate in the context where the function is called.
        Used in error messages to indicate which learning rate has issues.
    optimizer : torch.optim.Optimizer
        The PyTorch optimizer instance containing parameter groups.
    lr : float or list/tuple of float
        The learning rate value(s) to be validated. Can be a single float (which is
        applied to all parameter groups) or a list/tuple of floats with a separate
        learning rate for each parameter group.

    Returns
    -------
    numpy.ndarray
        An array of learning rates with one learning rate per parameter group.

    Raises
    ------
    ValueError
        If a list or tuple of learning rates is provided and its length does not match
        the number of parameter groups in the optimizer.

    Examples
    --------
    >>> optimizer = torch.optim.Adam([
    ...     {'params': model.base.parameters()},
    ...     {'params': model.classifier.parameters()},
    ... ])
    >>> lr = _check_lr('base', optimizer, 0.001)
    >>> lr
    array([0.001, 0.001])

    >>> lr = _check_lr('base', optimizer, [0.001, 0.002])
    >>> lr
    array([0.001, 0.002])

    >>> lr = _check_lr('base', optimizer, [0.001])
    ValueError: 1 lr values were passed for base but there are 2 param groups.
    """
    # Ensure we have as many learning rates as there are parameter groups
    n = len(optimizer.param_groups)
    if not isinstance(lr, (list, tuple)):
        # If lr is a single number, create a numpy array with lr repeated for each param group
        return lr * np.ones(n)

    # If a list or tuple of learning rates is provided, check if its length matches the param groups
    if len(lr) != n:
        raise ValueError("{} lr values were passed for {} but there are "
                         "{} param groups.".format(len(lr), name, n))
    return np.array(lr)


class LRScheduler(Callback):
    """
    Schedules the learning rate according to the specified policy.

    This callback adjusts the learning rate for each parameter group
    in the optimizer according to a predefined learning rate scheduling policy.
    The adjustment can be made at the end of each epoch or after each training batch.

    Parameters
    ----------
    policy : str or _LRScheduler class, optional
        The policy to use for learning rate scheduling. Can be a string
        specifying the name of a predefined policy, or an instance of a subclass
        of `_LRScheduler`. Common options include 'StepLR', 'MultiStepLR', and
        custom schedulers. Default is 'WarmRestartLR'.
    monitor : str or callable, optional
        The metric from the training history to monitor, which is used to make
        decisions by the scheduler (e.g., 'val_loss'). Can also be a callable
        that returns a score (float) when passed the net instance. Default is 'train_loss'.
    event_name : str, optional
        The name of the event that is recorded in the history when the learning
        rate scheduler takes a step. If set to `None`, no event will be recorded.
        This feature is only supported in PyTorch version 1.4 or later. Default is 'event_lr'.
    step_every : str, optional
        Determines when the scheduler should take a step. Can be 'epoch' to step
        after each epoch or 'batch' to step after each training batch. Default is 'epoch'.
    kwargs : dict
        Additional keyword arguments passed to the learning rate scheduler's
        constructor.

    Examples
    --------
    >>> from torch.optim.lr_scheduler import StepLR
    >>> from stockpy.callbacks import LRScheduler
    >>> lr_scheduler = LRScheduler(policy=StepLR, step_size=1)
    >>> net = NeuralNet(classifier, criterion, optimizer, lr=0.05,
                        callbacks=[lr_scheduler])

    Notes
    -----
    - The actual policy class must be a subclass of `_LRScheduler` from PyTorch's
      `torch.optim.lr_scheduler` module.
    - The `step_every` parameter is particularly useful when using learning rate
      warm-up or other fine-grained learning rate policies.
    - When using custom scheduler policies, ensure they are compatible with
      the version of PyTorch you are using.
    """

    def __init__(self,
                 policy='WarmRestartLR',
                 monitor='train_loss',
                 event_name="event_lr",
                 step_every='epoch',
                 **kwargs):
        self.policy = policy
        self.monitor = monitor
        self.event_name = event_name
        self.step_every = step_every
        vars(self).update(kwargs)

    def simulate(self, steps, initial_lr):
        """
        Simulate the learning rate schedule over a specified number of steps.

        This method allows for the visualization and analysis of the learning rate
        adjustments that would be made by the scheduler during training, based on an
        initial learning rate. It uses a mock optimizer and parameter group to track
        the changes to the learning rate according to the specified policy.

        Parameters
        ----------
        steps : int
            The number of steps over which to simulate the learning rate schedule. Each
            step represents an iteration where the scheduler would typically update the
            learning rate.
        initial_lr : float
            The initial learning rate to use for the simulation. This value is set for
            the mock optimizer's parameter group before starting the simulation.

        Returns
        -------
        lrs : numpy.ndarray
            An array containing the simulated learning rates for each step. The array
            length is equal to the number of `steps` parameter.

        Examples
        --------
        >>> scheduler = LRScheduler(policy='StepLR', step_size=5, gamma=0.1)
        >>> scheduler.simulate(15, initial_lr=0.05)
        array([0.05, 0.05, 0.05, 0.05, 0.05, 0.005, 0.005, ..., 0.0005])

        Notes
        -----
        - The actual update logic for the learning rate on each step is determined by
          the policy class that is set for the `LRScheduler`.
        - This method does not modify the state of the `LRScheduler` instance or the
          associated neural network model. It is purely for simulation purposes.

        """

        test = torch.ones(1, requires_grad=True)
        opt = torch.optim.SGD([{'params': test, 'lr': initial_lr}])
        policy_cls = self._get_policy_cls()
        sch = policy_cls(opt, **self.kwargs)

        lrs = []
        for _ in range(steps):
            opt.step()  # suppress warning about .step call order
            lrs.append(opt.param_groups[0]['lr'])
            sch.step()

        return np.array(lrs)

    def initialize(self):
        """
        Initializes the learning rate scheduler.

        This method prepares the learning rate scheduler for use by determining the
        policy class based on the `policy` attribute, and initializing internal
        state variables.

        The `policy_` attribute is set by referencing a string name of a policy class
        if `policy` is a string, or by using the policy class directly if it is not
        a string. The scheduler itself (`lr_scheduler_`) and the batch index
        counter (`batch_idx_`) are set to `None` and `0`, respectively.

        Returns
        -------
        self : LRScheduler
            Returns an instance of itself, with the policy class and initial state
            variables set.

        Examples
        --------
        >>> scheduler = LRScheduler(policy='StepLR')
        >>> scheduler.initialize()
        <LRScheduler object with StepLR policy initialized>

        Notes
        -----
        - This method should be called before using the scheduler in the training loop
          to ensure that the policy class is correctly determined and that the scheduler
          is properly initialized.
        - If `policy` is a string, it should be the name of a class present in the
          namespace where the `LRScheduler` class is defined. Otherwise, it should be
          a class that implements a learning rate scheduler.

        """

        self.policy_ = self._get_policy_cls()
        self.lr_scheduler_ = None
        self.batch_idx_ = 0
        return self

    def _get_policy_cls(self):
        """
        Retrieves the learning rate policy class.

        This method determines the class to use as the learning rate policy. If the `policy`
        attribute is a string, the method treats it as the name of the policy class and tries
        to retrieve it from the current module's namespace. If `policy` is not a string,
        it is expected to be the policy class itself or a callable representing the policy.

        Returns
        -------
        class or callable
            The class or callable that represents the learning rate policy.

        Raises
        ------
        AttributeError
            If `policy` is a string and no class with that name exists in the current module's
            namespace.

        Examples
        --------
        >>> scheduler = LRScheduler(policy='StepLR')
        >>> scheduler._get_policy_cls()
        <class 'torch.optim.lr_scheduler.StepLR'>

        >>> policy_callable = lambda optimizer, step_size: StepLR(optimizer, step_size)
        >>> scheduler = LRScheduler(policy=policy_callable)
        >>> scheduler._get_policy_cls()
        <function <lambda> at 0x...>

        Notes
        -----
        - This method is typically called internally by the `initialize` method of the
        `LRScheduler` class.
        - It is assumed that if `policy` is a string, the corresponding policy class has
        been properly imported and is available in the current module's namespace.

        """
        if isinstance(self.policy, str):
            return getattr(sys.modules[__name__], self.policy)
        return self.policy

    @property
    def kwargs(self):
        """
        Retrieves keyword arguments for the scheduler.

        This property filters out the reserved keys from the instance's __dict__ and
        prepares a dictionary of keyword arguments that are applicable to the learning
        rate scheduler's constructor.

        Reserved keys that are excluded from the resulting kwargs dictionary include
        'policy', 'monitor', 'event_name', 'step_every', and any keys that end with
        an underscore.

        Returns
        -------
        dict
            A dictionary of keyword arguments that can be used for initializing the
            learning rate scheduler.

        Examples
        --------
        >>> scheduler = LRScheduler(policy='StepLR', gamma=0.1, step_size=5)
        >>> scheduler.kwargs
        {'gamma': 0.1, 'step_size': 5}

        """
        excluded = ('policy', 'monitor', 'event_name', 'step_every')
        kwargs = {key: val for key, val in vars(self).items()
                  if not (key in excluded or key.endswith('_'))}
        return kwargs

    def on_train_begin(self, net, **kwargs):
        """
        Initialize the learning rate scheduler at the beginning of training.

        This method sets up the learning rate scheduler by retrieving the appropriate
        scheduler based on the provided policy and keyword arguments. If there's
        existing training history, it uses that history to determine the starting
        batch index for the scheduler.

        Parameters
        ----------
        net : stockpy.NeuralNet
            The neural network instance associated with the current training session.
        **kwargs : dict, optional
            Additional keyword arguments not used by this method but may be consumed
            by other callbacks or overridden methods.

        Raises
        ------
        KeyError
            If there's an issue accessing the required history items, a KeyError may
            be raised.

        Notes
        -----
        - The scheduler is only initialized if it hasn't been created previously.
        - The method takes care of resuming the batch index count from the last
          recorded state in the training history, allowing for the continuation of
          training seamlessly.

        """

        if net.history:
            try:
                self.batch_idx_ = sum(net.history[:, 'train_batch_count'])
            except KeyError:
                self.batch_idx_ = sum(len(b) for b in net.history[:, 'batches'])
        self.lr_scheduler_ = self._get_scheduler(
            net, self.policy_, **self.kwargs
        )

    def _step(self, net, lr_scheduler, score=None):
        """
        Step the learning rate scheduler.

        This helper method advances the learning rate scheduler. If using a scheduler
        like ReduceLROnPlateau, which requires a performance score to determine the
        learning rate adjustment, this method will pass that score.

        Additionally, this method ensures compatibility with acceleration tools like
        HuggingFace's Accelerate, which may affect when the scheduler should step
        based on whether an optimizer step was skipped due to gradient accumulation.

        Parameters
        ----------
        net : stockpy.NeuralNet
            The neural network instance that is currently being trained.
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler instance that will be stepped.
        score : float, optional
            The performance score that some schedulers (e.g., ReduceLROnPlateau)
            require for adjusting the learning rate. Default is None, which is
            suitable for schedulers that do not require a score.

        Notes
        -----
        - If the net uses the AccelerateMixin from HuggingFace and the optimizer step
        was skipped due to gradient accumulation, the scheduler step will also be
        skipped to maintain consistency in the training loop.
        - The learning rate scheduler is expected to have a `step` method that can
        optionally take a `score` argument if it's a ReduceLROnPlateau scheduler.

        """
        accelerator_maybe = getattr(net, 'accelerator', None)
        accelerator_step_skipped = (
            accelerator_maybe and accelerator_maybe.optimizer_step_was_skipped
        )
        if accelerator_step_skipped:
            return

        if score is None:
            lr_scheduler.step()
        else:
            lr_scheduler.step(score)

    def on_epoch_end(self, net, **kwargs):
        """
        Adjust the learning rate at the end of each epoch.

        If 'epoch' is specified as the step frequency (`step_every` attribute), this method
        adjusts the learning rate using the learning rate scheduler (`lr_scheduler_` attribute).
        For schedulers that require a performance score (e.g., ReduceLROnPlateau), it retrieves
        the score from the network's history based on the specified metric to monitor (`monitor`
        attribute). If the `monitor` is a callable, it will be called with the net as an argument
        to obtain the score. If the learning rate scheduler has the method `get_last_lr`, it
        records the current learning rate into the network's history under the event name
        (`event_name` attribute).

        Parameters
        ----------
        net : stockpy.NeuralNet
            The neural network being trained.
        **kwargs : dict, optional
            Additional keyword arguments not used by this callback.

        Raises
        ------
        ValueError
            If the specified metric to monitor is not found in the network's history and the
            scheduler is of type ReduceLROnPlateau, a ValueError will be raised indicating
            that a Scoring callback with the specified name should be added before the
            LRScheduler callback.

        Examples
        --------
        >>> # This method is automatically called by the net at the end of each epoch.

        Notes
        -----
        - If `step_every` is not set to 'epoch', the method exits without adjusting the learning rate.
        - If `event_name` is None, no event will be recorded even if the scheduler supports it.
        - For ReduceLROnPlateau, if the score is provided, it will be used to determine the
          learning rate adjustment; otherwise, an error is raised.
        - If the scheduler does not require a score, it will step without additional arguments.

        """

        if self.step_every != 'epoch':
            return
        if isinstance(self.lr_scheduler_, ReduceLROnPlateau):
            if callable(self.monitor):
                score = self.monitor(net)
            else:
                try:
                    score = net.history[-1, self.monitor]
                except KeyError as e:
                    raise ValueError(
                        f"'{self.monitor}' was not found in history. A "
                        f"Scoring callback with name='{self.monitor}' "
                        "should be placed before the LRScheduler callback"
                    ) from e

            self._step(net, self.lr_scheduler_, score=score)
            # ReduceLROnPlateau does not expose the current lr so it can't be recorded
        else:
            if (
                    (self.event_name is not None)
                    and hasattr(self.lr_scheduler_, "get_last_lr")
            ):
                net.history.record(self.event_name, self.lr_scheduler_.get_last_lr()[0])
            self._step(net, self.lr_scheduler_)

    def on_batch_end(self, net, training, **kwargs):
        """
        Adjust the learning rate at the end of each training batch if specified.

        This method steps the learning rate scheduler (`lr_scheduler_` attribute) after each
        training batch if `step_every` is set to 'batch'. It also increments the batch index
        counter (`batch_idx_` attribute) which tracks the number of batches processed. If an
        event name (`event_name` attribute) is provided and the scheduler implements the
        `get_last_lr` method, it records the current learning rate into the network's batch
        history.

        Parameters
        ----------
        net : stockpy.NeuralNet
            The neural network being trained.
        training : bool
            A flag indicating whether the current batch is part of training or validation.
        **kwargs : dict, optional
            Additional keyword arguments not used by this callback.

        Examples
        --------
        >>> # This method is automatically called by the net at the end of each training batch.

        Notes
        -----
        - The learning rate is adjusted only if `training` is True, indicating that the batch is
          part of the training process.
        - If `step_every` is not 'batch', the method exits without making any adjustments.
        - The current learning rate is recorded in the network's history if `event_name` is not None
          and the learning rate scheduler supports retrieving the last learning rate.
        - The batch index is incremented regardless of whether the learning rate was adjusted.
          This counter can be used by other processes that need to track the number of batches.

        """

        if not training or self.step_every != 'batch':
            return
        if (
                (self.event_name is not None)
                and hasattr(self.lr_scheduler_, "get_last_lr")
        ):
            net.history.record_batch(
                self.event_name, self.lr_scheduler_.get_last_lr()[0])
        self._step(net, self.lr_scheduler_)
        self.batch_idx_ += 1

    def _get_scheduler(self, net, policy, **scheduler_kwargs):
        """
        Initialize the learning rate scheduler based on the given policy.

        This method creates a learning rate scheduler object from the policy passed to it.
        If 'last_epoch' is not already in `scheduler_kwargs`, it is set to the number of
        epochs that have already been completed, which is inferred from the length of the
        network's history.

        Parameters
        ----------
        net : nn.Module or PyroModule
            The neural network for which the learning rate scheduler will be set.
        policy : _LRScheduler or subclass
            The learning rate scheduler class to be used. This can be one of the predefined
            scheduler classes from PyTorch or a custom defined scheduler class.
        **scheduler_kwargs : dict
            Arbitrary keyword arguments that are passed to the scheduler's constructor. If
            `last_epoch` is not provided, it is set to the index of the last completed epoch.

        Returns
        -------
        scheduler : _LRScheduler
            An instance of the learning rate scheduler initialized with the provided arguments.

        Raises
        ------
        ValueError
            If the policy is not a recognized learning rate scheduler class.

        Examples
        --------
        >>> # Typically called inside the LRScheduler callback methods
        >>> scheduler = self._get_scheduler(net, policy, **scheduler_kwargs)

        Notes
        -----
        - The method checks if the policy is `ReduceLROnPlateau` since it does not require
        the `last_epoch` argument, which is specific to schedulers that change the learning
        rate at fixed intervals.
        - For non-`ReduceLROnPlateau` policies, if `last_epoch` is missing in `scheduler_kwargs`,
        it is calculated from the network's history length, which corresponds to the number of
        epochs completed so far.
        - The method is a private method intended to be used within the `LRScheduler` callback
        class and not meant to be accessed directly by users.

        """
        if (
                (policy not in [ReduceLROnPlateau])
                and ('last_epoch' not in scheduler_kwargs)
        ):
            last_epoch = len(net.history) - 1
            scheduler_kwargs['last_epoch'] = last_epoch

        return policy(net.optimizer_, **scheduler_kwargs)


class WarmRestartLR(_LRScheduler):
    """
    Implements Stochastic Gradient Descent with Warm Restarts (SGDR).

    This scheduler adjusts the learning rate for each parameter group according
    to the SGDR policy. The learning rate is reset to a maximum value at the
    beginning of each warm restart period and is decreased to a minimum value
    during the course of the period. The periods increase after each restart
    by a factor of `period_mult`.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    min_lr : float or list of floats, optional (default=1e-6)
        A single float to set the minimum learning rate for all parameter groups
        or a list of floats to set the learning rate for each group individually.
    max_lr : float or list of floats, optional (default=0.05)
        A single float to set the maximum learning rate for all parameter groups
        or a list of floats to set the learning rate for each group individually.
    base_period : int, optional (default=10)
        Initial period for the first restart. The period for each subsequent
        restart is calculated as `base_period * (period_mult ** num_restarts)`.
    period_mult : int, optional (default=2)
        Factor by which the period length will be multiplied after each restart.
    last_epoch : int, optional (default=-1)
        The index of the last epoch. This is used when resuming a training job.
        Passing -1 sets the epoch to the start of the training.

    References
    ----------
    .. [1] Loshchilov, I. and Hutter, F. (2017)
        SGDR: Stochastic Gradient Descent with Warm Restarts.
        ICLR 2017. `<https://arxiv.org/abs/1608.03983>`

    Examples
    --------
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = WarmRestartLR(optimizer)
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()

    Notes
    -----
    - `last_epoch` is set to -1 by default, which corresponds to starting the
      training from scratch. When resuming from a checkpoint, set `last_epoch`
      to the last completed epoch index.
    - The learning rate for each parameter group is computed as:
      `lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * current_epoch / period))`
    - Ensure that `min_lr` and `max_lr` are set correctly as per the individual
      learning rate requirements of each parameter group when passed as lists.

    """


    def __init__(
            self, optimizer,
            min_lr=1e-6,
            max_lr=0.05,
            base_period=10,
            period_mult=2,
            last_epoch=-1
    ):
        self.min_lr = _check_lr('min_lr', optimizer, min_lr)
        self.max_lr = _check_lr('max_lr', optimizer, max_lr)
        self.base_period = base_period
        self.period_mult = period_mult
        super(WarmRestartLR, self).__init__(optimizer, last_epoch)

    def _get_current_lr(self, min_lr, max_lr, period, epoch):
        """
        Calculate the learning rate for the current epoch using cosine annealing.

        The learning rate is computed using the formula from SGDR:
        `min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * epoch / period))`

        Parameters
        ----------
        min_lr : float
            Minimum learning rate for the annealing period.
        max_lr : float
            Maximum learning rate for the annealing period.
        period : float
            The period of the cosine annealing cycle.
        epoch : float
            The current epoch within the period.

        Returns
        -------
        float
            The calculated learning rate for the current epoch.

        Notes
        -----
        - The method computes the learning rate for a given epoch within a period
        based on the cosine annealing schedule.
        - The learning rate starts at `max_lr` at the beginning of the period and
        gradually decreases to `min_lr` towards the end of the period.
        - The method is a private method intended to be used within the `WarmRestartLR`
        class and not meant to be accessed directly by users.

        """

        return min_lr + 0.5 * (max_lr - min_lr) * (
            1 + np.cos(epoch * np.pi / period))

    def get_lr(self):
        """
        Compute the learning rate for each parameter group using the SGDR policy.

        This method calculates the learning rate for each parameter group based on
        the progress through the current period. The adjustment is made considering
        the base period and its multiplication factor, taking into account the number
        of restarts.

        Returns
        -------
        list
            A list of learning rates, one for each parameter group in the optimizer.

        """

        epoch_idx = float(self.last_epoch)
        current_period = float(self.base_period)
        # Reduce the epoch index to the current period by subtracting the lengths
        # of all the completed periods.
        while epoch_idx / current_period > 1.0:
            epoch_idx -= current_period + 1
            current_period *= self.period_mult

        # Calculate the current learning rate based on the adjusted epoch index.
        current_lrs = self._get_current_lr(
            self.min_lr,
            self.max_lr,
            current_period,
            epoch_idx
        )
        # Return the list of learning rates to be applied to the parameter groups.
        return current_lrs.tolist()
