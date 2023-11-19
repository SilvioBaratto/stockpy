""" Callbacks for printing, logging and log information."""

import sys
import time
import tempfile
from contextlib import suppress
from numbers import Number
from itertools import cycle
from pathlib import Path

import numpy as np
import tqdm
from tabulate import tabulate

from stockpy.utils import Ansi
from stockpy.preprocessing import get_len
from stockpy.callbacks import Callback

__all__ = ['EpochTimer', 'PrintLog']

def filter_log_keys(keys, keys_ignored=None):
    """
    Yield keys from an iterable that are not designated to be ignored for logging.

    This utility function is used within callbacks to filter out certain keys from a 
    collection based on predefined criteria. Keys associated with epochs, those specified 
    in `keys_ignored`, keys ending with '_best', keys ending with '_batch_count', or 
    keys starting with 'event_' are excluded from the output.

    Parameters
    ----------
    keys : iterable of str
        An iterable containing keys that are to be considered for filtering.
    keys_ignored : iterable of str, optional
        An additional set of keys that should be ignored during filtering. If None 
        is given, no additional keys are ignored beyond the default criteria.

    Yields
    ------
    str
        Keys that are not to be ignored based on the filtering criteria.

    Examples
    --------
    >>> all_keys = ['loss', 'val_loss', 'epoch', 'event_on_batch_end', 'accuracy_best']
    >>> list(filter_log_keys(all_keys))
    ['loss', 'val_loss']

    >>> list(filter_log_keys(all_keys, keys_ignored=['val_loss']))
    ['loss']

    Notes
    -----
    The default keys ignored are epoch numbers, the best values of metrics, batch counts,
    and any keys that begin with 'event_'. Additional keys to be ignored can be specified 
    through the `keys_ignored` parameter.
    """

    keys_ignored = keys_ignored or ()
    for key in keys:
        if not (
                key == 'epoch' or
                (key in keys_ignored) or
                key.endswith('_best') or
                key.endswith('_batch_count') or
                key.startswith('event_')
        ):
            yield key

class EpochTimer(Callback):
    """
    Callback for tracking the duration of each training epoch.

    This callback measures the time taken for each epoch during the model's training 
    process and records it into the history object under the key 'dur'.

    Attributes
    ----------
    epoch_start_time_ : float
        The timestamp when the current epoch began. This is used to calculate the epoch
        duration.

    Notes
    -----
    The duration is stored in seconds.
    """
    def __init__(self, **kwargs):
        super(EpochTimer, self).__init__(**kwargs)
        self.epoch_start_time_ = None

    def on_epoch_begin(self, net, **kwargs):
        """Record the start time of the epoch."""
        self.epoch_start_time_ = time.time()

    def on_epoch_end(self, net, **kwargs):
        """Calculate and record the end time of the epoch, storing the duration."""
        net.history.record('dur', time.time() - self.epoch_start_time_)

class PrintLog(Callback):
    """
    Prints the training log as a formatted table after each epoch.

    This callback outputs a summary of the training process in a tabular format,
    showing the evolution of metrics over epochs. It is designed to work well with
    the output from `EpochScoring` callbacks, which can create entries in the log
    that indicate the best values for specific metrics.

    Parameters
    ----------
    keys_ignored : str or list of str, optional
        Keys from the log that should not be included in the printed table. This can
        be a single string key or a list of string keys. Keys that start with ``event_``
        or end with '_best' are automatically ignored, as they are handled
        specially. The default value is None, which means no additional keys are
        ignored beyond the defaults.
    sink : callable, optional
        A callable that will receive the formatted table string. By default, this is
        the print function, which will print the output to stdout. However, it could
        also be a logging function or any other callable that accepts a single string
        argument.
    tablefmt : str, optional
        The format specification for the table. This follows the conventions used by
        the `tabulate` library. Examples include 'plain', 'grid', 'pipe', 'html',
        'latex', and the default 'simple'.
    floatfmt : str, optional
        Format specification for floating-point numbers. This is a format string used
        to control the output of floating-point numbers. The default is '.4f', which
        rounds to four decimal places.
    stralign : str, optional
        Alignment of the columns containing string data. This can be 'left', 'center',
        'right', or None (which will disable alignment). The default is 'right' to
        align string columns to the right, matching the alignment of numeric columns.

    Notes
    -----
    By default, 'PrintLog' excludes keys with the ``event_`` prefix and those ending
    with '_best' from printing since they're handled specially. It also avoids
    printing the 'batches' key by default.

    'PrintLog' is sensitive to varying number of columns. For the best output,
    the set of metrics should remain consistent across epochs.
    """

    def __init__(
            self,
            keys_ignored=None,
            sink=print,
            tablefmt='simple',
            floatfmt='.4f',
            stralign='right',
    ):
        self.keys_ignored = keys_ignored
        self.sink = sink
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt
        self.stralign = stralign

    def initialize(self):
        """
        Initializes the callback before training begins.

        This method sets up the necessary properties for the callback, including handling
        which keys should be ignored when printing the log. The method is called before
        training starts to ensure that the callback is ready to be used.

        The `initialize` method is internally used by the callback mechanism and is not
        typically called manually by users.

        """

        self.first_iteration_ = True

        keys_ignored = self.keys_ignored
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored_ = set(keys_ignored or [])
        self.keys_ignored_.add('batches')
        return self

    def format_row(self, row: dict, key: str, color: Ansi) -> str:
        """
        Formats a single row entry for the log output based on the type of value.

        This method is responsible for formatting a value for a specific key
        from the history row to be displayed in the log table. It handles
        different types of values including booleans, None, numbers, and others.
        The method also adds color to the output if the value is the best one
        so far for a particular key.

        Parameters
        ----------
        row : dict
            A dictionary representing one row of the history. Each key corresponds
            to a column in the log table.
        key : str
            The key for which the corresponding value is to be formatted from the row.
        color : Ansi
            An instance of the Ansi class from the `colorama` package, providing the
            ANSI escape sequence to add color to the output.

        Returns
        -------
        str
            The formatted string representation of the value to be logged. This can
            be a simple string, a color-coded string, or an empty string for boolean
            values that are False or None.
        """
        value = row[key]

        if isinstance(value, bool) or value is None:
            return '+' if value else ''

        if not isinstance(value, Number):
            return value

        # determine if integer value
        is_integer = float(value).is_integer()
        template = '{}' if is_integer else '{:' + self.floatfmt + '}'

        # if numeric, there could be a 'best' key
        key_best = key + '_best'
        if (key_best in row) and row[key_best]:
            template = color + template + Ansi.ENDC.value
        return template.format(value)

    def _sorted_keys(self, keys):
        """
        Sort the keys for displaying in the log output with a predefined order.

        This method organizes the keys so that 'epoch' appears first, followed by
        the main metric keys, then any 'event_' keys, and finally 'dur' at the end.
        It also filters out keys that have been marked as ignored. The purpose is
        to create a consistent and easy-to-read logging output during training.

        Parameters
        ----------
        keys : list of str
            The list of keys from the history to be sorted for display.

        Returns
        -------
        list of str
            The list of sorted keys after filtering and arranging according to the
            desired order for the log output.
        """
        sorted_keys = []

        # make sure 'epoch' comes first
        if ('epoch' in keys) and ('epoch' not in self.keys_ignored_):
            sorted_keys.append('epoch')

        # ignore keys like *_best or event_*
        for key in filter_log_keys(sorted(keys), keys_ignored=self.keys_ignored_):
            if key != 'dur':
                sorted_keys.append(key)

        # add event_* keys
        for key in sorted(keys):
            if key.startswith('event_') and (key not in self.keys_ignored_):
                sorted_keys.append(key)

        # make sure 'dur' comes last
        if ('dur' in keys) and ('dur' not in self.keys_ignored_):
            sorted_keys.append('dur')

        return sorted_keys

    def _yield_keys_formatted(self, row: dict):
        """
        Yields formatted key-value pairs for printing the log row.

        This generator function cycles through each key in the sorted keys of the
        row, formats the associated value, and yields a tuple of the cleaned key
        name (without 'event_' prefix, if present) and the formatted value string.
        Event keys are specially treated by removing the 'event_' prefix.

        Parameters
        ----------
        row : dict
            A dictionary representing the log row where the key is the metric or
            attribute name, and the value is the corresponding data to be logged.

        Yields
        ------
        tuple of (str, str)
            A tuple containing the cleaned key name and the formatted value string
            ready for logging output.
        """
        # Use a color cycle to alternate colors for each key-value pair
        colors = cycle([color.value for color in Ansi if color != color.ENDC])

        # Iterate over the sorted keys and their corresponding color
        for key, color in zip(self._sorted_keys(row.keys()), colors):
            # Format the value using the defined format_row method
            formatted = self.format_row(row, key, color=color)

            # Clean up the 'event_' prefix from event keys for display purposes
            if key.startswith('event_'):
                key = key[6:]

            # Yield the cleaned key and formatted value as a tuple
            yield key, formatted

    def table(self, row: dict) -> str:
        """
        Create a formatted table from the log row data for printing.

        This method uses the `_yield_keys_formatted` generator to process the log row
        and collects the headers (the key names) and their formatted values. It then
        utilizes the `tabulate` library to create a table that is ready to be printed
        or logged.

        Parameters
        ----------
        row : dict
            The log row from which the table is created. Each key-value pair in the
            row corresponds to a column in the table.

        Returns
        -------
        str
            A string representing the formatted table, ready to be output to the
            configured `sink`.
        """
        # Lists to hold the headers and the formatted rows
        headers = []
        formatted = []

        # Use the generator function to process each key-value pair
        for key, formatted_row in self._yield_keys_formatted(row):
            headers.append(key)          # Collect the header
            formatted.append(formatted_row)  # Collect the formatted value

        # Use the `tabulate` library to create a formatted table from the collected data
        return tabulate(
            [formatted],                 # The row data, expected as a list of lists
            headers=headers,             # The headers for the table
            tablefmt=self.tablefmt,      # The table format to use
            floatfmt=self.floatfmt,      # The float format for numeric values
            stralign=self.stralign,      # The string alignment for headers
        )

    def _sink(self, text, verbose):
        """
        Output the text to the configured sink based on verbosity.

        This method sends the provided text to the `sink` function, which defaults
        to `print`, but could be overridden by any other callable that takes a string
        argument (such as a logging function). The method respects the verbosity setting;
        if `verbose` is set to False and `sink` is the `print` function, it won't output
        anything, thus suppressing the output.

        Parameters
        ----------
        text : str
            The text to be output.
        verbose : bool
            Flag indicating whether verbose output is enabled. If True, the text will
            always be output; if False, the text will only be output if `sink` is not
            the `print` function.

        Notes
        -----
        - This design allows the user to route the callback's output to a logger or
        other output streams easily without changing the verbosity of other parts
        of the code that use the standard `print` function.
        - It's important to note that setting `verbose=False` will only suppress
        output if the `sink` is `print`. If `sink` is a different callable, the
        output will happen regardless of the `verbose` flag, which allows for
        silent operation with logging.
        """
        # Check if the sink is not the print function or if verbose output is enabled
        if (self.sink is not print) or verbose:
            self.sink(text)  # Output the text using the configured sink function

    def on_epoch_end(self, net, **kwargs):
        """
        Actions performed at the end of an epoch: printing the epoch results.

        This method is called at the end of each epoch and is responsible for
        printing the results of the epoch in a nicely formatted table. If it's
        the first iteration, it prints the header and the first line of the table.
        For subsequent iterations, it only prints the latest results. The method
        ensures that the output is flushed to stdout if the sink is set to print,
        which is useful for real-time monitoring in environments where the stdout
        buffer might delay the display of the results.

        Parameters
        ----------
        net : nn.Module or PyroModule
            The neural network instance.
        **kwargs : dict
            Additional arguments passed to the callback. Not used in this method.

        """

        # Retrieve the latest epoch data from history
        data = net.history[-1]  
        # Get the verbosity setting from the network
        verbose = net.verbose  

        # Create a string representation of the table
        tabulated = self.table(data)  

        # Print the header and first line if it's the first epoch
        if self.first_iteration_:
            header, lines = tabulated.split('\n', 2)[:2]
            # Print the header
            self._sink(header, verbose)
            # Print the first line of the table  
            self._sink(lines, verbose) 
            # Update the flag to avoid reprinting the header  
            self.first_iteration_ = False 

        # Print the latest results
        self._sink(tabulated.rsplit('\n', 1)[-1], verbose)

        # Flush the stdout buffer if the output is being printed to the console
        if self.sink is print:
            # Ensures real-time output on the console
            sys.stdout.flush()  