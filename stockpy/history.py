import json
import pickle

from stockpy.utils import open_file_like


# pylint: disable=invalid-name
class _none:
    """Special placeholder since ``None`` is a valid value."""


def _not_none(items):
    """
    Check if the given item or items are not placeholders.

    Determines if a single item or each item in an iterable is not a placeholder
    used internally in Stockpy.

    Parameters
    ----------
    items : object or iterable of objects
        The item or items to check against the placeholder.

    Returns
    -------
    bool
        True if none of the items is a placeholder, otherwise False.
    """

    if not isinstance(items, (tuple, list)):
        items = (items,)
    return all(item is not _none for item in items)


def _getitem_list_list(items, keys, tuple_=False):
    """
    Retrieve multiple values from a list of dicts or similar mappings based on provided keys.

    This function is designed to be efficient but may be hard to read. It attempts
    to retrieve a set of values indexed by the keys from each mapping in a list. If
    a key is not present in an individual mapping, that mapping is skipped.

    Parameters
    ----------
    items : list of mappings
        The list of mappings from which to retrieve the values.
    keys : iterable
        The keys for the values to be retrieved from each mapping.
    tuple_ : bool, optional
        If True, the returned list will contain tuples instead of lists (default is False).

    Returns
    -------
    list of lists or list of tuples
        A list of the extracted values, where each item is a list or tuple (depending on `tuple_`)
        of values corresponding to the provided keys. If no valid extraction is possible, returns
        `_none`.

    Notes
    -----
    - If any of the keys is not present in a mapping, that mapping is excluded from the result.
    - The function returns `_none` if no mappings contain all provided keys.
    """

    filtered = []
    for item in items:
        row = []
        for key in keys:
            try:
                row.append(item[key])
            except KeyError:
                break
        else:  # no break
            if row:
                filtered.append(tuple(row) if tuple_ else row)
    if items and not filtered:
        return _none
    return filtered


def _getitem_list_tuple(items, keys):
    """
    Retrieve multiple values as tuples from a list of mappings based on provided keys.

    This is a wrapper around `_getitem_list_list` that sets `tuple_` to True, thereby
    ensuring that the returned items are tuples of values.

    Parameters
    ----------
    items : list of mappings
        The list of mappings from which to retrieve the values.
    keys : iterable
        The keys for the values to be retrieved from each mapping.

    Returns
    -------
    list of tuples
        A list of tuples of the extracted values corresponding to the provided keys.
        If no valid extraction is possible, returns `_none`.

    See Also
    --------
    _getitem_list_list : The underlying function that performs the extraction.
    """

    return _getitem_list_list(items, keys, tuple_=True)

def _getitem_list_str(items, key):
    """
    Retrieve values associated with a specified key from a list of mappings.

    This function iterates over a list of mapping objects, such as dictionaries,
    and collects the value associated with a specified key in each mapping.

    Parameters
    ----------
    items : list of mappings
        The list of mappings from which to retrieve the values.
    key : hashable
        The key corresponding to the value to be retrieved from each mapping.

    Returns
    -------
    list
        A list containing the values associated with the `key` in each mapping.
        If the key is not present in any mapping, `_none` is returned.

    Notes
    -----
    If the key is not found in a mapping, that mapping is skipped.
    """

    filtered = []
    for item in items:
        try:
            filtered.append(item[key])
        except KeyError:
            continue
    if items and not filtered:
        return _none
    return filtered


def _getitem_dict_list(item, keys):
    """
    Retrieve a list of values from a mapping based on multiple keys.

    Given a single mapping and a list of keys, this function retrieves the value
    associated with each key in the mapping. If a key is not present, `_none` is
    used in its place.

    Parameters
    ----------
    item : mapping
        The mapping from which to retrieve the values.
    keys : iterable
        The keys for which values are to be retrieved.

    Returns
    -------
    list
        A list of retrieved values, with `_none` for any keys that are not present.
    """

    return [item.get(key, _none) for key in keys]


def _getitem_dict_tuple(item, keys):
    """
    Retrieve a tuple of values from a mapping based on multiple keys.

    This is similar to `_getitem_dict_list` but returns a tuple of values instead of a list.
    If a key is not present in the mapping, `_none` is used in its place.

    Parameters
    ----------
    item : mapping
        The mapping from which to retrieve the values.
    keys : iterable
        The keys for which values are to be retrieved.

    Returns
    -------
    tuple
        A tuple of retrieved values, with `_none` for any keys that are not present.
    """

    return tuple(item.get(key, _none) for key in keys)


def _getitem_dict_str(item, key):
    """
    Retrieve a single value from a mapping based on a single key.

    Given a mapping and a key, this function retrieves the value associated with that
    key. If the key is not present, `_none` is returned.

    Parameters
    ----------
    item : mapping
        The mapping from which to retrieve the value.
    key : hashable
        The key corresponding to the value to be retrieved.

    Returns
    -------
    value
        The value retrieved from the mapping, or `_none` if the key is not present.
    """

    return item.get(key, _none)


def _get_getitem_method(items, key):
    """
    Dynamically selects the correct method to extract values from 'items' based on the type of 'key'.

    Depending on whether 'items' is a list or a dictionary and on the type of 'key' (list, tuple, or str),
    a specialized function is returned for efficient item extraction. This assumes that the type of 'items'
    and the type of 'key' do not change throughout the process.

    The function handles the following patterns:
    - history[0, 'foo', :10]: expects a list of items.
    - history[0, 'foo', 0]: expects a dictionary.
    - history[0, 'foo', :, 'bar']: expects 'key' as a string.
    - history[0, 'foo', :, ('bar', 'baz')]: expects 'key' as a list/tuple of strings.

    Parameters
    ----------
    items : list or dict
        The collection of items from which values will be extracted.
    key : list, tuple, or str
        The key(s) representing the values to extract from the collection.

    Returns
    -------
    function
        A function that is appropriate for extracting the desired item based on the provided 'items' and 'key'.

    Raises
    ------
    TypeError
        If the combination of the type of 'items' and 'key' is not supported, a TypeError is raised.

    Examples
    --------
    >>> _get_getitem_method([{'foo': 1}], 'foo')
    <function _getitem_list_str at 0x...>

    >>> _get_getitem_method({'foo': 1}, 'foo')
    <function _getitem_dict_str at 0x...>

    >>> _get_getitem_method([{'foo': 1}], ['foo', 'bar'])
    <function _getitem_list_list at 0x...>
    """

    if isinstance(items, list):
        if isinstance(key, list):
            return _getitem_list_list
        if isinstance(key, tuple):
            return _getitem_list_tuple
        if isinstance(key, str):
            return _getitem_list_str
        raise TypeError("History access with given types not supported")

    if isinstance(items, dict):
        if isinstance(key, list):
            return _getitem_dict_list
        if isinstance(key, tuple):
            return _getitem_dict_tuple
        if isinstance(key, str):
            return _getitem_dict_str
    raise TypeError("History access with given types not supported")


def _unpack_index(i):
    """
    Unpacks a given index and ensures it contains exactly four elements.

    Indices are used to access a structured dataset, where the maximum depth
    can go up to four levels. If the provided index is shorter, `None` is used
    to fill the missing dimensions. However, if the index has more than four
    elements, a KeyError is raised.

    Parameters
    ----------
    i : tuple
        The index to unpack, which can contain up to four elements.

    Returns
    -------
    tuple
        A four-element tuple where the original elements are preserved, and
        `None` is used to fill in any missing dimensions.

    Raises
    ------
    KeyError
        If the input index contains more than four elements, which is not supported.

    Examples
    --------
    >>> _unpack_index((1, 'loss'))
    (1, 'loss', None, None)

    >>> _unpack_index((1, 'loss', 0, 'valid'))
    (1, 'loss', 0, 'valid')

    >>> _unpack_index((1, 'loss', 0, 'valid', 'extra'))
    KeyError: Tried to index history with 5 indices but only 4 indices are possible.

    Notes
    -----
    This function is primarily used for accessing elements in a hierarchical data
    structure with a maximum of four levels, such as a nested history object in
    a machine learning framework.
    """

    if len(i) > 4:
        raise KeyError(
            "Tried to index history with {} indices but only "
            "4 indices are possible.".format(len(i)))

    # fill trailing indices with None
    i_e, k_e, i_b, k_b = i + tuple([None] * (4 - len(i)))

    return i_e, k_e, i_b, k_b


class History(list):
    """
    Container for managing the training history during the training process of a neural network.

    The `History` object is essentially a list of dictionaries, each representing an epoch,
    with keys providing various metrics and values corresponding to these metrics at each epoch.

    Enhanced slicing notation and convenience methods are provided to ease the interaction with the
    training history.

    Methods
    -------
    __getitem__(self, i):
        Enhanced slicing to retrieve items from the history.

    new_epoch(self):
        Start recording a new epoch.

    record(self, column, value):
        Record a new item within the current epoch.

    new_batch(self):
        Start recording a new batch within the current epoch.

    record_batch(self, column, value):
        Record a new item within the current batch.

    Examples
    --------
    >>> history = net.history  # Assuming 'net' is an instance of a trained neural network
    >>> history[-1]  # Access the last epoch's information
    >>> history[:, 'train_loss']  # Access training losses from all epochs
    >>> history[:, ('train_loss', 'valid_loss')]  # Access training and validation losses from all epochs
    >>> history[-1, 'batches', -1]  # Access the last batch of the last epoch
    >>> history[-1, 'batches', :, 'train_loss']  # Access training losses from all batches of the last epoch

    Notes
    -----
    Accessing non-existing keys will lead to a KeyError. Also, if multiple keys are requested,
    only the epochs or batches containing all those keys will be returned.
    """

    def new_epoch(self):
        """
        Register a new epoch.

        Appends a new dictionary to the history list, representing a new epoch. This dictionary is initialized with an empty 'batches' list.

        Examples
        --------
        >>> history = History()
        >>> history.new_epoch()
        >>> print(history)
        [{'batches': []}]
        """

        self.append({'batches': []})

    def new_batch(self):
        """
        Add a new batch to the last recorded epoch in the history.

        This method is used to record the start of a new batch during the training of a model. It appends an empty dictionary 
        to the 'batches' list of the last recorded epoch in the history. This dictionary will later be filled with the metrics 
        recorded during the batch.

        Raises
        ------
        IndexError
            If there are no epochs recorded in the history. This is because batches are recorded within epochs, so there must 
            be at least one epoch recorded before a batch can be added.

        Notes
        -----
        This method modifies the history in-place, adding a new batch to the last recorded epoch.
        """
        
        self[-1]['batches'].append({})

    def record(self, attr, value):
        """
        Record a new attribute value for the current epoch.

        Parameters
        ----------
        attr : str
            The attribute name where the value needs to be recorded.
        value : any
            The value to be recorded under the attribute name for the current epoch.

        Raises
        ------
        ValueError
            If the history is empty and no epoch has been created yet.

        Examples
        --------
        >>> history = History()
        >>> history.new_epoch()
        >>> history.record('train_loss', 0.25)
        >>> print(history)
        [{'batches': [], 'train_loss': 0.25}]
        """

        if not self:
            raise ValueError("Call new_epoch before recording for the first time.")
        self[-1][attr] = value

    def record_batch(self, attr, value):
        """
        Record a new attribute value for the current batch.

        This method logs a value for a specified attribute in the current batch within the current epoch.

        Parameters
        ----------
        attr : str
            The attribute name where the value needs to be recorded.
        value : any
            The value to be recorded under the attribute name for the current batch.

        Raises
        ------
        IndexError
            If there are no epochs or batches in the history to which the value can be appended.

        Examples
        --------
        >>> history = History()
        >>> history.new_epoch()
        >>> history.new_batch()
        >>> history.record_batch('batch_loss', 0.15)
        >>> print(history)
        [{'batches': [{'batch_loss': 0.15}]}]
        """
        
        self[-1]['batches'][-1][attr] = value

    def to_list(self):
        """
        Convert the History object to a list representation.

        Returns
        -------
        list
            A list that contains the same information as the History object.

        Examples
        --------
        >>> history = History()
        >>> history.new_epoch()
        >>> history.record('epoch_loss', 0.1)
        >>> history_list = history.to_list()
        >>> print(history_list)
        [{'batches': [], 'epoch_loss': 0.1}]
        """

        return list(self)

    @classmethod
    def from_file(cls, f):
        """
        Create a History instance from a JSON file.

        Parameters
        ----------
        f : file-like object or str
            The file from which to load the history. Can be a file path or a file-like object.

        Returns
        -------
        History
            An instance of History initialized with data from the provided file.

        Examples
        --------
        >>> # Assuming 'history.json' contains the JSON-encoded history
        >>> history = History.from_file('history.json')
        >>> print(history)
        [{'epoch': 1, 'train_loss': 0.2}, {'epoch': 2, 'train_loss': 0.15}]
        """

        with open_file_like(f, 'r') as fp:
            return cls(json.load(fp))

    def to_file(self, f):
        """
        Save the History instance to a JSON file.

        Parameters
        ----------
        f : file-like object or str
            The file path or file-like object where the history will be saved.

        Examples
        --------
        >>> history = History()
        >>> history.new_epoch()
        >>> history.record('epoch_loss', 0.1)
        >>> history.to_file('history.json')  # Saves the history to 'history.json'.
        """

        with open_file_like(f, 'w') as fp:
            json.dump(self.to_list(), fp)

    def __getitem__(self, i):
        """
        Retrieve items from the history using advanced indexing.

        Parameters
        ----------
        i : int, slice, tuple
            The index. Can be an integer for epochs, a slice for a range of epochs, or a tuple 
            that specifies a detailed path in the history structure (up to four elements: epoch index,
            epoch-level key, batch index, and batch-level key).

        Returns
        -------
        item : list, dict, or value
            Depending on the index, may return a list of items, a single dictionary, or a value.

        Raises
        ------
        KeyError
            If a specified key is not found or the indexing is incorrect.

        Examples
        --------
        >>> history = History()
        >>> history.new_epoch()
        >>> history.record('loss', 0.1)
        >>> history.new_batch()
        >>> history.record_batch('batch_loss', 0.01)
        >>> history[0]  # Returns the first epoch data
        {'loss': 0.1, 'batches': [{'batch_loss': 0.01}]}
        >>> history[:, 'loss']  # Returns all 'loss' values across epochs
        [0.1]
        >>> history[-1, 'batches', -1]  # Returns the last batch of the last epoch
        {'batch_loss': 0.01}
        >>> history[:, ('loss',)]  # Returns a tuple of 'loss' values across epochs
        [(0.1,)]

        Notes
        -----
        This method allows for flexible and powerful indexing mechanisms to navigate
        through the hierarchical structure of training history, facilitating quick
        and intuitive retrieval of data.
        """

        # This implementation resolves indexing backwards,
        # i.e. starting from the batches, then progressing to the
        # epochs.
        if isinstance(i, (int, slice)):
            i = (i,)

        # i_e: index epoch, k_e: key epoch
        # i_b: index batch, k_b: key batch
        i_e, k_e, i_b, k_b = _unpack_index(i)
        keyerror_msg = "Key {!r} was not found in history."

        if i_b is not None and k_e != 'batches':
            raise KeyError("History indexing beyond the 2nd level is "
                           "only possible if key 'batches' is used, "
                           "found key {!r}.".format(k_e))

        items = self.to_list()

        # extract the epochs
        # handles: history[i_e]
        if i_e is not None:
            items = items[i_e]
            if isinstance(i_e, int):
                items = [items]

        # extract indices of batches
        # handles: history[..., k_e, i_b]
        if i_b is not None:
            items = [row[k_e][i_b] for row in items]

        # extract keys of epochs or batches
        # handles: history[..., k_e]
        # handles: history[..., ..., ..., k_b]
        if k_e is not None and (i_b is None or k_b is not None):
            key = k_e if k_b is None else k_b

            if items:
                extract = _get_getitem_method(items[0], key)
                items = [extract(item, key) for item in items]

                # filter out epochs with missing keys
                items = list(filter(_not_none, items))

            if not items and not (k_e == 'batches' and i_b is None):
                # none of the epochs matched
                raise KeyError(keyerror_msg.format(key))

            if (
                    isinstance(i_b, slice)
                    and k_b is not None
                    and not any(batches for batches in items)
            ):
                # none of the batches matched
                raise KeyError(keyerror_msg.format(key))

        if isinstance(i_e, int):
            items, = items

        return items