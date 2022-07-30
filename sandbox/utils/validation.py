"""The validation module."""
from numpy import ndarray
from pandas import DataFrame, Index, Series


def is_dataframe(obj):
    """Check whether object is pandas.DataFrame.

    Parameters
    ----------
    obj : pandas.DataFrame
        Input data

    Returns
    -------
    is_dataframe : bool
        If True, input is pandas.DataFrame.

    Examples
    --------
    >>> from sandbox.utils.validation import is_dataframe
    >>> import pandas as pd
    >>> obj1 = pd.DataFrame({"col1": [1] * 10})
    >>> is_dataframe_or_series(obj1)
    True
    >>> obj2 = 10
    >>> is_dataframe_or_series(obj2)
    False
    """
    return isinstance(obj, DataFrame)


def is_series(obj):
    """Check whether object is pandas.Series.

    Parameters
    ----------
    obj : pandas.Series
        Input data

    Returns
    -------
    is_series : bool
        If True, input is pandas.Series.

    Examples
    --------
    >>> from sandbox.utils.validation import is_series
    >>> import pandas as pd
    >>> obj1 = pd.Series({"col1": [1] * 10})
    >>> is_series(obj1)
    True
    """
    return isinstance(obj, Series)


def is_index(obj):
    """Check whether object is pandas.Index.

    Parameters
    ----------
    obj : pd.Index
        Input data

    Returns
    -------
    is_index : bool
        Whether the object was pandas.Index.

    Examples
    --------
    >>> from sandbox.utils.validation import is_index
    >>> import pandas as pd
    >>> obj1 = pd.pd.RangeIndex(start=0, stop=10, step=1)
    >>> is_index(obj1)
    True
    """
    return isinstance(obj, Index)


def is_dataframe_or_series(obj):
    """Check whether object is pandas.DataFrame or pandas.Series.

    Parameters
    ----------
    obj : pd.DataFrame or pd.Series
        Input data

    Returns
    -------
    is_dataframe_or_series : bool
        Whether the object was pandas.DataFrame or pd.Series.

    Examples
    --------
    >>> from sandbox.utils.validation import is_dataframe_or_series
    >>> import pandas as pd
    >>> obj1 = pd.DataFrame({"col1": [1] * 10})
    >>> is_dataframe_or_series(obj1)
    True
    >>> obj2 = pd.Series({"col1": [1] * 10})
    >>> is_dataframe_or_series(obj2)
    True
    >>> obj3 = [1, 1, 1]
    >>> is_dataframe_or_series(obj3)
    False
    """
    return isinstance(obj, (DataFrame, Series))


def is_using_padnas(X, y):
    """Whether both X and y are the class of pandas (DataFrame, Series).

    Parameters
    ----------
    X : pd.DataFrame or pd.Series
        Input data for X
    y : pd.DataFrame or pd.Series
        Input data for y

    Returns
    -------
    is_using_pandas : bool
        If True, both X and y are the class of pandas (DataFrame, Series).
    """
    return (is_dataframe_or_series(y) and (is_dataframe_or_series(X) or X is None)) or (
        is_dataframe_or_series(X) and (is_dataframe_or_series(y) or y is None)
    )


def is_ndarray(obj):
    """Check whether object is numpy.ndarray.
    Parameters
    ----------
    obj : numpy.ndarray
        Input data

    Returns
    -------
    is_series : bool
        If True, input is numpy.ndarray.

    Examples
    --------
    >>> from sandbox.utils.validation import is_ndarray
    >>> import numpy as np
    >>> obj1 = np.array([[1, 0], [0, 1]])
    >>> is_ndarray(obj1)
    True
    """
    return isinstance(obj, ndarray)


def is_using_ndarray(X, y):
    """Whether both X and y are the class of numpy (ndarray).

    Parameters
    ----------
    X : numpy.ndarray
        Input data for X
    y : numpy.ndarray
        Input data for y

    Returns
    -------
    is_using_numpy : bool
        If True, both X and y are the class of numpy (ndarray).
    """
    return (is_ndarray(y) and (is_ndarray(X) or X is None)) or (
        is_ndarray(X) and (is_ndarray(y) or y is None)
    )


def is_arraylike(obj):
    """Returns whether the input is array-like.

    Parameters
    ----------
    obj : array-like
        Input data

    Returns
    -------
    is_arraylike : bool
        Whether the object was array-like.

    Examples
    --------
    >>> from sandbox.utils.validation import is_arraylike
    >>> is_arraylike(["aa", "bb"])
    True

    >>> is_arraylike("cc")
    False
    """
    return isinstance(obj, (list, tuple, ndarray, Series))
