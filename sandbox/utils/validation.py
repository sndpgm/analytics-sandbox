from numpy import ndarray
from pandas import DataFrame, Series


def is_dataframe(obj):
    return isinstance(obj, DataFrame)


def is_series(obj):
    return isinstance(obj, Series)


def is_dataframe_or_series(obj):
    """Check whether object is pd.DataFrame or pd.Series.

    Parameters
    ----------
    obj : pd.DataFrame or pd.Series
        Input data

    Returns
    -------
    is_dataframe_or_series : bool
        Whether the object was pd.DataFrame or pd.Series.

    Examples
    --------
    >>> from sandbox.utils.validation import is_dataframe_or_series
    >>> import pandas as pd
    >>> obj1 = pd.DataFrame({"col1": [1] * 10})
    >>> is_dataframe_or_series(obj1)
    True
    """
    return isinstance(obj, (DataFrame, Series))


def is_using_padnas(X, y):
    return (is_dataframe_or_series(y) and (is_dataframe_or_series(X) or X is None)) or (
        is_dataframe_or_series(X) and (is_dataframe_or_series(y) or y is None)
    )


def is_ndarray(obj):
    return isinstance(obj, ndarray)


def is_using_ndarray(X, y):
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
    return isinstance(obj, (list, set, ndarray, Series))


def check_1d_series(y):
    if not is_dataframe_or_series(y):
        msg = "y must be DataFrame or Series of pandas."
        raise TypeError(msg)
    if isinstance(y, DataFrame):
        if y.shape[1] != 1:
            msg = "y must be Series or DataFrame having just one column."
            raise ValueError(msg)
        name = y.columns[0]
        return Series(y[name].copy())
    else:
        return y.copy()


def check_2d_dataframe(X, name=None):
    if not is_dataframe_or_series(X):
        msg = "X must be DataFrame or Series of pandas."
        raise TypeError(msg)

    if name is not None and not is_arraylike(name):
        msg = "Specified name is not array-like."
        raise TypeError(msg)

    # If name is defined, that is used as generated DataFrame column name.
    if isinstance(X, Series):
        if name is not None and len(name) == 1:
            return DataFrame(X.copy().values, columns=name)
        else:
            return DataFrame(X.copy())
    else:
        _X = X.copy()

        if name is not None and len(X.columns) == len(name):
            _X.columns = name

        return _X


def check_X_y(X, y):
    if y is None:
        msg = "y must be not None."
        raise ValueError(msg)

    y = check_1d_series(y)

    if X is not None:
        X = check_2d_dataframe(X)
        if not X.index.equals(y.index):
            msg = "X and Y must have the same index, given: {}, {}".format(
                X.index, y.index
            )
            raise ValueError(msg)

    return X, y
