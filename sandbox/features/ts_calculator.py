from __future__ import annotations

import inspect

import pandas as pd

try:
    from dask import dataframe as dd
except ImportError:
    dd = None


def set_property(key, value):
    """This method returns a decorator that sets the property key of the function to value"""

    def decorate_func(func):
        setattr(func, key, value)
        return func

    return decorate_func


def hour(
    x: pd.DataFrame | dd.DataFrame, value: str
) -> tuple[pd.Series | dd.Series, str]:
    """The hours of the datetime.
    Internal execution is `pandas.Series.dt.hour` or `dask.dataframe.Series.dt.hour`.

    Parameters
    ----------
    x : {pandas.DataFrame, dask.DataFrame}
        Target input.
    value : str
        Expressing the target column name.

    Returns
    -------
    {pandas.Series, dask.Series}
        Series on the hours of the datetime.
    list[str]
        List of column names.

    Examples
    --------
    >>> import pandas as pd
    >>> import sandbox.features.ts_calculator as tsc
    >>> x = pd.DataFrame({"x": pd.to_datetime(["2020-01-01 12:00:00", "2020-06-15 23:00:00",
    ...                                       "2021-02-12 01:00:00"])})
    >>> tsc.hour(x, value="x")[0]
    0    12
    1    23
    2     1
    Name: x__hour, dtype: int64
    >>> tsc.hour(x, value="x")[1]
    'x__hour'

    See Also
    --------
    pandas.Series.dt.hour : The hours of the datetime in pandas.
    dask.dataframe.Series.dt.hour : The hours of the datetime in dask.dataframe.
    """
    column_name = value + "__" + inspect.currentframe().f_code.co_name
    ret = x[value].dt.hour
    ret.name = column_name
    return ret, column_name


def dayofweek(
    x: pd.DataFrame | dd.DataFrame, value: str
) -> tuple[pd.Series | dd.Series, str]:
    """The day of the week with Monday=0, Sunday=6.
    Internal execution is `pandas.Series.dt.dayofweek` or `dask.dataframe.Series.dt.dayofweek`.

    Parameters
    ----------
    x : {pandas.DataFrame, dask.DataFrame}
        Target input.
    value : str
        Expressing the target column name.

    Returns
    -------
    {pandas.Series, dask.Series}
        Series on the day of the week of the datetime.
    list[str]
        List of column names.

    Examples
    --------
    >>> import pandas as pd
    >>> import sandbox.features.ts_calculator as tsc
    >>> x = pd.DataFrame({"x": pd.to_datetime(["2020-01-01 12:00:00", "2020-06-15 23:00:00",
    ...                                       "2021-02-12 01:00:00"])})
    >>> tsc.dayofweek(x, value="x")[0]
    0    2
    1    0
    2    4
    Name: x__dayofweek, dtype: int64
    >>> tsc.dayofweek(x, value="x")[1]
    'x__dayofweek'

    See Also
    --------
    pandas.Series.dt.dayofweek : The day of the week in pandas.
    dask.dataframe.Series.dt.dayofweek : The day of the week in dask.dataframe.
    """
    column_name = value + "__" + inspect.currentframe().f_code.co_name
    ret = x[value].dt.dayofweek
    ret.name = column_name
    return ret, column_name


def quarter(
    x: pd.DataFrame | dd.DataFrame, value: str
) -> tuple[pd.Series | dd.Series, str]:
    """The quarter of the date.
    Internal execution is `pandas.Series.dt.quarter` or `dask.dataframe.Series.dt.quarter`.

    Parameters
    ----------
    x : {pandas.DataFrame, dask.DataFrame}
        Target input.
    value : str
        Expressing the target column name.

    Returns
    -------
    {pandas.Series, dask.Series}
        Series on the quarter of the date.
    list[str]
        List of column names.

    Examples
    --------
    >>> import pandas as pd
    >>> import sandbox.features.ts_calculator as tsc
    >>> x = pd.DataFrame({"x": pd.to_datetime(["2020-01-01 12:00:00", "2020-06-15 23:00:00",
    ...                                       "2021-02-12 01:00:00"])})
    >>> tsc.quarter(x, value="x")[0]
    0    1
    1    2
    2    1
    Name: x__quarter, dtype: int64
    >>> tsc.quarter(x, value="x")[1]
    'x__quarter'

    See Also
    --------
    pandas.Series.dt.quarter : The quarter of the date in pandas.
    dask.dataframe.Series.dt.quarter : The quarter of the date in dask.dataframe.
    """
    column_name = value + "__" + inspect.currentframe().f_code.co_name
    ret = x[value].dt.quarter
    ret.name = column_name
    return ret, column_name


def month(
    x: pd.DataFrame | dd.DataFrame, value: str
) -> tuple[pd.Series | dd.Series, str]:
    """The month of the date.
    Internal execution is `pandas.Series.dt.month` or `dask.dataframe.Series.dt.month`.

    Parameters
    ----------
    x : {pandas.DataFrame, dask.DataFrame}
        Target input.
    value : str
        Expressing the target column name.

    Returns
    -------
    {pandas.Series, dask.Series}
        Series on the month of the date.
    list[str]
        List of column names.

    Examples
    --------
    >>> import pandas as pd
    >>> import sandbox.features.ts_calculator as tsc
    >>> x = pd.DataFrame({"x": pd.to_datetime(["2020-01-01 12:00:00", "2020-06-15 23:00:00",
    ...                                       "2021-02-12 01:00:00"])})
    >>> tsc.month(x, value="x")[0]
    0    1
    1    6
    2    2
    Name: x__month, dtype: int64
    >>> tsc.month(x, value="x")[1]
    'x__month'

    See Also
    --------
    pandas.Series.dt.month : The month of the date in pandas.
    dask.dataframe.Series.dt.month : The month of the date in dask.dataframe.
    """
    column_name = value + "__" + inspect.currentframe().f_code.co_name
    ret = x[value].dt.month
    ret.name = column_name
    return ret, column_name


def year(
    x: pd.DataFrame | dd.DataFrame, value: str
) -> tuple[pd.Series | dd.Series, str]:
    """The month of the date.
    Internal execution is `pandas.Series.dt.year` or `dask.dataframe.Series.dt.year`.

    Parameters
    ----------
    x : {pandas.DataFrame, dask.DataFrame}
        Target input.
    value : str
        Expressing the target column name.

    Returns
    -------
    {pandas.Series, dask.Series}
        Series on the year of the date.
    list[str]
        List of column names.

    Examples
    --------
    >>> import pandas as pd
    >>> import sandbox.features.ts_calculator as tsc
    >>> x = pd.DataFrame({"x": pd.to_datetime(["2020-01-01 12:00:00", "2020-06-15 23:00:00",
    ...                                       "2021-02-12 01:00:00"])})
    >>> tsc.year(x, value="x")[0]
    0    2020
    1    2020
    2    2021
    Name: x__year, dtype: int64
    >>> tsc.year(x, value="x")[1]
    'x__year'

    See Also
    --------
    pandas.Series.dt.year : The year of the date in pandas.
    dask.dataframe.Series.dt.year : The year of the date in dask.dataframe.
    """
    column_name = value + "__" + inspect.currentframe().f_code.co_name
    ret = x[value].dt.year
    ret.name = column_name
    return ret, column_name


def dayofyear(
    x: pd.DataFrame | dd.DataFrame, value: str
) -> tuple[pd.Series | dd.Series, str]:
    """The ordinal day of the year.
    Internal execution is `pandas.Series.dt.dayofyear` or `dask.dataframe.Series.dt.dayofyear`.

    Parameters
    ----------
    x : {pandas.DataFrame, dask.DataFrame}
        Target input.
    value : str
        Expressing the target column name.

    Returns
    -------
    {pandas.Series, dask.Series}
        Series on the ordinal day of the year.
    list[str]
        List of column names.

    Examples
    --------
    >>> import pandas as pd
    >>> import sandbox.features.ts_calculator as tsc
    >>> x = pd.DataFrame({"x": pd.to_datetime(["2020-01-01 12:00:00", "2020-06-15 23:00:00",
    ...                                       "2021-02-12 01:00:00"])})
    >>> tsc.dayofyear(x, value="x")[0]
    0      1
    1    167
    2     43
    Name: x__dayofyear, dtype: int64
    >>> tsc.dayofyear(x, value="x")[1]
    'x__dayofyear'

    See Also
    --------
    pandas.Series.dt.dayofyear : The ordinal day of the year in pandas.
    dask.dataframe.Series.dt.dayofyear : The ordinal day of the year in dask.dataframe.
    """
    column_name = value + "__" + inspect.currentframe().f_code.co_name
    ret = x[value].dt.dayofyear
    ret.name = column_name
    return ret, column_name


def dayofmonth(
    x: pd.DataFrame | dd.DataFrame, value: str
) -> tuple[pd.Series | dd.Series, str]:
    """The day of the datetime.
    Internal execution is `pandas.Series.dt.day` or `dask.dataframe.Series.dt.day`.

    Parameters
    ----------
    x : {pandas.DataFrame, dask.DataFrame}
        Target input.
    value : str
        Expressing the target column name.

    Returns
    -------
    {pandas.Series, dask.Series}
        Series on the day of the datetime.
    list[str]
        List of column names.

    Examples
    --------
    >>> import pandas as pd
    >>> import sandbox.features.ts_calculator as tsc
    >>> x = pd.DataFrame({"x": pd.to_datetime(["2020-01-01 12:00:00", "2020-06-15 23:00:00",
    ...                                       "2021-02-12 01:00:00"])})
    >>> tsc.dayofmonth(x, value="x")[0]
    0     1
    1    15
    2    12
    Name: x__dayofmonth, dtype: int64
    >>> tsc.dayofmonth(x, value="x")[1]
    'x__dayofmonth'

    See Also
    --------
    pandas.Series.dt.day : The day of the datetime in pandas.
    dask.dataframe.Series.dt.day : The day of the datetime in dask.dataframe.
    """
    column_name = value + "__" + inspect.currentframe().f_code.co_name
    ret = x[value].dt.day
    ret.name = column_name
    return ret, column_name


def weekofyear(
    x: pd.DataFrame | dd.DataFrame, value: str
) -> tuple[pd.Series | dd.Series, str]:
    """Calculate week of the year according to the ISO 8601 standard.
    Internal execution is `pandas.Series.dt.isocalendar` or `dask.dataframe.Series.dt.isocalendar`.

    Parameters
    ----------
    x : {pandas.DataFrame, dask.DataFrame}
        Target input.
    value : str
        Expressing the target column name.

    Returns
    -------
    {pandas.Series, dask.Series}
        Series on week of the year according to the ISO 8601 standard.
    list[str]
        List of column names.

    Examples
    --------
    >>> import pandas as pd
    >>> import sandbox.features.ts_calculator as tsc
    >>> x = pd.DataFrame({"x": pd.to_datetime(["2020-01-01 12:00:00", "2020-06-15 23:00:00",
    ...                                       "2021-02-12 01:00:00"])})
    >>> tsc.weekofyear(x, value="x")[0]
    0     1
    1    25
    2     6
    Name: x__weekofyear, dtype: int64
    >>> tsc.weekofyear(x, value="x")[1]
    'x__weekofyear'

    See Also
    --------
    pandas.Series.dt.isocalendar : The week of the year according to the ISO 8601 standard in pandas.
    dask.dataframe.Series.dt.isocalendar : The week of the year according to the ISO 8601 standard in dask.dataframe.
    """
    column_name = value + "__" + inspect.currentframe().f_code.co_name
    ret = x[value].dt.isocalendar().week.astype(int)
    ret.name = column_name
    return ret, column_name


@set_property("column_id", True)
def lag(
    x: pd.DataFrame | dd.DataFrame,
    value: str,
    params: dict,
    by: list[str] | None = None,
    sort: list[str] | None = None,
) -> tuple[pd.Series | dd.Series, str]:
    """Shift index by desired number of periods.

    Parameters
    ----------
    x : {pandas.DataFrame, dask.DataFrame}
        Target input.
    value : str
        Expressing the target column name.
    params : dict
        dict of the parameters required in the function.
        Required key and value are as follows:

        - 'lag' : int expressing the number of periods to shift. Can be positive or negative.

    by : {list[str], None}
        Used to determine the groups for the groupby.
    sort : {list[str], None}
        List of names to sort by. Sorting is supported for pandas.DataFrame, the if you
        use dask.DataFrame, you must sort the order of data in advance of executing the function.

    Returns
    -------
    {pandas.DataFrame, dask.DataFrame}
        Shifted input.
    list[str]
        List of column names.
    """
    ret = _sort_and_groupby(x, value, by, sort)

    dtype = x[value].dtype
    if dd and isinstance(ret, dd.groupby.SeriesGroupBy):
        ret = ret.shift(params["lag"], meta=(value, dtype)).compute().sort_index()
    else:
        ret = ret.shift(params["lag"])

    # naming
    column_base_name = value + "__" + inspect.currentframe().f_code.co_name
    name_suffix = "_".join(["{0}_{1}".format(k, v) for k, v in params.items()])
    column_name = column_base_name + "_" + name_suffix
    ret.name = column_name

    return ret, column_name


@set_property("column_id", True)
def rolling(
    x: pd.DataFrame | dd.DataFrame,
    value: str,
    params: dict,
    by: list[str] | None = None,
    sort: list[str] | None = None,
) -> tuple[pd.DataFrame | dd.DataFrame, str]:
    """Return the dataframe on rolling statistics.

    Parameters
    ----------
    x : {pandas.DataFrame, dask.DataFrame}
        Target input.
    value : str
        Expressing the target column name.
    params : dict
        dict of the parameters required in the function.
        Required key and value are as follows:

        - 'lag' : int expressing the number of periods to shift. Can be positive or negative.
        - 'window' : int expressing the fixed number of observations used for each window.
        - 'stats': list of str, which are chose in:

            - 'max': the maximum of the values.
            - 'mean': the mean of the values.
            - 'median': the median of the values.
            - 'min': the minimum of the values.
            - 'sum': the sum of the values.
            - 'std': sample standard deviation.
            - 'var': unbiased variance.

    by : {list[str], None}
        Used to determine the groups for the groupby.
    sort : {list[str], None}
        List of names to sort by. Sorting is supported for pandas.DataFrame, the if you
        use dask.DataFrame, you must sort the order of data in advance of executing the function.

    Returns
    -------
    {pandas.DataFrame, dask.DataFrame}
        Rolling statistics data.
    list[str]
        List of column names.
    """
    ret = _sort_and_groupby(x, value, by, sort)

    dtype = x[value].dtype
    if dd and isinstance(ret, dd.groupby.SeriesGroupBy):
        ret = (
            ret.shift(params["lag"], meta=(value, dtype))
            .rolling(params["window"])
            .agg(params["stats"])
            .compute()
            .sort_index()
        )
    else:
        ret = ret.shift(params["lag"]).rolling(params["window"]).agg(params["stats"])

    # naming
    column_base_name = value + "__" + inspect.currentframe().f_code.co_name
    suffix_list = list()
    for k, v in params.items():
        if k != "stats":
            suffix_list.append("{0}_{1}".format(k, v))
    name_suffix = "_".join(suffix_list)
    column_name = [
        column_base_name + "_" + s + "_" + name_suffix for s in params["stats"]
    ]
    ret.columns = column_name

    return ret, column_name


def _sort_and_groupby(
    x: pd.DataFrame | dd.DataFrame,
    value: str,
    by: list[str] | None = None,
    sort: list[str] | None = None,
):
    if sort:
        if dd and isinstance(x, dd.DataFrame):
            msg = (
                "In dask format, sorting the order of rows is unsupported, "
                "then please sort your dask data in advance."
            )
            raise NotImplementedError(msg)
        else:
            ret = x.sort_values(sort)
    else:
        ret = x

    if by is None:
        ret = ret[value]
    else:
        ret = ret.groupby(by)[value]

    return ret
