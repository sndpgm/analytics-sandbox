:py:mod:`sandbox.features.ts_calculator`
========================================

.. py:module:: sandbox.features.ts_calculator


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   sandbox.features.ts_calculator.set_property
   sandbox.features.ts_calculator.hour
   sandbox.features.ts_calculator.dayofweek
   sandbox.features.ts_calculator.quarter
   sandbox.features.ts_calculator.month
   sandbox.features.ts_calculator.year
   sandbox.features.ts_calculator.dayofyear
   sandbox.features.ts_calculator.dayofmonth
   sandbox.features.ts_calculator.weekofyear
   sandbox.features.ts_calculator.lag
   sandbox.features.ts_calculator.rolling



Attributes
~~~~~~~~~~

.. autoapisummary::

   sandbox.features.ts_calculator.dd


.. py:data:: dd
   

   

.. py:function:: set_property(key, value)

   
   This method returns a decorator that sets the property key of the function to value
















   ..
       !! processed by numpydoc !!

.. py:function:: hour(x: pd.DataFrame | dd.DataFrame, value: str) -> tuple[pd.Series | dd.Series, str]

   
   The hours of the datetime.
   Internal execution is `pandas.Series.dt.hour` or `dask.dataframe.Series.dt.hour`.

   :param x: Target input.
   :type x: {pandas.DataFrame, dask.DataFrame}
   :param value: Expressing the target column name.
   :type value: str

   :returns: * *{pandas.Series, dask.Series}* -- Series on the hours of the datetime.
             * *list[str]* -- List of column names.

   .. rubric:: Examples

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

   .. seealso::

      :obj:`pandas.Series.dt.hour`
          The hours of the datetime in pandas.

      :obj:`dask.dataframe.Series.dt.hour`
          The hours of the datetime in dask.dataframe.















   ..
       !! processed by numpydoc !!

.. py:function:: dayofweek(x: pd.DataFrame | dd.DataFrame, value: str) -> tuple[pd.Series | dd.Series, str]

   
   The day of the week with Monday=0, Sunday=6.
   Internal execution is `pandas.Series.dt.dayofweek` or `dask.dataframe.Series.dt.dayofweek`.

   :param x: Target input.
   :type x: {pandas.DataFrame, dask.DataFrame}
   :param value: Expressing the target column name.
   :type value: str

   :returns: * *{pandas.Series, dask.Series}* -- Series on the day of the week of the datetime.
             * *list[str]* -- List of column names.

   .. rubric:: Examples

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

   .. seealso::

      :obj:`pandas.Series.dt.dayofweek`
          The day of the week in pandas.

      :obj:`dask.dataframe.Series.dt.dayofweek`
          The day of the week in dask.dataframe.















   ..
       !! processed by numpydoc !!

.. py:function:: quarter(x: pd.DataFrame | dd.DataFrame, value: str) -> tuple[pd.Series | dd.Series, str]

   
   The quarter of the date.
   Internal execution is `pandas.Series.dt.quarter` or `dask.dataframe.Series.dt.quarter`.

   :param x: Target input.
   :type x: {pandas.DataFrame, dask.DataFrame}
   :param value: Expressing the target column name.
   :type value: str

   :returns: * *{pandas.Series, dask.Series}* -- Series on the quarter of the date.
             * *list[str]* -- List of column names.

   .. rubric:: Examples

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

   .. seealso::

      :obj:`pandas.Series.dt.quarter`
          The quarter of the date in pandas.

      :obj:`dask.dataframe.Series.dt.quarter`
          The quarter of the date in dask.dataframe.















   ..
       !! processed by numpydoc !!

.. py:function:: month(x: pd.DataFrame | dd.DataFrame, value: str) -> tuple[pd.Series | dd.Series, str]

   
   The month of the date.
   Internal execution is `pandas.Series.dt.month` or `dask.dataframe.Series.dt.month`.

   :param x: Target input.
   :type x: {pandas.DataFrame, dask.DataFrame}
   :param value: Expressing the target column name.
   :type value: str

   :returns: * *{pandas.Series, dask.Series}* -- Series on the month of the date.
             * *list[str]* -- List of column names.

   .. rubric:: Examples

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

   .. seealso::

      :obj:`pandas.Series.dt.month`
          The month of the date in pandas.

      :obj:`dask.dataframe.Series.dt.month`
          The month of the date in dask.dataframe.















   ..
       !! processed by numpydoc !!

.. py:function:: year(x: pd.DataFrame | dd.DataFrame, value: str) -> tuple[pd.Series | dd.Series, str]

   
   The month of the date.
   Internal execution is `pandas.Series.dt.year` or `dask.dataframe.Series.dt.year`.

   :param x: Target input.
   :type x: {pandas.DataFrame, dask.DataFrame}
   :param value: Expressing the target column name.
   :type value: str

   :returns: * *{pandas.Series, dask.Series}* -- Series on the year of the date.
             * *list[str]* -- List of column names.

   .. rubric:: Examples

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

   .. seealso::

      :obj:`pandas.Series.dt.year`
          The year of the date in pandas.

      :obj:`dask.dataframe.Series.dt.year`
          The year of the date in dask.dataframe.















   ..
       !! processed by numpydoc !!

.. py:function:: dayofyear(x: pd.DataFrame | dd.DataFrame, value: str) -> tuple[pd.Series | dd.Series, str]

   
   The ordinal day of the year.
   Internal execution is `pandas.Series.dt.dayofyear` or `dask.dataframe.Series.dt.dayofyear`.

   :param x: Target input.
   :type x: {pandas.DataFrame, dask.DataFrame}
   :param value: Expressing the target column name.
   :type value: str

   :returns: * *{pandas.Series, dask.Series}* -- Series on the ordinal day of the year.
             * *list[str]* -- List of column names.

   .. rubric:: Examples

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

   .. seealso::

      :obj:`pandas.Series.dt.dayofyear`
          The ordinal day of the year in pandas.

      :obj:`dask.dataframe.Series.dt.dayofyear`
          The ordinal day of the year in dask.dataframe.















   ..
       !! processed by numpydoc !!

.. py:function:: dayofmonth(x: pd.DataFrame | dd.DataFrame, value: str) -> tuple[pd.Series | dd.Series, str]

   
   The day of the datetime.
   Internal execution is `pandas.Series.dt.day` or `dask.dataframe.Series.dt.day`.

   :param x: Target input.
   :type x: {pandas.DataFrame, dask.DataFrame}
   :param value: Expressing the target column name.
   :type value: str

   :returns: * *{pandas.Series, dask.Series}* -- Series on the day of the datetime.
             * *list[str]* -- List of column names.

   .. rubric:: Examples

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

   .. seealso::

      :obj:`pandas.Series.dt.day`
          The day of the datetime in pandas.

      :obj:`dask.dataframe.Series.dt.day`
          The day of the datetime in dask.dataframe.















   ..
       !! processed by numpydoc !!

.. py:function:: weekofyear(x: pd.DataFrame | dd.DataFrame, value: str) -> tuple[pd.Series | dd.Series, str]

   
   Calculate week of the year according to the ISO 8601 standard.
   Internal execution is `pandas.Series.dt.isocalendar` or `dask.dataframe.Series.dt.isocalendar`.

   :param x: Target input.
   :type x: {pandas.DataFrame, dask.DataFrame}
   :param value: Expressing the target column name.
   :type value: str

   :returns: * *{pandas.Series, dask.Series}* -- Series on week of the year according to the ISO 8601 standard.
             * *list[str]* -- List of column names.

   .. rubric:: Examples

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

   .. seealso::

      :obj:`pandas.Series.dt.isocalendar`
          The week of the year according to the ISO 8601 standard in pandas.

      :obj:`dask.dataframe.Series.dt.isocalendar`
          The week of the year according to the ISO 8601 standard in dask.dataframe.















   ..
       !! processed by numpydoc !!

.. py:function:: lag(x: pd.DataFrame | dd.DataFrame, value: str, params: dict, by: list[str] | None = None, sort: list[str] | None = None) -> tuple[pd.Series | dd.Series, str]

   
   Shift index by desired number of periods.

   :param x: Target input.
   :type x: {pandas.DataFrame, dask.DataFrame}
   :param value: Expressing the target column name.
   :type value: str
   :param params: dict of the parameters required in the function.
                  Required key and value are as follows:

                  - 'lag' : int expressing the number of periods to shift. Can be positive or negative.
   :type params: dict
   :param by: Used to determine the groups for the groupby.
   :type by: {list[str], None}
   :param sort: List of names to sort by. Sorting is supported for pandas.DataFrame, the if you
                use dask.DataFrame, you must sort the order of data in advance of executing the function.
   :type sort: {list[str], None}

   :returns: * *{pandas.DataFrame, dask.DataFrame}* -- Shifted input.
             * *list[str]* -- List of column names.















   ..
       !! processed by numpydoc !!

.. py:function:: rolling(x: pd.DataFrame | dd.DataFrame, value: str, params: dict, by: list[str] | None = None, sort: list[str] | None = None) -> tuple[pd.DataFrame | dd.DataFrame, str]

   
   Return the dataframe on rolling statistics.

   :param x: Target input.
   :type x: {pandas.DataFrame, dask.DataFrame}
   :param value: Expressing the target column name.
   :type value: str
   :param params: dict of the parameters required in the function.
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
   :type params: dict
   :param by: Used to determine the groups for the groupby.
   :type by: {list[str], None}
   :param sort: List of names to sort by. Sorting is supported for pandas.DataFrame, the if you
                use dask.DataFrame, you must sort the order of data in advance of executing the function.
   :type sort: {list[str], None}

   :returns: * *{pandas.DataFrame, dask.DataFrame}* -- Rolling statistics data.
             * *list[str]* -- List of column names.















   ..
       !! processed by numpydoc !!

