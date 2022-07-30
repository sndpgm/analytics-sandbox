:py:mod:`sandbox.utils.validation`
==================================

.. py:module:: sandbox.utils.validation

.. autoapi-nested-parse::

   The validation module.



Module Contents
---------------

.. py:function:: is_dataframe(obj)

   Check whether object is pandas.DataFrame.

   :param obj: Input data
   :type obj: pandas.DataFrame

   :returns: **is_dataframe** -- If True, input is pandas.DataFrame.
   :rtype: bool

   .. rubric:: Examples

   >>> from sandbox.utils.validation import is_dataframe
   >>> import pandas as pd
   >>> obj1 = pd.DataFrame({"col1": [1] * 10})
   >>> is_dataframe_or_series(obj1)
   True
   >>> obj2 = 10
   >>> is_dataframe_or_series(obj2)
   False


.. py:function:: is_series(obj)

   Check whether object is pandas.Series.

   :param obj: Input data
   :type obj: pandas.Series

   :returns: **is_series** -- If True, input is pandas.Series.
   :rtype: bool

   .. rubric:: Examples

   >>> from sandbox.utils.validation import is_series
   >>> import pandas as pd
   >>> obj1 = pd.Series({"col1": [1] * 10})
   >>> is_series(obj1)
   True


.. py:function:: is_index(obj)

   Check whether object is pandas.Index.

   :param obj: Input data
   :type obj: pd.Index

   :returns: **is_index** -- Whether the object was pandas.Index.
   :rtype: bool

   .. rubric:: Examples

   >>> from sandbox.utils.validation import is_index
   >>> import pandas as pd
   >>> obj1 = pd.pd.RangeIndex(start=0, stop=10, step=1)
   >>> is_index(obj1)
   True


.. py:function:: is_dataframe_or_series(obj)

   Check whether object is pandas.DataFrame or pandas.Series.

   :param obj: Input data
   :type obj: pd.DataFrame or pd.Series

   :returns: **is_dataframe_or_series** -- Whether the object was pandas.DataFrame or pd.Series.
   :rtype: bool

   .. rubric:: Examples

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


.. py:function:: is_using_padnas(X, y)

   Whether both X and y are the class of pandas (DataFrame, Series).

   :param X: Input data for X
   :type X: pd.DataFrame or pd.Series
   :param y: Input data for y
   :type y: pd.DataFrame or pd.Series

   :returns: **is_using_pandas** -- If True, both X and y are the class of pandas (DataFrame, Series).
   :rtype: bool


.. py:function:: is_ndarray(obj)

   Check whether object is numpy.ndarray.
   :param obj: Input data
   :type obj: numpy.ndarray

   :returns: **is_series** -- If True, input is numpy.ndarray.
   :rtype: bool

   .. rubric:: Examples

   >>> from sandbox.utils.validation import is_ndarray
   >>> import numpy as np
   >>> obj1 = np.array([[1, 0], [0, 1]])
   >>> is_ndarray(obj1)
   True


.. py:function:: is_using_ndarray(X, y)

   Whether both X and y are the class of numpy (ndarray).

   :param X: Input data for X
   :type X: numpy.ndarray
   :param y: Input data for y
   :type y: numpy.ndarray

   :returns: **is_using_numpy** -- If True, both X and y are the class of numpy (ndarray).
   :rtype: bool


.. py:function:: is_arraylike(obj)

   Returns whether the input is array-like.

   :param obj: Input data
   :type obj: array-like

   :returns: **is_arraylike** -- Whether the object was array-like.
   :rtype: bool

   .. rubric:: Examples

   >>> from sandbox.utils.validation import is_arraylike
   >>> is_arraylike(["aa", "bb"])
   True

   >>> is_arraylike("cc")
   False


