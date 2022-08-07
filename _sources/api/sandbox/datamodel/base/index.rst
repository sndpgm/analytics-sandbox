:py:mod:`sandbox.datamodel.base`
================================

.. py:module:: sandbox.datamodel.base


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   sandbox.datamodel.base.BaseData
   sandbox.datamodel.base.BaseModelDataset
   sandbox.datamodel.base.SupervisedModelDataset
   sandbox.datamodel.base.UnsupervisedModelDataset
   sandbox.datamodel.base.BaseDataSimulator




Attributes
~~~~~~~~~~

.. autoapisummary::

   sandbox.datamodel.base.StructuralDataType


.. py:data:: StructuralDataType
   

   

.. py:class:: BaseData(data: StructuralDataType)

   Base data class.

   :param data: Input data. Supported format is `pandas.DataFrame`, `pandas.Series`, `pandas.Index`, `numpy.ndarray`,
                `dask.dataframe.DataFrame`, `dask.dataframe.Series`, `dask.dataframe.Index`, `dask.array.Array`.
   :type data: StructuralDataType

   .. warning::

      In case of `Pandas` and `NumPy` format, :py:attr:`values <sandbox.datamodel.base.BaseData.values>`
      returns the actual data. However, the format of `Dask` returns before-compute objects, and if you want
      to get the actual data, you need to :py:func:`compute <dask.dataframe.DataFrame.compute>`.

   .. py:method:: __repr__() -> str

      Return repr(self).


   .. py:method:: __len__() -> int


   .. py:method:: nobs() -> int
      :property:

      Number of observations.


   .. py:method:: nparams() -> int
      :property:

      Number of parameters.


   .. py:method:: values() -> Union[numpy.ndarray, dask.array.Array]
      :property:

      Return a Numpy representation of data.
      In case of Dask format, return a Dask.array.Array.


   .. py:method:: index() -> Union[pandas.Index, dask.dataframe.Index]
      :property:

      Return the index (row labels) of data.


   .. py:method:: names() -> pandas.Index
      :property:

      Returns the column labels of data.


   .. py:method:: shape() -> tuple[int, int]
      :property:

      Return a tuple representing the dimensionality of data.


   .. py:method:: to_pandas() -> Union[pandas.DataFrame, pandas.Series, pandas.Index]

      Convert the BaseData to Pandas dataframe.

      :rtype: {pandas.DataFrame, pandas.Series, pandas.Index}


   .. py:method:: to_numpy() -> numpy.ndarray

      Convert the BaseData to NumPy array.

      :rtype: numpy.ndarray


   .. py:method:: to_dask_dataframe(**from_pandas_kwargs) -> Union[dask.dataframe.DataFrame, dask.dataframe.Series, dask.dataframe.Index]

      Convert the BaseData to Dask dataframe.

      :param from_pandas_kwargs: :py:func:`from_pandas <dask.dataframe.from_pandas>` in Dask converts data, and `from_pandas_kwargs`
                                 is the argument which is used in the function.
      :type from_pandas_kwargs: dict

      :rtype: {dask.dataframe.DataFrame, dask.dataframe.Series, dask.dataframe.Index}

      .. seealso:: :obj:`dask.dataframe.from_pandas`


   .. py:method:: to_dask_numpy(**from_array_kwargs) -> dask.array.Array

      Convert the BaseData to Dask array.

      :param from_array_kwargs: :py:func:`from_array <dask.array.from_array>` in Dask converts data, and `from_array_kwargs`
                                is the argument which is used in the function.
      :type from_array_kwargs: dict

      :rtype: dask.array.Array

      .. seealso:: :obj:`dask.array.from_array`



.. py:class:: BaseModelDataset(X, y)

   Base class for data model of algorithm.

   :param X: Training data. In classification model, it is for classifying and clustering the data.
             In regression model, it is feature vectors or matrix, but can be ignored when the regression
             components are not defined in the case of time series analysis.
   :type X: StructuralDataType
   :param y: Target values. If algorithm is unsupervised, this should be ignored.
   :type y: StructuralDataType

   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: nobs()
      :property:

      Number of observations.


   .. py:method:: nfeatures()
      :property:

      Number of feature variables.


   .. py:method:: common_index()
      :property:

      Common index of X and y


   .. py:method:: X_name()
      :property:

      X name columns


   .. py:method:: y_name()
      :property:

      y name.



.. py:class:: SupervisedModelDataset(X, y=None)

   Bases: :py:obj:`BaseModelDataset`

   Base class for data model for supervised model.

   :param X: The feature vectors or matrix. If regression is not defined, you should
             handle the position of X as the one of y.
   :type X: StructuralDataType
   :param y: Target values. If regression is not defined, ignore that.
   :type y: {StructuralDataType, None}, optional

   .. py:method:: get_index_and_values_from_X_pred(X_pred)

      Get index and features design matrix from X_pred
      that is assumed to be data of predictive range.

      :param X_pred: Data to split into index and design matrix.
      :type X_pred: {array_like, int}

      :returns: * **index** (*pandas.Index*) -- Index split into.
                * **X** (*{numpy.ndarray, None}*) -- Design matrix split into.



.. py:class:: UnsupervisedModelDataset(X, y)

   Bases: :py:obj:`BaseModelDataset`

   Base class for data model of algorithm.

   :param X: Training data. In classification model, it is for classifying and clustering the data.
             In regression model, it is feature vectors or matrix, but can be ignored when the regression
             components are not defined in the case of time series analysis.
   :type X: StructuralDataType
   :param y: Target values. If algorithm is unsupervised, this should be ignored.
   :type y: StructuralDataType


.. py:class:: BaseDataSimulator(seed=123456789, **kwargs)

   Base class for data simulator.


