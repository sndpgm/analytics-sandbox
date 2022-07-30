:py:mod:`sandbox.datamodel.base`
================================

.. py:module:: sandbox.datamodel.base


Module Contents
---------------

.. py:class:: BaseData(**kwargs)

   .. py:method:: _get_1d_arr(obj, default_name='y')
      :staticmethod:


   .. py:method:: _get_2d_arr(obj, default_name='x')
      :staticmethod:


   .. py:method:: _get_index(obj)
      :staticmethod:


   .. py:method:: _as_pandas_from_ndarray(obj, index, key, pandas_type='dataframe')
      :staticmethod:


   .. py:method:: _as_ndarray_from_pandas(obj)
      :staticmethod:



.. py:class:: BaseModelData(X, y, **kwargs)

   Bases: :py:obj:`BaseData`

   .. py:method:: _both_X_y_are_none(X, y)
      :staticmethod:


   .. py:method:: _check_X_or_y_is_not_none(X, y)


   .. py:method:: _check_X_y_length(X, y)
      :staticmethod:


   .. py:method:: nobs()
      :property:


   .. py:method:: _get_index_from_X_and_y()


   .. py:method:: common_index() -> pandas.Index
      :property:


   .. py:method:: X_name() -> list[str] | None
      :property:


   .. py:method:: y_name() -> str | None
      :property:


   .. py:method:: _convert_pandas()


   .. py:method:: convert_pandas()


   .. py:method:: _convert_ndarray()


   .. py:method:: convert_ndarray()



.. py:class:: SupervisedModelData(X, y=None, **kwargs)

   Bases: :py:obj:`BaseModelData`

   .. py:method:: split_index_and_X_from_X_pred(X_pred)



.. py:class:: UnsupervisedModelData(X, **kwargs)

   Bases: :py:obj:`BaseModelData`


.. py:class:: BaseDataSimulator(seed=123456789, **kwargs)


.. py:function:: _get_1d_arr(obj, default_name='y')


.. py:function:: _get_2d_arr(obj, default_name='x')


.. py:function:: check_for_compatible_data(*data)


