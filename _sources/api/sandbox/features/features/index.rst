:py:mod:`sandbox.features.features`
===================================

.. py:module:: sandbox.features.features


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   sandbox.features.features.FeaturesManager
   sandbox.features.features.FeaturesFactory



Functions
~~~~~~~~~

.. autoapisummary::

   sandbox.features.features.get_features



Attributes
~~~~~~~~~~

.. autoapisummary::

   sandbox.features.features.dd
   sandbox.features.features.DataFrame


.. py:data:: dd
   

   

.. py:data:: DataFrame
   

   

.. py:function:: get_features(data: DataFrame, column_values: list[str] | None = None, column_id: list[str] | None = None, sort_values: list[str] | None = None, func_params_list: list[dict] | None = None, params_path: str | None = None) -> DataFrame

   
   Get the dataframe with features columns.

   :param data:
   :type data: {pandas.DataFrame, dask.DataFrame}
   :param column_values:
   :type column_values: {list [str], None}
   :param column_id:
   :type column_id: {list[str], None}
   :param sort_values:
   :type sort_values: {list[str], None}
   :param func_params_list:
   :type func_params_list: {list[dict], None}
   :param params_path:
   :type params_path: {str, None}

   :returns: A dataframe with features columns whose format is in accordance with input `data`.
   :rtype: pandas.DataFrame or dask.DataFrame















   ..
       !! processed by numpydoc !!

.. py:class:: FeaturesManager(column_values: list[str] | None = None, column_id: list[str] | None = None, sort_values: list[str] | None = None, func_params_list: list[dict] | None = None, params_path: str | None = None)

   .. py:method:: features_parameters() -> dict[list | dict]
      :property:

      
















      ..
          !! processed by numpydoc !!


.. py:class:: FeaturesFactory(features_parameters: dict)

   .. py:method:: validate_input(data) -> None


   .. py:method:: create_features(data)


   .. py:method:: inspect_output(data)


   .. py:method:: process(data)



