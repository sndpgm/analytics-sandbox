:py:mod:`sandbox.datamodel.base`
================================

.. py:module:: sandbox.datamodel.base


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   sandbox.datamodel.base.BaseData
   sandbox.datamodel.base.BaseModelData
   sandbox.datamodel.base.SupervisedModelData
   sandbox.datamodel.base.UnsupervisedModelData
   sandbox.datamodel.base.BaseDataSimulator



Functions
~~~~~~~~~

.. autoapisummary::

   sandbox.datamodel.base.get_1d_arr
   sandbox.datamodel.base.get_2d_arr



.. py:class:: BaseData(**kwargs)


.. py:class:: BaseModelData(X, y, **kwargs)

   Bases: :py:obj:`BaseData`

   Base class for data model of algorithm.

   :param X: Training data. In classification model, it is for classifying and clustering the data.
             In regression model, it is feature vectors or matrix, but can be ignored when the regression
             components are not defined in the case of time series analysis.
   :type X: array_like
   :param y: Target values. If algorithm is unsupervised, this should be ignored.
   :type y: array_like

   .. attribute:: X

      Training data.

      :type: numpy.ndarray

   .. attribute:: y

      Target values.

      :type: numpy.ndarray

   .. attribute:: orig_X

      Original X for input.

      :type: array_like

   .. attribute:: orig_y

      Original y for input.

      :type: array_like

   .. py:method:: nobs()
      :property:

      Number of observations.


   .. py:method:: common_index() -> pandas.Index
      :property:

      Common index of X and y


   .. py:method:: X_name() -> list[str] | None
      :property:

      X name columns


   .. py:method:: y_name() -> str | None
      :property:


   .. py:method:: convert_pandas()


   .. py:method:: convert_ndarray()



.. py:class:: SupervisedModelData(X, y=None, **kwargs)

   Bases: :py:obj:`BaseModelData`

   Base class for data model for supervised model.

   :param X: The feature vectors or matrix. If regression is not defined, you should
             handle the position of X as the one of y.
   :type X: array_like
   :param y: Target values. If regression is not defined, ignore that.
   :type y: {array_like, None}, optional

   .. attribute:: X

      Training data.

      :type: numpy.ndarray

   .. attribute:: y

      Target values.

      :type: numpy.ndarray

   .. attribute:: orig_X

      Original X for input.

      :type: array_like

   .. attribute:: orig_y

      Original y for input.

      :type: array_like

   .. py:method:: split_index_and_X_from_X_pred(X_pred)

      Split index and regression features design matrix
      from X_pred that is assumed to be data of predictive range.

      :param X_pred: Data to split into index and design matrix.
      :type X_pred: {array_like, int}

      :returns: * **index** (*pandas.Index*) -- Index split into.
                * **X** (*{numpy.ndarray, None}*) -- Design matrix split into.



.. py:class:: UnsupervisedModelData(X, **kwargs)

   Bases: :py:obj:`BaseModelData`

   Base class for data model for unsupervised model.

   :param X: Training data.
   :type X: array_like
   :param y: Ignored.
   :type y: Ignored

   .. attribute:: X

      Training data.

      :type: numpy.ndarray

   .. attribute:: orig_X

      Original X for input.

      :type: array_like


.. py:class:: BaseDataSimulator(seed=123456789, **kwargs)

   Base class for data simulator.


.. py:function:: get_1d_arr(obj, default_name='y')

   Get the module-standard 1-dimensional array from input.

   :param obj: Input data.
   :type obj: array_like
   :param default_name: Name of input data.
   :type default_name: str

   :returns: * **obj_arr** (*numpy.ndarray*) -- Converted array.
             * **obj_name** (*str*) -- Name.


.. py:function:: get_2d_arr(obj, default_name='x')

   Get the module-standard 2-dimensional array from input.

   :param obj: Input data.
   :type obj: array_like
   :param default_name: Name of input data.
   :type default_name: str

   :returns: * **obj_arr** (*numpy.ndarray*) -- Converted array.
             * **obj_name** (*list[str]*) -- Names.


