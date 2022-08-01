:py:mod:`sandbox.datamodel.ts_datamodel`
========================================

.. py:module:: sandbox.datamodel.ts_datamodel


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   sandbox.datamodel.ts_datamodel.TimeSeriesModelData




.. py:class:: TimeSeriesModelData(X, y=None, **kwargs)

   Bases: :py:obj:`sandbox.datamodel.base.SupervisedModelData`

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


