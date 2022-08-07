:py:mod:`sandbox.datamodel.ts_datamodel`
========================================

.. py:module:: sandbox.datamodel.ts_datamodel


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   sandbox.datamodel.ts_datamodel.TimeSeriesModelData




.. py:class:: TimeSeriesModelData(X, y=None)

   Bases: :py:obj:`sandbox.datamodel.base.SupervisedModelDataset`

   Base class for data model for supervised model.

   :param X: The feature vectors or matrix. If regression is not defined, you should
             handle the position of X as the one of y.
   :type X: StructuralDataType
   :param y: Target values. If regression is not defined, ignore that.
   :type y: {StructuralDataType, None}, optional


