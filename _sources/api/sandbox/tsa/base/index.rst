:py:mod:`sandbox.tsa.base`
==========================

.. py:module:: sandbox.tsa.base

.. autoapi-nested-parse::

   Base classes for time series estimators.



Module Contents
---------------

.. py:class:: _BaseModelMeta

   Bases: :py:obj:`abc.ABCMeta`

   Meta class for model estimators.

   .. py:method:: wrapper_fit(child_fit)
      :classmethod:

      Common pre-/post-process in fit method.


   .. py:method:: wrapper_predict(child_predict)
      :classmethod:

      Common pre-/post-process in predict method.



.. py:class:: BaseTimeSeriesModel

   Bases: :py:obj:`sklearn.base.BaseEstimator`

   Base class for time series model.

   .. py:method:: fit(X, y=None)
      :abstractmethod:

      Fit time series model.


   .. py:method:: predict(X)
      :abstractmethod:

      Predict using time series model.


   .. py:method:: conf_int(X, alpha)

      Construct confidence interval for the fitted parameters.


   .. py:method:: score(X, y, scorer='r2', **kwargs)

      Return the coefficient of determination of the prediction.



