:py:mod:`sandbox.tsa.dpregressor`
=================================

.. py:module:: sandbox.tsa.dpregressor

.. autoapi-nested-parse::

   The :mod:`sandbox.tsa.dpregressor` module includes classes and
   functions on the linear regressor models on deterministic process for time series.

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   sandbox.tsa.dpregressor.DeterministicProcessRegressor




.. py:class:: DeterministicProcessRegressor(level=True, trend=False, seasonal=None, freq_seasonal=None)

   Bases: :py:obj:`sklearn.linear_model.LinearRegression`, :py:obj:`sandbox.graphics.ts_grapher.TimeSeriesGrapherMixin`

   
   Ordinary least squares Linear Regression.

   LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
   to minimize the residual sum of squares between the observed targets in
   the dataset, and the targets predicted by the linear approximation.

   :param fit_intercept: Whether to calculate the intercept for this model. If set
                         to False, no intercept will be used in calculations
                         (i.e. data is expected to be centered).
   :type fit_intercept: bool, default=True
   :param normalize: This parameter is ignored when ``fit_intercept`` is set to False.
                     If True, the regressors X will be normalized before regression by
                     subtracting the mean and dividing by the l2-norm.
                     If you wish to standardize, please use
                     :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
                     on an estimator with ``normalize=False``.

                     .. deprecated:: 1.0
                        `normalize` was deprecated in version 1.0 and will be
                        removed in 1.2.
   :type normalize: bool, default=False
   :param copy_X: If True, X will be copied; else, it may be overwritten.
   :type copy_X: bool, default=True
   :param n_jobs: The number of jobs to use for the computation. This will only provide
                  speedup in case of sufficiently large problems, that is if firstly
                  `n_targets > 1` and secondly `X` is sparse or if `positive` is set
                  to `True`. ``None`` means 1 unless in a
                  :obj:`joblib.parallel_backend` context. ``-1`` means using all
                  processors. See :term:`Glossary <n_jobs>` for more details.
   :type n_jobs: int, default=None
   :param positive: When set to ``True``, forces the coefficients to be positive. This
                    option is only supported for dense arrays.

                    .. versionadded:: 0.24
   :type positive: bool, default=False

   .. attribute:: coef_

      Estimated coefficients for the linear regression problem.
      If multiple targets are passed during the fit (y 2D), this
      is a 2D array of shape (n_targets, n_features), while if only
      one target is passed, this is a 1D array of length n_features.

      :type: array of shape (n_features, ) or (n_targets, n_features)

   .. attribute:: rank_

      Rank of matrix `X`. Only available when `X` is dense.

      :type: int

   .. attribute:: singular_

      Singular values of `X`. Only available when `X` is dense.

      :type: array of shape (min(X, y),)

   .. attribute:: intercept_

      Independent term in the linear model. Set to 0.0 if
      `fit_intercept = False`.

      :type: float or array of shape (n_targets,)

   .. attribute:: n_features_in_

      Number of features seen during :term:`fit`.

      .. versionadded:: 0.24

      :type: int

   .. attribute:: feature_names_in_

      Names of features seen during :term:`fit`. Defined only when `X`
      has feature names that are all strings.

      .. versionadded:: 1.0

      :type: ndarray of shape (`n_features_in_`,)

   .. seealso::

      :obj:`Ridge`
          Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of the coefficients with l2 regularization.

      :obj:`Lasso`
          The Lasso is a linear model that estimates sparse coefficients with l1 regularization.

      :obj:`ElasticNet`
          Elastic-Net is a linear regression model trained with both l1 and l2 -norm regularization of the coefficients.

   .. rubric:: Notes

   From the implementation point of view, this is just plain Ordinary
   Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares
   (scipy.optimize.nnls) wrapped as a predictor object.

   .. rubric:: Examples

   >>> import numpy as np
   >>> from sklearn.linear_model import LinearRegression
   >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
   >>> # y = 1 * x_0 + 2 * x_1 + 3
   >>> y = np.dot(X, np.array([1, 2])) + 3
   >>> reg = LinearRegression().fit(X, y)
   >>> reg.score(X, y)
   1.0
   >>> reg.coef_
   array([1., 2.])
   >>> reg.intercept_
   3.0...
   >>> reg.predict(np.array([[3, 5]]))
   array([16.])















   ..
       !! processed by numpydoc !!
   .. py:method:: data_()
      :property:


   .. py:method:: get_deterministic_process(index)


   .. py:method:: deterministic_process_()
      :property:


   .. py:method:: fit(X, y=None, **kwargs)

      
      Fit linear model.

      :param X: Training data.
      :type X: {array-like, sparse matrix} of shape (n_samples, n_features)
      :param y: Target values. Will be cast to X's dtype if necessary.
      :type y: array-like of shape (n_samples,) or (n_samples, n_targets)
      :param sample_weight: Individual weights for each sample.

                            .. versionadded:: 0.17
                               parameter *sample_weight* support to LinearRegression.
      :type sample_weight: array-like of shape (n_samples,), default=None

      :returns: **self** -- Fitted Estimator.
      :rtype: object















      ..
          !! processed by numpydoc !!

   .. py:method:: predict(X)

      
      Predict using the linear model.

      :param X: Samples.
      :type X: array-like or sparse matrix, shape (n_samples, n_features)

      :returns: **C** -- Returns predicted values.
      :rtype: array, shape (n_samples,)















      ..
          !! processed by numpydoc !!

   .. py:method:: features_index_in_()
      :property:


   .. py:method:: fittedvalues_()
      :property:


   .. py:method:: components_name_()
      :property:


   .. py:method:: trend_()
      :property:


   .. py:method:: seasonal_()
      :property:


   .. py:method:: freq_seasonal_()
      :property:


   .. py:method:: trend_predicted_(X)


   .. py:method:: seasonal_predicted_(X)


   .. py:method:: freq_seasonal_predicted_(X)



