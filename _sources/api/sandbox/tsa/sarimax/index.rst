:py:mod:`sandbox.tsa.sarimax`
=============================

.. py:module:: sandbox.tsa.sarimax

.. autoapi-nested-parse::

   SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables).



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   sandbox.tsa.sarimax.SARIMAXModel




.. py:class:: SARIMAXModel(trend=None, s=1, seasonal=True, method='lbfgs', start_p=2, d=None, start_q=2, max_p=5, max_d=2, max_q=5, start_P=1, D=None, start_Q=1, max_P=2, max_D=1, max_Q=2, stepwise=True, max_order=5, n_jobs=1, trace=False)

   Bases: :py:obj:`sandbox.tsa.base.BaseTimeSeriesModel`, :py:obj:`sandbox.graphics.ts_grapher.TimeSeriesGrapherMixin`

   Linear Gaussian state space model.

   :param trend: Parameter controlling the deterministic trend polynomial :math:`A(t)`.
                 Can be specified as a string where 'c' indicates a constant (i.e. a
                 degree zero component of the trend polynomial), 't' indicates a
                 linear trend with time, and 'ct' is both. Can also be specified as an
                 iterable defining the non-zero polynomial exponents to include, in
                 increasing order. For example, `[1,1,0,1]` denotes
                 :math:`a + bt + ct^3`. Default is to not include a trend component.
   :type trend: str{'n','c','t','ct'} or iterable, optional
   :param s: The period for seasonal differencing, ``s`` refers to the number of
             periods in each season. For example, ``s`` is 4 for quarterly data, 12
             for monthly data, or 1 for annual (non-seasonal) data. Default is 1.
             Note that if ``s`` == 1 (i.e., is non-seasonal), ``seasonal`` will be
             set to False.
   :type s: int, optional
   :param seasonal: Whether to fit a seasonal ARIMA. Default is True. Note that if
                    ``seasonal`` is True and ``s`` == 1, ``seasonal`` will be set to False.
   :type seasonal: bool, optional
   :param method: The ``method`` determines which solver from ``scipy.optimize``
                  is used, and it can be chosen from among the following strings:
                  - 'newton' for Newton-Raphson
                  - 'nm' for Nelder-Mead
                  - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
                  - 'lbfgs' for limited-memory BFGS with optional box constraints
                  - 'powell' for modified Powell's method
                  - 'cg' for conjugate gradient
                  - 'ncg' for Newton-conjugate gradient
                  - 'basinhopping' for global basin-hopping solver
                  The explicit arguments in ``fit`` are passed to the solver,
                  except for the basin-hopping solver. Each
                  solver has several optional arguments that are not the same across
                  solvers. These can be passed as **fit_kwargs
   :type method: str, optional
   :param start_p: The starting value of ``p``, the order (or number of time lags)
                   of the auto-regressive ("AR") model. Must be a positive integer.
   :type start_p: int, optional
   :param d: The order of first-differencing. If None (by default), the value
             will automatically be selected based on the results of the ``test``
             (i.e., either the Kwiatkowski–Phillips–Schmidt–Shin, Augmented
             Dickey-Fuller or the Phillips–Perron test will be conducted to find
             the most probable value). Must be a positive integer or None. Note
             that if ``d`` is None, the runtime could be significantly longer.
   :type d: int, optional
   :param start_q: The starting value of ``q``, the order of the moving-average
                   ("MA") model. Must be a positive integer.
   :type start_q: int, optional
   :param max_p: The maximum value of ``p``, inclusive. Must be a positive integer
                 greater than or equal to ``start_p``.
   :type max_p: int, optional
   :param max_d: The maximum value of ``d``, or the maximum number of non-seasonal
                 differences. Must be a positive integer greater than or equal to ``d``.
   :type max_d: int, optional
   :param max_q: The maximum value of ``q``, inclusive. Must be a positive integer
                 greater than ``start_q``.
   :type max_q: int, optional
   :param start_P: The starting value of ``P``, the order of the autoregressive portion
                   of the seasonal model.
   :type start_P: int, optional
   :param D: The order of the seasonal differencing. If None (by default, the value
             will automatically be selected. Must be a positive integer or None.
   :type D: int, optional
   :param start_Q: The starting value of ``Q``, the order of the moving-average portion
                   of the seasonal model.
   :type start_Q: int, optional
   :param max_P: The maximum value of ``P``, inclusive. Must be a positive integer
                 greater than ``start_P``.
   :type max_P: int, optional
   :param max_D: The maximum value of ``D``. Must be a positive integer greater
                 than ``D``.
   :type max_D: int, optional
   :param max_Q: The maximum value of ``Q``, inclusive. Must be a positive integer
                 greater than ``start_Q``.
   :type max_Q: int, optional
   :param stepwise: Whether to use the stepwise algorithm outlined in [1]_ Hyndman and Khandakar
                    (2008) to identify the optimal model parameters. The stepwise algorithm
                    can be significantly faster than fitting all hyperparameter combinations
                    and is less likely to over-fit the model.
   :type stepwise: bool, optional
   :param max_order: Maximum value of :math:`p+q+P+Q` if model selection is not stepwise.
                     If the sum of ``p`` and ``q`` is >= ``max_order``, a model will
                     *not* be fit with those parameters, but will progress to the next
                     combination. Default is 5. If ``max_order`` is None, it means there
                     are no constraints on maximum order.
   :type max_order: int, optional
   :param n_jobs: The number of models to fit in parallel in the case of a grid search
                  (``stepwise=False``). Default is 1, but -1 can be used to designate
                  "as many as possible".
   :type n_jobs: int, optional
   :param trace: Whether to print status on the fits. A value of False will print no
                 debugging information. A value of True will print some. Integer values
                 exceeding 1 will print increasing amounts of debug information at each
                 fit.
   :type trace: {bool, int}, optional

   .. rubric:: Examples

   >>> from sklearn.model_selection import train_test_split
   >>> from sandbox.datasets import air_passengers
   >>> from sandbox.tsa.sarimax import SARIMAXModel
   >>> # Get test data
   >>> y = air_passengers.load().data
   >>> y_train, y_test = train_test_split(y, test_size=0.20, shuffle=False)
   >>> # Build model and fitting
   >>> sarimax = SARIMAXModel(trend="c", s=12, trace=True)
   >>> sarimax.fit(y_train)
   Out[1]: SARIMAXModel(s=12, trace=True, trend='c')
   >>> # Predict
   >>> sarimax.predict(y_test.index)
   Out[2]:
   array([490.57261133, 428.30635863, 371.39237605, 329.61111601,
          360.80266057, 364.99977758, 343.19575277, 387.39196193,
          373.58812314, 388.78429418, 460.98046321, 517.17663265,
          516.94541333, 454.87533   , 398.15751679, 356.57242612,
          387.96014005, 392.35342643, 370.74557099, 415.13794951,
          401.5302801 , 416.9226205 , 489.3149589 , 545.70729771,
          545.67224776, 483.7983338 , 427.27668995, 385.88776865,
          417.47165195])

   .. rubric:: Notes

   The SARIMA model is specified :math:`(p, d, q) \times (P, D, Q)_s`.

   .. math::

       \phi_p (L) \tilde \phi_P (L^s) \Delta^d \Delta_s^D y_t = A(t) +
           \theta_q (L) \tilde \theta_Q (L^s) \zeta_t

   In terms of a univariate structural model, this can be represented as

   .. math::

       y_t & = u_t + \eta_t \\
       \phi_p (L) \tilde \phi_P (L^s) \Delta^d \Delta_s^D u_t & = A(t) +
           \theta_q (L) \tilde \theta_Q (L^s) \zeta_t

   where :math:`\eta_t` is only applicable in the case of measurement error
   (although it is also used in the case of a pure regression model, i.e. if
   p=q=0).

   In terms of this model, regression with SARIMA errors can be represented
   easily as

   .. math::

       y_t & = \beta_t x_t + u_t \\
       \phi_p (L) \tilde \phi_P (L^s) \Delta^d \Delta_s^D u_t & = A(t) +
           \theta_q (L) \tilde \theta_Q (L^s) \zeta_t

   this model is the one used when exogenous regressors are provided.
   Note that the reduced form lag polynomials will be written as:

   .. math::

       \Phi (L) \equiv \phi_p (L) \tilde \phi_P (L^s) \\
       \Theta (L) \equiv \theta_q (L) \tilde \theta_Q (L^s)

   .. seealso:: :obj:`statsmodels.tsa.statespace.sarimax.SARIMAX`, :obj:`pmdarima.arima.ARIMA`, :obj:`pmdarima.arima.AutoARIMA`

   .. rubric:: References

   .. [1] Hyndman, R. J., & Khandakar, Y. (2008).
          Automatic time series forecasting: the forecast package for R.
          Journal of statistical software, 27, 1-22.

   .. py:method:: fit(X, y=None, **kwargs)

      Fit the model.

      :param X: Training data on regressions. If no regression is defined,
                just y is to be defined.
      :type X: array_like
      :param y: Target values. If no regression is defined, just y is to be
                defined in the place of X.
      :type y: {array_like, None}, default

      :returns: **self** -- Returns the instance itself.
      :rtype: object


   .. py:method:: has_model_result()

      Whether an instance has ``model_result_``.

      Some method needs ``model_result_`` that can be gained after
      :py:func:`fit <sandbox.tsa.sarimaxSARIMAXModel.fit>`.

      :returns: **result** -- If an instance has ``model_result_``, True. Otherwise, False.
      :rtype: bool


   .. py:method:: estimated_params_()
      :property:

      Estimated parameters.

      :py:class:`SARIMAXModel <sandbox.tsa.sarimax.SARIMAXModel>` estimates (1) regression
      , (2) autoregressive, (3) moving average, (4) seasonal autoregressive, (5) seasonal
      moving average coefficients and (6) variance of noise.

      :returns: **estimated_params** -- The estimated parameters.
      :rtype: dict


   .. py:method:: fittedvalues_()
      :property:

      The fitted values of the model.

      :returns: **fittedvalues** -- The fitted values to be estimated.
      :rtype: numpy.ndarray


   .. py:method:: predict(X, is_pandas=False)

      Predict using the model.

      :param X: Design matrix expressing the regression dummies or variables in
                the period to be predicted. If no regression is defined in the model,
                the index expressing the period or the period steps to be predicted
                must be set.
      :type X: {array-like, int}
      :param is_pandas: If True, the return data type is pandas.Series. Otherwise, numpy.ndarray.
      :type is_pandas: bool, optional

      :returns: **predicted_mean** -- Mean of predictive distribution of query points.
      :rtype: array-like


   .. py:method:: conf_int(X, alpha=0.95, is_pandas=False)

      Compute the confidence interval.

      :param X: Design matrix expressing the regression dummies or variables in
                the period to be predicted. If no regression is defined in the model,
                the index expressing the period or the period steps to be predicted
                must be set.
      :type X: {array-like, int}
      :param alpha: The `alpha` level for the confidence interval. The default
                    `alpha` = .95 returns a 95% confidence interval.
      :type alpha: float, optional
      :param is_pandas: If True, the return data type is pandas.Series. Otherwise, numpy.ndarray.
      :type is_pandas: bool, optional

      :returns: The confidence intervals.
      :rtype: array_like


   .. py:method:: score(X, y, scorer='r2')

      Return the coefficient of determination of the prediction.

      The default coefficient of determination :math:`R^2` is defined as
      :math:`(1 - \frac{u}{v})`, where :math:`u` is the residual
      sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
      is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
      The best possible score is 1.0, and it can be negative (because the
      model can be arbitrarily worse). A constant model that always predicts
      the expected value of `y`, disregarding the input features, would get
      a :math:`R^2` score of 0.0.

      :param X: Design matrix expressing the regression dummies or variables in
                the period to be predicted. If no regression is defined in the model,
                the index expressing the period or the period steps to be predicted
                must be set.
      :type X: {array-like, int}
      :param y: True values for `X`.
      :type y: array-like
      :param scorer: Expressing the type of the coefficient of determination.
      :type scorer: str, optional

      :returns: **score** -- :math:`R^2` of ``self.predict(X)``.
      :rtype: float


   .. py:method:: components_name_()
      :property:

      Return component names.

      Although SARIMAX model has no state parameter, present here for API
      consistency.



