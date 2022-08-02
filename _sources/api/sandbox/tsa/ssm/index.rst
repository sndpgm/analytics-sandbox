:py:mod:`sandbox.tsa.ssm`
=========================

.. py:module:: sandbox.tsa.ssm

.. autoapi-nested-parse::

   State space model.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   sandbox.tsa.ssm.LinearGaussianStateSpaceModel




.. py:class:: LinearGaussianStateSpaceModel(level=False, trend=False, seasonal=None, freq_seasonal=None, cycle=False, autoregressive=None, irregular=False, stochastic_level=False, stochastic_trend=False, stochastic_seasonal=True, stochastic_freq_seasonal=None, stochastic_cycle=False, damped_cycle=False, cycle_period_bounds=None, mle_regression=True, use_exact_diffuse=False)

   Bases: :py:obj:`sandbox.tsa.base.BaseTimeSeriesModel`, :py:obj:`sandbox.graphics.ts_grapher.TimeSeriesGrapherMixin`

   Linear Gaussian state space model.

   :param level: Whether to include a level component. Default is False.
   :type level: bool, optional
   :param trend: Whether to include a trend component. Default is False. If True,
                 `level` must also be True.
   :type trend: bool, optional
   :param seasonal: The period of the seasonal component, if any. Default is None.
   :type seasonal: {int, None}, optional
   :param freq_seasonal: Whether (and how) to model seasonal component(s) with trig. functions.
                         If specified, there is one dictionary for each frequency-domain
                         seasonal component.  Each dictionary must have the key, value pair for
                         'period' -- integer and may have a key, value pair for
                         'harmonics' -- integer. If 'harmonics' is not specified in any of the
                         dictionaries, it defaults to the floor of period/2.
   :type freq_seasonal: {list[dict], None}, optional.
   :param cycle: Whether to include a cycle component. Default is False.
   :type cycle: bool, optional
   :param autoregressive: The order of the autoregressive component. Default is None.
   :type autoregressive: {int, None}, optional
   :param irregular: Whether to include an irregular component. Default is False.
   :type irregular: bool, optional
   :param stochastic_level: Whether any level component is stochastic. Default is False.
   :type stochastic_level: bool, optional
   :param stochastic_trend: Whether any trend component is stochastic. Default is False.
   :type stochastic_trend: bool, optional
   :param stochastic_seasonal: Whether any seasonal component is stochastic. Default is True.
   :type stochastic_seasonal: bool, optional
   :param stochastic_freq_seasonal: Whether each seasonal component(s) is (are) stochastic.  Default
                                    is True for each component.  The list should be of the same length as
                                    freq_seasonal.
   :type stochastic_freq_seasonal: list[bool], optional
   :param stochastic_cycle: Whether any cycle component is stochastic. Default is False.
   :type stochastic_cycle: bool, optional
   :param damped_cycle: Whether the cycle component is damped. Default is False.
   :type damped_cycle: bool, optional
   :param cycle_period_bounds: A tuple with lower and upper allowed bounds for the period of the
                               cycle. If not provided, the following default bounds are used:
                               (1) if no date / time information is provided, the frequency is
                               constrained to be between zero and :math:`\pi`, so the period is
                               constrained to be in [0.5, infinity].
                               (2) If the date / time information is provided, the default bounds
                               allow the cyclical component to be between 1.5 and 12 years; depending
                               on the frequency of the endogenous variable, this will imply different
                               specific bounds.
   :type cycle_period_bounds: tuple, optional
   :param mle_regression: Whether to estimate regression coefficients by maximum likelihood
                          as one of hyperparameters. Default is True.
                          If False, the regression coefficients are estimated by recursive OLS,
                          included in the state vector.
   :type mle_regression: bool, optional
   :param use_exact_diffuse: Whether to use exact diffuse initialization for non-stationary
                             states. Default is False (in which case approximate diffuse
                             initialization is used).
   :type use_exact_diffuse: bool, optional

   .. rubric:: Examples

   >>> from sandbox.datamodel.ts_simulator import UnobservedComponentsSimulator
   >>> from sandbox.tsa.ssm import LinearGaussianStateSpaceModel
   >>> # Simulation data
   >>> sim = UnobservedComponentsSimulator(
   >>>     steps=400,
   >>>     level=True,
   >>>     trend=True,
   >>>     freq_seasonal=[{"period": 50, "harmonics": 4}, {"period": 100, "harmonics": 6}],
   >>>     exog_params=[5, ],
   >>>     start_param_level=10,
   >>>     stddev_level=0.001,
   >>>     stddev_trend=0.01,
   >>>     stddev_freq_seasonal=[0.01, 0.01],
   >>> )
   >>> ret = sim.simulate()
   >>> # Split data
   >>> X_train, X_test, y_train, y_test = train_test_split(ret.exog, ret.endog, test_size=0.10, shuffle=False)
   >>> model = LinearGaussianStateSpaceModel(
   >>>     level=True,
   >>>     trend=True,
   >>>     freq_seasonal=[{"period": 12, "harmonics": 4}, {"period": 100, "harmonics": 6}],
   >>> )
   >>> model.fit(X_train, y_train)
   Out[1]:
   LinearGaussianStateSpaceModel(freq_seasonal=[{'harmonics': 4, 'period': 12},
                                                {'harmonics': 6, 'period': 100}],
                                 level=True, trend=True)
   >>> model.score(X_test, y_test)
   Out[2]: 0.9834446210596552

   .. rubric:: Notes

   These models take the general form (see [1]_ Chapter 3.2 for all details)

   .. math::

       y_t = \mu_t + \gamma_t + c_t + \varepsilon_t

   where :math:`y_t` refers to the observation vector at time :math:`t`,
   :math:`\mu_t` refers to the trend component, :math:`\gamma_t` refers to the
   seasonal component, :math:`c_t` refers to the cycle, and
   :math:`\varepsilon_t` is the irregular. The modeling details of these
   components are given below.

   **Trend**

   The trend component is a dynamic extension of a regression model that
   includes an intercept and linear time-trend. It can be written:

   .. math::

       \mu_t &= \mu_{t-1} + \beta_{t-1} + \eta_{t-1} \\
       \beta_t &= \beta_{t-1} + \zeta_{t-1}

   where the level is a generalization of the intercept term that can
   dynamically vary across time, and the trend is a generalization of the
   time-trend such that the slope can dynamically vary across time.

   Here :math:`\eta_t \sim N(0, \sigma_\eta^2)` and
   :math:`\zeta_t \sim N(0, \sigma_\zeta^2)`.

   For both elements (level and trend), we can consider models in which:

   - The element is included vs excluded (if the trend is included, there must
     also be a level included).
   - The element is deterministic vs stochastic (i.e. whether or not the
     variance on the error term is confined to be zero or not)

   The only additional parameters to be estimated via MLE are the variances of
   any included stochastic components.

   **Seasonal (Time-domain)**

   The seasonal component is modeled as:

   .. math::

       \gamma_t = - \sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_t \\
       \omega_t \sim N(0, \sigma_\omega^2)

   The periodicity (number of seasons) is s, and the defining character is
   that (without the error term), the seasonal components sum to zero across
   one complete cycle. The inclusion of an error term allows the seasonal
   effects to vary over time (if this is not desired, :math:`\sigma_\omega^2`
   can be set to zero using the `stochastic_seasonal=False` keyword argument).

   This component results in one parameter to be selected via maximum
   likelihood: :math:`\sigma_\omega^2`, and one parameter to be chosen, the
   number of seasons `s`.

   Following the fitting of the model, the unobserved seasonal component
   time series is available in the results class in the `seasonal`
   attribute.

   **Frequency-domain Seasonal**

   Each frequency-domain seasonal component is modeled as:

   .. math::

       \gamma_t & =  \sum_{j=1}^h \gamma_{j, t} \\
       \gamma_{j, t+1} & = \gamma_{j, t}\cos(\lambda_j)
                       + \gamma^{*}_{j, t}\sin(\lambda_j) + \omega_{j,t} \\
       \gamma^{*}_{j, t+1} & = -\gamma^{(1)}_{j, t}\sin(\lambda_j)
                           + \gamma^{*}_{j, t}\cos(\lambda_j)
                           + \omega^{*}_{j, t}, \\
       \omega^{*}_{j, t}, \omega_{j, t} & \sim N(0, \sigma_{\omega^2}) \\
       \lambda_j & = \frac{2 \pi j}{s}

   where j ranges from 1 to h.

   The periodicity (number of "seasons" in a "year") is s and the number of
   harmonics is h.  Note that h is configurable to be less than s/2, but
   s/2 harmonics is sufficient to fully model all seasonal variations of
   periodicity s.  Like the time domain seasonal term (cf. Seasonal section,
   above), the inclusion of the error terms allows for the seasonal effects to
   vary over time.  The argument stochastic_freq_seasonal can be used to set
   one or more of the seasonal components of this type to be non-random,
   meaning they will not vary over time.

   This component results in one parameter to be fitted using maximum
   likelihood: :math:`\sigma_{\omega^2}`, and up to two parameters to be
   chosen, the number of seasons s and optionally the number of harmonics
   h, with :math:`1 \leq h \leq \lfloor s/2 \rfloor`.

   After fitting the model, each unobserved seasonal component modeled in the
   frequency domain is available in the results class in the `freq_seasonal`
   attribute.

   **Cycle**

   The cyclical component is intended to capture cyclical effects at time
   frames much longer than captured by the seasonal component. For example,
   in economics the cyclical term is often intended to capture the business
   cycle, and is then expected to have a period between "1.5 and 12 years"
   (see Durbin and Koopman).

   .. math::

       c_{t+1} & = \rho_c (\tilde c_t \cos \lambda_c t
               + \tilde c_t^* \sin \lambda_c) +
               \tilde \omega_t \\
       c_{t+1}^* & = \rho_c (- \tilde c_t \sin \lambda_c  t +
               \tilde c_t^* \cos \lambda_c) +
               \tilde \omega_t^* \\

   where :math:`\omega_t, \tilde \omega_t iid N(0, \sigma_{\tilde \omega}^2)`

   The parameter :math:`\lambda_c` (the frequency of the cycle) is an
   additional parameter to be estimated by MLE.
   If the cyclical effect is stochastic (`stochastic_cycle=True`), then there
   is another parameter to estimate (the variance of the error term - note
   that both of the error terms here share the same variance, but are assumed
   to have independent draws).

   If the cycle is damped (`damped_cycle=True`), then there is a third
   parameter to estimate, :math:`\rho_c`.

   In order to achieve cycles with the appropriate frequencies, bounds are
   imposed on the parameter :math:`\lambda_c` in estimation. These can be
   controlled via the keyword argument `cycle_period_bounds`, which, if
   specified, must be a tuple of bounds on the **period** `(lower, upper)`.
   The bounds on the frequency are then calculated from those bounds.

   The default bounds, if none are provided, are selected in the following
   way:

   1. If no date / time information is provided, the frequency is
      constrained to be between zero and :math:`\pi`, so the period is
      constrained to be in :math:`[0.5, \infty]`.
   2. If the date / time information is provided, the default bounds
      allow the cyclical component to be between 1.5 and 12 years; depending
      on the frequency of the endogenous variable, this will imply different
      specific bounds.

   Following the fitting of the model, the unobserved cyclical component
   time series is available in the results class in the `cycle`
   attribute.

   **Irregular**

   The irregular components are independent and identically distributed (iid):

   .. math::

       \varepsilon_t \sim N(0, \sigma_\varepsilon^2)

   **Autoregressive Irregular**

   An autoregressive component (often used as a replacement for the white
   noise irregular term) can be specified as:

   .. math::

       \varepsilon_t = \rho(L) \varepsilon_{t-1} + \epsilon_t \\
       \epsilon_t \sim N(0, \sigma_\epsilon^2)

   In this case, the AR order is specified via the `autoregressive` keyword,
   and the autoregressive coefficients are estimated.

   Following the fitting of the model, the unobserved autoregressive component
   time series is available in the results class in the `autoregressive`
   attribute.

   **Regression effects**

   Exogenous regressors can be pass to the `exog` argument. The regression
   coefficients will be estimated by maximum likelihood unless
   `mle_regression=False`, in which case the regression coefficients will be
   included in the state vector where they are essentially estimated via
   recursive OLS.

   If the regression_coefficients are included in the state vector, the
   recursive estimates are available in the results class in the
   `regression_coefficients` attribute.

   .. rubric:: References

   .. [1] Durbin, James, and Siem Jan Koopman. 2012.
      Time Series Analysis by State Space Methods: Second Edition.
      Oxford University Press.

   .. py:method:: fit(X, y=None)

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
      :py:func:`fit <sandbox.tsa.ssm.LinearGaussianState.fit>`.

      :returns: **result** -- If an instance has ``model_result_``, True. Otherwise, False.
      :rtype: bool


   .. py:method:: estimated_params_()
      :property:

      Estimated parameters.

      :py:class:`LinearGaussianState <sandbox.tsa.ssm.LinearGaussianState>` estimates (1) states parameters,
      (2) fixed parameters (e.g., fixed state variances, regression coefficients).

      This method returns (2) fixed parameters that are estimated in
      :py:func:`fit <sandbox.tsa.ssm.LinearGaussianState.fit>` as dict format.

      :returns: **estimated_params** -- The estimated parameters which are other than state parameters.
      :rtype: dict


   .. py:method:: fittedvalues_()
      :property:

      The fitted values of the model.

      :returns: **fittedvalues** -- The fitted values to be estimated.
      :rtype: numpy.ndarray


   .. py:method:: level_filtered_()
      :property:

      Filtered level component.

      :returns: **level** -- Filtered level component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: level_()
      :property:

      Smoothed level component.

      :returns: **level** -- Smoothed level component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: trend_filtered_()
      :property:

      Filtered trend component.

      :returns: **trend** -- Filtered trend component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: trend_()
      :property:

      Smoothed trend component.

      :returns: **trend** -- Smoothed trend component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: seasonal_filtered_()
      :property:

      Filtered seasonal component.

      :returns: **seasonal** -- Filtered seasonal component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: seasonal_()
      :property:

      Smoothed seasonal component.

      :returns: **seasonal** -- Smoothed seasonal component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: freq_seasonal_filtered_()
      :property:

      Filtered frequency domain seasonal component.

      :returns: **freq_seasonal** -- Filtered frequency domain seasonal component
      :rtype: {list[numpy.ndarray], None}


   .. py:method:: freq_seasonal_()
      :property:

      Smoothed frequency domain seasonal component.

      :returns: **freq_seasonal** -- Smoothed frequency domain seasonal component
      :rtype: {list[numpy.ndarray], None}


   .. py:method:: cycle_filtered_()
      :property:

      Filtered cycle component.

      :returns: **cycle** -- Filtered cycle component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: cycle_()
      :property:

      Smoothed cycle component.

      :returns: **cycle** -- Smoothed cycle component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: autoregressive_filtered_()
      :property:

      Filtered autoregressive component.

      :returns: **autoregressive** -- Filtered autoregressive component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: autoregressive_()
      :property:

      Smoothed autoregressive component.

      :returns: **autoregressive** -- Smoothed autoregressive component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: regression_filtered_()
      :property:

      Filtered regression component.

      :returns: **regression** -- Filtered regression component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: regression_()
      :property:

      Smoothed regression component.

      :returns: **regression** -- Smoothed regression component.
      :rtype: {numpy.ndarray, None}


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


   .. py:method:: score(X, y, scorer='r2', **kwargs)

      Return the coefficient of determination of the prediction.

      The default coefficient of determination :math:`R^2` is defined as
      :math:`(1 - \\frac{u}{v})`, where :math:`u` is the residual
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

      Return component names that are implemented in a defined model.


   .. py:method:: level_predicted_(X)

      Predicted level component.

      :param X: Design matrix expressing the regression dummies or variables in
                the period to be predicted. If no regression is defined in the model,
                the index expressing the period or the period steps to be predicted
                must be set.
      :type X: {array-like, int}

      :returns: **level** -- Predicted level component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: trend_predicted_(X)

      Predicted trend component.

      :param X: Design matrix expressing the regression dummies or variables in
                the period to be predicted. If no regression is defined in the model,
                the index expressing the period or the period steps to be predicted
                must be set.
      :type X: {array-like, int}

      :returns: **trend** -- Predicted trend component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: seasonal_predicted_(X)

      Predicted seasonal component.

      :param X: Design matrix expressing the regression dummies or variables in
                the period to be predicted. If no regression is defined in the model,
                the index expressing the period or the period steps to be predicted
                must be set.
      :type X: {array-like, int}

      :returns: **seasonal** -- Predicted seasonal component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: freq_seasonal_predicted_(X)

      Predicted frequency domain seasonal component.

      :param X: Design matrix expressing the regression dummies or variables in
                the period to be predicted. If no regression is defined in the model,
                the index expressing the period or the period steps to be predicted
                must be set.
      :type X: {array-like, int}

      :returns: **freq_seasonal** -- Predicted frequency domain seasonal component.
      :rtype: {list[numpy.ndarray], None}


   .. py:method:: cycle_predicted_(X)

      Predicted cycle component.

      :param X: Design matrix expressing the regression dummies or variables in
                the period to be predicted. If no regression is defined in the model,
                the index expressing the period or the period steps to be predicted
                must be set.
      :type X: {array-like, int}

      :returns: **cycle** -- Predicted cycle component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: autoregressive_predicted_(X)

      Predicted autoregressive component.

      :param X: Design matrix expressing the regression dummies or variables in
                the period to be predicted. If no regression is defined in the model,
                the index expressing the period or the period steps to be predicted
                must be set.
      :type X: {array-like, int}

      :returns: **autoregressive** -- Predicted autoregressive component.
      :rtype: {numpy.ndarray, None}


   .. py:method:: regression_predicted_(X)

      Predicted regression component.

      :param X: Design matrix expressing the regression dummies or variables in
                the period to be predicted. If no regression is defined in the model,
                the index expressing the period or the period steps to be predicted
                must be set.
      :type X: {array-like, int}

      :returns: **regression** -- Predicted regression component.
      :rtype: {numpy.ndarray, None}



