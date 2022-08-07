"""SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)."""
import pandas as pd

from sandbox.datamodel.ts_datamodel import TimeSeriesModelData
from sandbox.graphics.ts_grapher import TimeSeriesGrapherMixin
from sandbox.tsa.base import BaseTimeSeriesModel


class SARIMAXModel(BaseTimeSeriesModel, TimeSeriesGrapherMixin):
    r"""Linear Gaussian state space model.

    Parameters
    ----------
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the non-zero polynomial exponents to include, in
        increasing order. For example, `[1,1,0,1]` denotes
        :math:`a + bt + ct^3`. Default is to not include a trend component.
    s : int, optional
        The period for seasonal differencing, ``s`` refers to the number of
        periods in each season. For example, ``s`` is 4 for quarterly data, 12
        for monthly data, or 1 for annual (non-seasonal) data. Default is 1.
        Note that if ``s`` == 1 (i.e., is non-seasonal), ``seasonal`` will be
        set to False.
    seasonal : bool, optional
        Whether to fit a seasonal ARIMA. Default is True. Note that if
        ``seasonal`` is True and ``s`` == 1, ``seasonal`` will be set to False.
    method : str, optional
        The ``method`` determines which solver from ``scipy.optimize``
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
    start_p : int, optional
        The starting value of ``p``, the order (or number of time lags)
        of the auto-regressive ("AR") model. Must be a positive integer.
    d : int, optional
        The order of first-differencing. If None (by default), the value
        will automatically be selected based on the results of the ``test``
        (i.e., either the Kwiatkowski–Phillips–Schmidt–Shin, Augmented
        Dickey-Fuller or the Phillips–Perron test will be conducted to find
        the most probable value). Must be a positive integer or None. Note
        that if ``d`` is None, the runtime could be significantly longer.
    start_q : int, optional
        The starting value of ``q``, the order of the moving-average
        ("MA") model. Must be a positive integer.
    max_p : int, optional
        The maximum value of ``p``, inclusive. Must be a positive integer
        greater than or equal to ``start_p``.
    max_d : int, optional
        The maximum value of ``d``, or the maximum number of non-seasonal
        differences. Must be a positive integer greater than or equal to ``d``.
    max_q : int, optional
        The maximum value of ``q``, inclusive. Must be a positive integer
        greater than ``start_q``.
    start_P : int, optional
        The starting value of ``P``, the order of the autoregressive portion
        of the seasonal model.
    D : int, optional
        The order of the seasonal differencing. If None (by default, the value
        will automatically be selected. Must be a positive integer or None.
    start_Q : int, optional
        The starting value of ``Q``, the order of the moving-average portion
        of the seasonal model.
    max_P : int, optional
        The maximum value of ``P``, inclusive. Must be a positive integer
        greater than ``start_P``.
    max_D : int, optional
        The maximum value of ``D``. Must be a positive integer greater
        than ``D``.
    max_Q : int, optional
        The maximum value of ``Q``, inclusive. Must be a positive integer
        greater than ``start_Q``.
    stepwise : bool, optional
        Whether to use the stepwise algorithm outlined in [1]_ Hyndman and Khandakar
        (2008) to identify the optimal model parameters. The stepwise algorithm
        can be significantly faster than fitting all hyperparameter combinations
        and is less likely to over-fit the model.
    max_order : int, optional
        Maximum value of :math:`p+q+P+Q` if model selection is not stepwise.
        If the sum of ``p`` and ``q`` is >= ``max_order``, a model will
        *not* be fit with those parameters, but will progress to the next
        combination. Default is 5. If ``max_order`` is None, it means there
        are no constraints on maximum order.
    n_jobs : int, optional
        The number of models to fit in parallel in the case of a grid search
        (``stepwise=False``). Default is 1, but -1 can be used to designate
        "as many as possible".
    trace : {bool, int}, optional
        Whether to print status on the fits. A value of False will print no
        debugging information. A value of True will print some. Integer values
        exceeding 1 will print increasing amounts of debug information at each
        fit.

    Examples
    --------
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

    Notes
    -----
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

    See Also
    --------
    statsmodels.tsa.statespace.sarimax.SARIMAX
    pmdarima.arima.ARIMA
    pmdarima.arima.AutoARIMA

    References
    ----------
    .. [1] Hyndman, R. J., & Khandakar, Y. (2008).
           Automatic time series forecasting: the forecast package for R.
           Journal of statistical software, 27, 1-22.

    """

    def __init__(
        self,
        trend=None,
        s=1,
        seasonal=True,
        method="lbfgs",
        start_p=2,
        d=None,
        start_q=2,
        max_p=5,
        max_d=2,
        max_q=5,
        start_P=1,
        D=None,
        start_Q=1,
        max_P=2,
        max_D=1,
        max_Q=2,
        stepwise=True,
        max_order=5,
        n_jobs=1,
        trace=False,
    ):
        self.trend = trend
        self.s = s
        self.seasonal = seasonal
        self.method = method
        self.start_p = start_p
        self.d = d
        self.start_q = start_q
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.start_P = start_P
        self.D = D
        self.start_Q = start_Q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.stepwise = stepwise
        self.max_order = max_order
        self.n_jobs = n_jobs
        self.trace = trace
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y=None, **kwargs):
        """Fit the model.

        Parameters
        ----------
        X : array_like
            Training data on regressions. If no regression is defined,
            just y is to be defined.
        y : {array_like, None}, default
            Target values. If no regression is defined, just y is to be
            defined in the place of X.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # このモジュールで提供しているデータクラスに変換.
        self.data_ = TimeSeriesModelData(X, y)
        self.model_result_ = self._get_model_result(
            endog=self.data_.y.values, exog=self.data_.X.values
        )
        return self

    def _get_model_result(self, endog, exog):
        import warnings

        import pmdarima as pm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_result = pm.auto_arima(
                y=endog,
                X=exog,
                start_p=self.start_p,
                d=self.d,
                start_q=self.start_q,
                max_p=self.max_p,
                max_d=self.max_d,
                max_q=self.max_q,
                start_P=self.start_P,
                D=self.D,
                start_Q=self.start_Q,
                max_P=self.max_P,
                max_D=self.max_D,
                max_Q=self.max_Q,
                max_order=self.max_order,
                m=self.s,
                seasonal=self.seasonal,
                stepwise=self.stepwise,
                n_jobs=self.n_jobs,
                trend=self.trend,
                method=self.method,
                trace=self.trace,
            )

        return model_result.arima_res_

    def has_model_result(self):
        r"""Whether an instance has ``model_result_``.

        Some method needs ``model_result_`` that can be gained after
        :py:func:`fit <sandbox.tsa.sarimaxSARIMAXModel.fit>`.

        Returns
        -------
        result : bool
            If an instance has ``model_result_``, True. Otherwise, False.
        """
        if hasattr(self, "model_result_"):
            return True
        else:
            return False

    @property
    def estimated_params_(self):
        """Estimated parameters.

        :py:class:`SARIMAXModel <sandbox.tsa.sarimax.SARIMAXModel>` estimates (1) regression
        , (2) autoregressive, (3) moving average, (4) seasonal autoregressive, (5) seasonal
        moving average coefficients and (6) variance of noise.

        Returns
        -------
        estimated_params : dict
            The estimated parameters.
        """
        estimated_params = None
        if not self.has_model_result():
            import warnings

            msg = "This method works after performing `fit`."
            warnings.warn(msg)
        else:
            estimated_params = dict(
                zip(
                    self.model_result_.param_names,
                    self.model_result_.params,
                )
            )
        return estimated_params

    @property
    def fittedvalues_(self):
        """The fitted values of the model.

        Returns
        -------
        fittedvalues : numpy.ndarray
            The fitted values to be estimated.
        """
        fittedvalues = None
        if not self.has_model_result():
            import warnings

            msg = "This method works after performing `fit`."
            warnings.warn(msg)
        else:
            fittedvalues = self.model_result_.fittedvalues
        return fittedvalues

    def predict(self, X, is_pandas=False):
        """Predict using the model.

        Parameters
        ----------
        X : {array-like, int}
            Design matrix expressing the regression dummies or variables in
            the period to be predicted. If no regression is defined in the model,
            the index expressing the period or the period steps to be predicted
            must be set.
        is_pandas: bool, optional
            If True, the return data type is pandas.Series. Otherwise, numpy.ndarray.

        Returns
        -------
        predicted_mean : array-like
            Mean of predictive distribution of query points.
        """
        index, exog = self.data_.get_index_and_values_from_X_pred(X)
        start = self.data_.nobs
        end = self.data_.nobs + len(index) - 1
        pred = self._get_prediction(start=start, end=end, exog=exog)
        if is_pandas:
            return pd.DataFrame(
                pred.predicted_mean, index=index, columns=["predicted_mean"]
            )
        else:
            return pred.predicted_mean

    def conf_int(self, X, alpha=0.95, is_pandas=False):
        """
        Compute the confidence interval.

        Parameters
        ----------
        X : {array-like, int}
            Design matrix expressing the regression dummies or variables in
            the period to be predicted. If no regression is defined in the model,
            the index expressing the period or the period steps to be predicted
            must be set.
        alpha : float, optional
            The `alpha` level for the confidence interval. The default
            `alpha` = .95 returns a 95% confidence interval.
        is_pandas: bool, optional
            If True, the return data type is pandas.Series. Otherwise, numpy.ndarray.

        Returns
        -------
        array_like
            The confidence intervals.
        """
        index, exog = self.data_.get_index_and_values_from_X_pred(X)
        start = self.data_.nobs
        end = self.data_.nobs + len(index) - 1
        pred = self._get_prediction(start=start, end=end, exog=exog)
        if is_pandas:
            a = int(round(alpha * 100, 0))
            return pd.DataFrame(
                pred.conf_int(alpha=1 - alpha),
                index=index,
                columns=[f"lower_{a}", f"upper_{a}"],
            )
        else:
            return pred.conf_int(alpha=1 - alpha)

    def _get_prediction(self, start, end, exog):
        return self.model_result_.get_prediction(start=start, end=end, exog=exog)

    def score(self, X, y, scorer="r2"):
        """Return the coefficient of determination of the prediction.

        The default coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \\frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is 1.0, and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : {array-like, int}
            Design matrix expressing the regression dummies or variables in
            the period to be predicted. If no regression is defined in the model,
            the index expressing the period or the period steps to be predicted
            must be set.
        y : array-like
            True values for `X`.
        scorer : str, optional
            Expressing the type of the coefficient of determination.

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)``.

        """
        score = super(SARIMAXModel, self).score(X, y, scorer=scorer)
        return score

    @property
    def components_name_(self):
        """Return component names.

        Although SARIMAX model has no state parameter, present here for API
        consistency.
        """
        components_name = list()
        return components_name
