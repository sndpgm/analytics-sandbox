"""State space model."""
import numpy as np
import pandas as pd

from sandbox.datamodel.ts_datamodel import TimeSeriesModelData
from sandbox.graphics.ts_grapher import TimeSeriesGrapherMixin
from sandbox.tsa.base import BaseTimeSeriesModel


class LinearGaussianStateSpaceModel(BaseTimeSeriesModel, TimeSeriesGrapherMixin):
    r"""
    Linear Gaussian state space model.

    Parameters
    ----------
    level : bool, optional
        Whether to include a level component. Default is False.
    trend : bool, optional
        Whether to include a trend component. Default is False. If True,
        `level` must also be True.
    seasonal : {int, None}, optional
        The period of the seasonal component, if any. Default is None.
    freq_seasonal : {list[dict], None}, optional.
        Whether (and how) to model seasonal component(s) with trig. functions.
        If specified, there is one dictionary for each frequency-domain
        seasonal component.  Each dictionary must have the key, value pair for
        'period' -- integer and may have a key, value pair for
        'harmonics' -- integer. If 'harmonics' is not specified in any of the
        dictionaries, it defaults to the floor of period/2.
    cycle : bool, optional
        Whether to include a cycle component. Default is False.
    autoregressive : {int, None}, optional
        The order of the autoregressive component. Default is None.
    irregular : bool, optional
        Whether to include an irregular component. Default is False.
    stochastic_level : bool, optional
        Whether any level component is stochastic. Default is False.
    stochastic_trend : bool, optional
        Whether any trend component is stochastic. Default is False.
    stochastic_seasonal : bool, optional
        Whether any seasonal component is stochastic. Default is True.
    stochastic_freq_seasonal : list[bool], optional
        Whether each seasonal component(s) is (are) stochastic.  Default
        is True for each component.  The list should be of the same length as
        freq_seasonal.
    stochastic_cycle : bool, optional
        Whether any cycle component is stochastic. Default is False.
    damped_cycle : bool, optional
        Whether the cycle component is damped. Default is False.
    cycle_period_bounds : tuple, optional
        A tuple with lower and upper allowed bounds for the period of the
        cycle. If not provided, the following default bounds are used:
        (1) if no date / time information is provided, the frequency is
        constrained to be between zero and :math:`\pi`, so the period is
        constrained to be in [0.5, infinity].
        (2) If the date / time information is provided, the default bounds
        allow the cyclical component to be between 1.5 and 12 years; depending
        on the frequency of the endogenous variable, this will imply different
        specific bounds.
    mle_regression : bool, optional
        Whether to estimate regression coefficients by maximum likelihood
        as one of hyperparameters. Default is True.
        If False, the regression coefficients are estimated by recursive OLS,
        included in the state vector.
    use_exact_diffuse : bool, optional
        Whether to use exact diffuse initialization for non-stationary
        states. Default is False (in which case approximate diffuse
        initialization is used).

    Examples
    --------
    >>> from sandbox.datamodel.ts_simulator import UnobservedComponentsSimulator
    >>> from sandbox.tsa.ssm import LinearGaussianStateSpaceModel
    >>> from sklearn.model_selection import train_test_split
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

    Notes
    -----
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

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.
    """

    def __init__(
        self,
        level=False,
        trend=False,
        seasonal=None,
        freq_seasonal=None,
        cycle=False,
        autoregressive=None,
        irregular=False,
        stochastic_level=False,
        stochastic_trend=False,
        stochastic_seasonal=True,
        stochastic_freq_seasonal=None,
        stochastic_cycle=False,
        damped_cycle=False,
        cycle_period_bounds=None,
        mle_regression=True,
        use_exact_diffuse=False,
    ):
        self.level = level
        self.trend = trend
        self.seasonal = seasonal
        self.freq_seasonal = freq_seasonal
        self.cycle = cycle
        self.autoregressive = autoregressive
        self.irregular = irregular
        self.stochastic_level = stochastic_level
        self.stochastic_trend = stochastic_trend
        self.stochastic_seasonal = stochastic_seasonal
        self.stochastic_freq_seasonal = stochastic_freq_seasonal
        self.stochastic_cycle = stochastic_cycle
        self.damped_cycle = damped_cycle  # 減衰係数ρを乗じるか
        self.cycle_period_bounds = cycle_period_bounds
        self.mle_regression = mle_regression
        self.use_exact_diffuse = use_exact_diffuse

    def fit(self, X, y=None):
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
        # ToDo: util関数でimportしていなければエラー (または自動でインストール?) を返すメソッドを用意.
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        # ToDo: 下記のfitメソッドに合う引数も指定できるようにする.
        model_result = UnobservedComponents(
            endog=endog,
            level=self.level,
            trend=self.trend,
            seasonal=self.seasonal,
            freq_seasonal=self.freq_seasonal,
            cycle=self.cycle,
            autoregressive=self.autoregressive,
            exog=exog,
            irregular=self.irregular,
            stochastic_level=self.stochastic_level,
            stochastic_trend=self.stochastic_trend,
            stochastic_seasonal=self.stochastic_seasonal,
            stochastic_freq_seasonal=self.stochastic_freq_seasonal,
            stochastic_cycle=self.stochastic_cycle,
            damped_cycle=self.damped_cycle,
            cycle_period_bounds=self.cycle_period_bounds,
            mle_regression=self.mle_regression,
            use_exact_diffuse=self.use_exact_diffuse,
        ).fit()

        return model_result

    def has_model_result(self):
        r"""Whether an instance has ``model_result_``.

        Some method needs ``model_result_`` that can be gained after
        :py:func:`fit <sandbox.tsa.ssm.LinearGaussianState.fit>`.

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

        :py:class:`LinearGaussianState <sandbox.tsa.ssm.LinearGaussianState>` estimates (1) states parameters,
        (2) fixed parameters (e.g., fixed state variances, regression coefficients).

        This method returns (2) fixed parameters that are estimated in
        :py:func:`fit <sandbox.tsa.ssm.LinearGaussianState.fit>` as dict format.

        Returns
        -------
        estimated_params : dict
            The estimated parameters which are other than state parameters.
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
            # fittedvalues = get_1d_arr(self.model_result_.fittedvalues)[0]
            fittedvalues = self.model_result_.fittedvalues
        return fittedvalues

    @property
    def level_filtered_(self):
        """Filtered level component.

        Returns
        -------
        level : {numpy.ndarray, None}
            Filtered level component.
        """
        return self._level(which="filtered")

    @property
    def level_(self):
        """Smoothed level component.

        Returns
        -------
        level : {numpy.ndarray, None}
            Smoothed level component.
        """
        return self._level()

    @property
    def trend_filtered_(self):
        """Filtered trend component.

        Returns
        -------
        trend : {numpy.ndarray, None}
            Filtered trend component.
        """
        return self._trend(which="filtered")

    @property
    def trend_(self):
        """Smoothed trend component.

        Returns
        -------
        trend : {numpy.ndarray, None}
            Smoothed trend component.
        """
        return self._trend()

    @property
    def seasonal_filtered_(self):
        """Filtered seasonal component.

        Returns
        -------
        seasonal : {numpy.ndarray, None}
            Filtered seasonal component.
        """
        return self._seasonal(which="filtered")

    @property
    def seasonal_(self):
        """Smoothed seasonal component.

        Returns
        -------
        seasonal : {numpy.ndarray, None}
            Smoothed seasonal component.
        """
        return self._seasonal()

    @property
    def freq_seasonal_filtered_(self):
        """Filtered frequency domain seasonal component.

        Returns
        -------
        freq_seasonal : {list[numpy.ndarray], None}
            Filtered frequency domain seasonal component
        """
        return self._freq_seasonal(which="filtered")

    @property
    def freq_seasonal_(self):
        """Smoothed frequency domain seasonal component.

        Returns
        -------
        freq_seasonal : {list[numpy.ndarray], None}
            Smoothed frequency domain seasonal component
        """
        return self._freq_seasonal()

    @property
    def cycle_filtered_(self):
        """Filtered cycle component.

        Returns
        -------
        cycle : {numpy.ndarray, None}
            Filtered cycle component.
        """
        return self._cycle(which="filtered")

    @property
    def cycle_(self):
        """Smoothed cycle component.

        Returns
        -------
        cycle : {numpy.ndarray, None}
            Smoothed cycle component.
        """
        return self._cycle()

    @property
    def autoregressive_filtered_(self):
        """Filtered autoregressive component.

        Returns
        -------
        autoregressive : {numpy.ndarray, None}
            Filtered autoregressive component.
        """
        return self._autoregressive(which="filtered")

    @property
    def autoregressive_(self):
        """Smoothed autoregressive component.

        Returns
        -------
        autoregressive : {numpy.ndarray, None}
            Smoothed autoregressive component.
        """
        return self._autoregressive()

    @property
    def regression_filtered_(self):
        """Filtered regression component.

        Returns
        -------
        regression : {numpy.ndarray, None}
            Filtered regression component.
        """
        return self._regression(which="filtered")

    @property
    def regression_(self):
        """Smoothed regression component.

        Returns
        -------
        regression : {numpy.ndarray, None}
            Smoothed regression component.
        """
        return self._regression()

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
        pred = self._get_prediction(X)
        if is_pandas:
            index = self.data_.get_index_and_values_from_X_pred(X)[0]
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
        pred = self._get_prediction(X)
        if is_pandas:
            index = self.data_.get_index_and_values_from_X_pred(X)[0]
            a = int(round(alpha * 100, 0))
            return pd.DataFrame(
                pred.conf_int(alpha=1 - alpha),
                index=index,
                columns=[f"lower_{a}", f"upper_{a}"],
            )
        else:
            return pred.conf_int(alpha=1 - alpha)

    def _get_prediction(self, X):
        index, exog = self.data_.get_index_and_values_from_X_pred(X)
        start = self.data_.nobs
        end = self.data_.nobs + len(index) - 1
        return self.model_result_.get_prediction(start=start, end=end, exog=exog)

    def score(self, X, y, scorer="r2", **kwargs):
        r"""Return the coefficient of determination of the prediction.

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
        score = super(LinearGaussianStateSpaceModel, self).score(
            X, y, scorer=scorer, **kwargs
        )
        return score

    @property
    def components_name_(self):
        """Return component names that are implemented in a defined model."""
        components_name = list()
        if self.level:
            components_name.append("level_")
        if self.trend:
            components_name.append("trend_")
        if self.seasonal:
            components_name.append("seasonal_")
        if self.freq_seasonal:
            components_name.append("freq_seasonal_")
        if self.cycle:
            components_name.append("cycle_")
        if self.autoregressive:
            components_name.append("autoregressive_")
        return components_name

    def _predicted_state(self, X):
        predicted_state = None
        if self.has_model_result():
            predicted_state = self._get_prediction(
                X
            )._results.prediction_results.results.predicted_state
        return predicted_state

    def level_predicted_(self, X):
        """Predicted level component.

        Parameters
        ----------
        X : {array-like, int}
            Design matrix expressing the regression dummies or variables in
            the period to be predicted. If no regression is defined in the model,
            the index expressing the period or the period steps to be predicted
            must be set.

        Returns
        -------
        level : {numpy.ndarray, None}
            Predicted level component.
        """
        return self._level(which="predicted", X=X)

    def trend_predicted_(self, X):
        """Predicted trend component.

        Parameters
        ----------
        X : {array-like, int}
            Design matrix expressing the regression dummies or variables in
            the period to be predicted. If no regression is defined in the model,
            the index expressing the period or the period steps to be predicted
            must be set.

        Returns
        -------
        trend : {numpy.ndarray, None}
            Predicted trend component.
        """
        return self._trend(which="predicted", X=X)

    def seasonal_predicted_(self, X):
        """Predicted seasonal component.

        Parameters
        ----------
        X : {array-like, int}
            Design matrix expressing the regression dummies or variables in
            the period to be predicted. If no regression is defined in the model,
            the index expressing the period or the period steps to be predicted
            must be set.

        Returns
        -------
        seasonal : {numpy.ndarray, None}
            Predicted seasonal component.
        """
        return self._seasonal(which="predicted", X=X)

    def freq_seasonal_predicted_(self, X):
        """Predicted frequency domain seasonal component.

        Parameters
        ----------
        X : {array-like, int}
            Design matrix expressing the regression dummies or variables in
            the period to be predicted. If no regression is defined in the model,
            the index expressing the period or the period steps to be predicted
            must be set.

        Returns
        -------
        freq_seasonal : {list[numpy.ndarray], None}
            Predicted frequency domain seasonal component.
        """
        return self._freq_seasonal(which="predicted", X=X)

    def cycle_predicted_(self, X):
        """Predicted cycle component.

        Parameters
        ----------
        X : {array-like, int}
            Design matrix expressing the regression dummies or variables in
            the period to be predicted. If no regression is defined in the model,
            the index expressing the period or the period steps to be predicted
            must be set.

        Returns
        -------
        cycle : {numpy.ndarray, None}
            Predicted cycle component.
        """
        return self._freq_seasonal(which="predicted", X=X)

    def autoregressive_predicted_(self, X):
        """Predicted autoregressive component.

        Parameters
        ----------
        X : {array-like, int}
            Design matrix expressing the regression dummies or variables in
            the period to be predicted. If no regression is defined in the model,
            the index expressing the period or the period steps to be predicted
            must be set.

        Returns
        -------
        autoregressive : {numpy.ndarray, None}
            Predicted autoregressive component.
        """
        return self._autoregressive(which="predicted", X=X)

    def regression_predicted_(self, X):
        """Predicted regression component.

        Parameters
        ----------
        X : {array-like, int}
            Design matrix expressing the regression dummies or variables in
            the period to be predicted. If no regression is defined in the model,
            the index expressing the period or the period steps to be predicted
            must be set.

        Returns
        -------
        regression : {numpy.ndarray, None}
            Predicted regression component.
        """
        return self._regression(which="predicted", X=X)

    def _level(self, which="smoothed", X=None):
        out = None
        if self.has_model_result():
            spec = self.model_result_.specification
            if spec.level:
                if which in ["filtered", "smoothed"]:
                    out = self.model_result_.level[which]
                if which == "predicted":
                    offset = 0
                    predicted_state = self._predicted_state(X)
                    out = predicted_state[offset, :-1]
        return out

    def _trend(self, which="smoothed", X=None):
        out = None
        if self.has_model_result():
            spec = self.model_result_.specification
            if spec.trend:
                if which in ["filtered", "smoothed"]:
                    out = self.model_result_.trend[which]
                if which == "predicted":
                    offset = int(spec.level)
                    predicted_state = self._predicted_state(X)
                    out = predicted_state[offset, :-1]
        return out

    def _seasonal(self, which="smoothed", X=None):
        out = None
        if self.has_model_result():
            spec = self.model_result_.specification
            if spec.seasonal:
                if which in ["filtered", "smoothed"]:
                    out = self.model_result_.seasonal[which]
                if which == "predicted":
                    offset = int(spec.trend + spec.level)
                    predicted_state = self._predicted_state(X)
                    out = predicted_state[offset, :-1]
        return out

    def _freq_seasonal(self, which="smoothed", X=None):
        out = []
        if self.has_model_result():
            spec = self.model_result_.specification
            if spec.freq_seasonal:
                if which in ["filtered", "smoothed"]:
                    n_freq_seasonal = len(self.model_result_.freq_seasonal)
                    for i in range(n_freq_seasonal):
                        item = self.model_result_.freq_seasonal[i][which]
                        out.append(item)
                if which == "predicted":
                    predicted_state = self._predicted_state(X)
                    previous_states_offset = int(
                        spec.trend
                        + spec.level
                        + self.model_result_._k_states_by_type["seasonal"]
                    )
                    previous_f_seas_offset = 0
                    for ix, h in enumerate(spec.freq_seasonal_harmonics):
                        offset = previous_states_offset + previous_f_seas_offset
                        states_in_sum = np.arange(0, 2 * h, 2)
                        item = np.sum(
                            [predicted_state[offset + j] for j in states_in_sum], axis=0
                        )
                        out.append(item[:-1])
                        previous_f_seas_offset += 2 * h
        return np.array(out)

    def _cycle(self, which="smoothed", X=None):
        out = None
        if self.has_model_result():
            spec = self.model_result_.specification
            if spec.cycle:
                if which in ["filtered", "smoothed"]:
                    out = self.model_result_.cycle[which]
                if which == "predicted":
                    offset = int(
                        spec.trend
                        + spec.level
                        + self.model_result_._k_states_by_type["seasonal"]
                        + self.model_result_._k_states_by_type["freq_seasonal"]
                    )
                    predicted_state = self._predicted_state(X)
                    out = predicted_state[offset, :-1]
        return out

    def _autoregressive(self, which="smoothed", X=None):
        out = None
        if self.has_model_result():
            spec = self.model_result_.specification
            if spec.autoregressive:
                if which in ["filtered", "smoothed"]:
                    out = self.model_result_.autoregressive[which]
                if which == "predicted":
                    offset = int(
                        spec.trend
                        + spec.level
                        + self.model_result_._k_states_by_type["seasonal"]
                        + self.model_result_._k_states_by_type["freq_seasonal"]
                        + self.model_result_._k_states_by_type["cycle"]
                    )
                    predicted_state = self._predicted_state(X)
                    out = predicted_state[offset, :-1]
        return out

    def _regression(self, which="smoothed", X=None):
        out = None
        if self.has_model_result():
            spec = self.model_result_.specification
            if spec.regression:
                # mle_regression = True のとき, 最尤推定法によって回帰係数を推定しているため,
                # 状態変数を格納している配列には保存されていない.
                if not spec.mle_regression:
                    if which in ["filtered", "smoothed"]:
                        out = self.model_result_.regression_coefficients[which]
                    if which == "predicted":
                        offset = int(
                            spec.trend
                            + spec.level
                            + self.model_result_._k_states_by_type["seasonal"]
                            + self.model_result_._k_states_by_type["freq_seasonal"]
                            + self.model_result_._k_states_by_type["cycle"]
                            + spec.ar_order
                        )
                        start = offset
                        end = offset + spec.k_exog
                        predicted_state = self._predicted_state(X)
                        out = predicted_state[start:end, :-1]
                else:
                    coefficient = np.zeros(spec.k_exog)
                    offset = 0
                    for k, v in self.estimated_params_.items():
                        if "beta." in k:
                            coefficient[offset] = v
                            offset += 1

                    # NumPy において `*` はアダマール積
                    # ToDo: 下ではdata_.Xがnumpyを想定しているが, そもそもTimeSeriesModelDataの標準スタイルをnumpyにするか再検討.
                    if which in ["filtered", "smoothed"]:
                        out = (self.data_.X * coefficient).T
                    if which == "predicted":
                        exog = self.data_.get_index_and_values_from_X_pred(X)[1]
                        out = (exog * coefficient).T
        return out
