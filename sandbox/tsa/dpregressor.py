"""
The :mod:`sandbox.tsa.dpregressor` module includes classes and
functions on the linear regressor models on deterministic process for time series.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted
from statsmodels.tsa.deterministic import (
    DeterministicProcess,
    Fourier,
    Seasonality,
    TimeTrend,
)

from sandbox.datamodel.ts_datamodel import TimeSeriesModelData
from sandbox.graphics.ts_grapher import TimeSeriesGrapherMixin


class DeterministicProcessRegressor(LinearRegression, TimeSeriesGrapherMixin):
    def __init__(self, level=True, trend=False, seasonal=None, freq_seasonal=None):
        super(DeterministicProcessRegressor, self).__init__(fit_intercept=False)

        # validate arguments.
        level, trend, seasonal, freq_seasonal = self._validate_args(
            level=level, trend=trend, seasonal=seasonal, freq_seasonal=freq_seasonal
        )

        # set instance variables.
        self.level = level
        self.trend = trend
        self.seasonal = seasonal
        self.freq_seasonal = freq_seasonal

        # whether this model has the following components: trend (level + trend),
        # seasonal and frequency-domain seasonal.
        self.has_trend = self.level or self.trend > 0
        self.has_seasonal = self.seasonal is not None
        self.has_freq_seasonal = self.freq_seasonal is not None

        self.terms = self._initialize_terms()
        self._deterministic_process = None
        self._data = None

    @staticmethod
    def _validate_args(level, trend, seasonal, freq_seasonal):
        """Validate arguments."""
        # check level
        if not isinstance(level, bool):
            msg = "Specified level must be bool, not {}".format(type(level))
            raise TypeError(msg)

        # check trend
        if isinstance(trend, bool):
            trend = 1 if trend else 0
        elif not isinstance(int(trend), int):
            msg = "Specified trend must be bool or int, not {}".format(type(trend))
            raise TypeError(msg)

        # check seasonal
        if seasonal is not None:
            if not isinstance(seasonal, list):
                msg = "Specified seasonal must be list, not {}".format(type(seasonal))
                raise TypeError(msg)
            for seas in seasonal:
                if not isinstance(int(seas), int):
                    msg = "All elements in seasonal must be int."
                    raise TypeError(msg)

        # check freq_seasonal
        if freq_seasonal is not None:
            if not isinstance(freq_seasonal, list):
                msg = "Specified freq_seasonal must be list, not {}".format(
                    type(freq_seasonal)
                )
                raise TypeError(msg)
            for seas in freq_seasonal:
                if not isinstance(seas, dict):
                    msg = "All elements in freq_seasonal must be dict."
                    raise TypeError(msg)
                else:
                    if set(seas.keys()) != {"period", "order"}:
                        msg = (
                            "All elements in freq_seasonal must be dict "
                            "that has the following two keys: period, order."
                        )
                        raise TypeError(msg)
                    if not isinstance(seas["period"], int):
                        msg = "Specified period in freq_seasonal must be int, not {}".format(
                            type(seas["period"])
                        )
                        raise TypeError(msg)
                    if not isinstance(seas["order"], int):
                        msg = "Specified order in freq_seasonal must be int, not {}".format(
                            type(seas["order"])
                        )
                        raise TypeError(msg)

        return level, trend, seasonal, freq_seasonal

    def _initialize_terms(self):
        terms = []

        if self.has_trend:
            tt = TimeTrend(constant=self.level, order=self.trend)
            terms.append(tt)

        if self.has_seasonal:
            for period in self.seasonal:
                seas = Seasonality(period=period)
                terms.append(seas)

        if self.has_freq_seasonal:
            for freq in self.freq_seasonal:
                period = freq["period"]
                order = freq["order"]
                seas = Fourier(period=period, order=order)
                terms.append(seas)

        return terms

    @property
    def data_(self):
        return self._data

    def get_deterministic_process(self, index):
        return DeterministicProcess(index, additional_terms=self.terms)

    @property
    def deterministic_process_(self):
        return self._deterministic_process

    def fit(self, X, y=None, **kwargs):
        # set dataset (private)
        self._data = TimeSeriesModelData(X, y)

        # set deterministic process (private)
        index = self.data_.y.index
        self._deterministic_process = self.get_deterministic_process(index)

        # override and execute fit method
        X = self.deterministic_process_.in_sample()
        y = self.data_.y.values
        super(DeterministicProcessRegressor, self).fit(X=X, y=y, **kwargs)

        return self

    def predict(self, X):
        X = self._get_X_pred(X)
        y = super(DeterministicProcessRegressor, self).predict(X)

        return y

    def _get_X_pred(self, X):
        index = self.data_.get_index_and_values_from_X_pred(X)[0]
        if isinstance(index, int):
            X = self.deterministic_process_.out_of_sample(index)
        else:
            start = index.min()
            stop = index.max()
            X = self.deterministic_process_.range(start=start, stop=stop)
        return X

    @property
    def features_index_in_(self):
        index = {}

        def f(pattern):
            exists = []
            for fet in self.feature_names_in_.tolist():
                ret = any(map(lambda x: fet.startswith(x), pattern))
                exists.append(ret)
            return np.where(exists)[0]

        if self.has_trend:
            pattern = ["const", "trend"]
            index["trend"] = f(pattern)
        if self.has_seasonal:
            seasonal = []
            for period in self.seasonal:
                pattern = [f"s({i},{period})" for i in range(2, period + 1)]
                seasonal.append(f(pattern))
            index["seasonal"] = seasonal
        if self.has_freq_seasonal:
            freq_seasonal = []
            for seas in self.freq_seasonal:
                period = seas["period"]
                order = seas["order"]
                pattern = []
                for j in range(1, order + 1):
                    pattern.append(f"sin({j},{period})")
                    pattern.append(f"cos({j},{period})")
                freq_seasonal.append(f(pattern))
            index["freq_seasonal"] = freq_seasonal

        return index

    @property
    def fittedvalues_(self):
        check_is_fitted(self)
        X = self.deterministic_process_.in_sample()
        return self.predict(X=X)

    @property
    def components_name_(self):
        component_name = []
        if self.has_trend:
            component_name.append("trend_")
        if self.has_seasonal:
            component_name.append("seasonal_")
        if self.has_freq_seasonal:
            component_name.append("freq_seasonal_")
        return component_name

    def _trend(self, which="fitted", X=None):
        check_is_fitted(self)
        out = None
        if self.has_trend:
            index = self.features_index_in_["trend"]
            if which == "fitted":
                out = (
                    self.coef_[:, index]
                    @ self.deterministic_process_.in_sample().values[:, index].T
                )
            elif which == "predicted":
                X = self._get_X_pred(X)
                out = self.coef_[:, index] @ X.values[:, index].T
            else:
                msg = (
                    "Specified `which` must be ['fitted', 'predicted'], not {}".format(
                        which
                    )
                )
                raise ValueError(msg)
        else:
            msg = "The components related with trend are not defined in this model."
            UserWarning(msg)
        return out

    def _seasonal(self, which="fitted", X=None):
        check_is_fitted(self)
        out = None
        if self.has_seasonal:
            indices = self.features_index_in_["seasonal"]
            out = []
            for index in indices:
                if which == "fitted":
                    ret = (
                        self.coef_[:, index]
                        @ self.deterministic_process_.in_sample().values[:, index].T
                    )
                elif which == "predicted":
                    X = self._get_X_pred(X)
                    ret = self.coef_[:, index] @ X.values[:, index].T
                else:
                    msg = "Specified `which` must be ['fitted', 'predicted'], not {}".format(
                        which
                    )
                    raise ValueError(msg)
                out.append(ret)
            out = np.array(out)
        else:
            msg = "The components related with seasonal are not defined in this model."
            UserWarning(msg)
        return out

    def _freq_seasonal(self, which="fitted", X=None):
        check_is_fitted(self)
        out = None
        if self.has_freq_seasonal:
            indices = self.features_index_in_["freq_seasonal"]
            out = []
            for index in indices:
                if which == "fitted":
                    ret = (
                        self.coef_[:, index]
                        @ self.deterministic_process_.in_sample().values[:, index].T
                    )
                elif which == "predicted":
                    X = self._get_X_pred(X)
                    ret = self.coef_[:, index] @ X.values[:, index].T
                else:
                    msg = "Specified `which` must be ['fitted', 'predicted'], not {}".format(
                        which
                    )
                    raise ValueError(msg)
                out.append(ret)
        else:
            msg = "The components related with freq_seasonal are not defined in this model."
            UserWarning(msg)
        return np.array(out)

    @property
    def trend_(self):
        return self._trend()

    @property
    def seasonal_(self):
        return self._seasonal()

    @property
    def freq_seasonal_(self):
        return self._freq_seasonal()

    def trend_predicted_(self, X):
        return self._trend(which="predicted", X=X)

    def seasonal_predicted_(self, X):
        return self._seasonal(which="predicted", X=X)

    def freq_seasonal_predicted_(self, X):
        return self._freq_seasonal(which="predicted", X=X)
