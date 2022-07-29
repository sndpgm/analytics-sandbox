from warnings import warn

import numpy as np
import pandas as pd

from sandbox.datamodel.base import BaseData, BaseDataSimulator


class UnobservedComponentsSimulator(BaseDataSimulator):
    def __init__(
        self,
        steps,
        level=True,
        trend=False,
        freq_seasonal=None,
        exog_params=None,
        start_param_level=0.0,
        stddev_level=1.0,
        stddev_trend=1.0,
        stddev_freq_seasonal=None,
        seed=123456789,
        **kwargs
    ):
        super(UnobservedComponentsSimulator, self).__init__(seed=seed, **kwargs)

        self.steps = int(steps)
        self.level = level
        self.trend = trend

        if freq_seasonal:
            self.freq_seasonal_periods = [d["period"] for d in freq_seasonal]
            self.freq_seasonal_harmonics = [
                d.get("harmonics", int(np.floor(d["period"] / 2)))
                for d in freq_seasonal
            ]
        else:
            self.freq_seasonal_periods = []
            self.freq_seasonal_harmonics = []
        self.freq_seasonal = any(x > 0 for x in self.freq_seasonal_periods)

        self.start_param_level = float(start_param_level)
        self.stddev_level = float(stddev_level)
        self.stddev_trend = float(stddev_trend) if not self.trend else float(0.0)

        if stddev_freq_seasonal is None:
            self.stddev_freq_seasonal = [1.0] * len(self.freq_seasonal_periods)
        else:
            if len(stddev_freq_seasonal) != len(freq_seasonal):
                msg = "Length of stddev_freq_seasonal must be equal to the one of freq_seasonal."
                raise ValueError(msg)
            self.stddev_freq_seasonal = [float(d) for d in stddev_freq_seasonal]

        if trend and not level:
            msg = "Trend component specified without level component; deterministic level component added."
            warn(msg, UserWarning)

        if self.freq_seasonal:
            for p in self.freq_seasonal_periods:
                if p < 2:
                    msg = (
                        "Simulated data on frequency domain seasonal component"
                        " must have a seasonal period of at least 2."
                    )
                    raise ValueError(msg)

        if exog_params is not None:
            self.regression = True
            exog_params = [float(b) for b in exog_params]
        else:
            self.regression = False
        self.exog_param = exog_params

    def simulate(self):
        return self._simulate()

    def _simulate(self):
        simulate_result = UnobservedComponentsSimulatorResult()

        trend = None
        if self.level:
            trend = self._simulate_trend_term(
                steps=self.steps,
                start_param=self.start_param_level,
                stddev_level=self.stddev_level,
                stddev_trend=self.stddev_trend,
            )
        simulate_result.trend = trend

        freq_seasonal = None
        if self.freq_seasonal:
            n_wave = len(self.freq_seasonal_periods)
            freq_seasonal = np.zeros((self.steps, n_wave))
            for i in range(n_wave):
                period = self.freq_seasonal_periods[i]
                harmonics = self.freq_seasonal_harmonics[i]
                stddev_freq_seasonal = self.stddev_freq_seasonal[i]
                total_cycles = np.ceil(self.steps / period)

                sea = self._simulate_seasonal_term(
                    periodicity=period,
                    total_cycles=total_cycles,
                    stddev_freq_seasonal=stddev_freq_seasonal,
                    harmonics=harmonics,
                )
                freq_seasonal[:, i] = sea[: self.steps]
        simulate_result.freq_seasonal = freq_seasonal

        exog = None
        reg = None
        if self.regression:
            exog = self._simulate_exog(
                steps=self.steps, exog_params=len(self.exog_param)
            )
            reg = exog @ np.diag(self.exog_param)
        simulate_result.exog = exog
        simulate_result.reg = reg

        trend = trend if trend is not None else 0
        freq_seasonal = freq_seasonal.sum(axis=1) if freq_seasonal is not None else 0
        reg = reg.sum(axis=1) if reg is not None else 0
        endog = trend + freq_seasonal + reg
        simulate_result.endog = endog
        return simulate_result

    def _simulate_trend_term(self, steps, start_param, stddev_level, stddev_trend):
        series = np.zeros(steps)
        level_t = start_param + stddev_level * self.prng.standard_normal()
        trend_t = (
            0 if stddev_trend is None else stddev_trend * self.prng.standard_normal()
        )

        for t in range(steps):
            trend_tp = trend_t + stddev_trend * self.prng.standard_normal()
            level_tp = level_t + trend_tp + stddev_level * self.prng.standard_normal()
            series[t] = level_tp
            trend_t = trend_tp
            level_t = level_tp

        return series

    def _simulate_seasonal_term(
        self,
        periodicity,
        total_cycles,
        stddev_freq_seasonal,
        harmonics,
    ):
        """Simulate seasonality component data.

        Parameters
        ----------
        periodicity : int
            Base cycle.
        total_cycles : int
            Number of cycles.
        stddev_freq_seasonal : float
            Standard deviation on seasonality component noise.
        harmonics : {int, None}
            The numbers of harmonics

        Return
        ------
        series : numpy.ndarray
            Simulated data array.
        """
        # 基本周期 (periodicity) × 周期数 (total_cycles) で時系列長を設定.
        # 時系列長 (duration) が整数であることも同時にチェック.
        duration = periodicity * total_cycles
        assert duration == int(duration)
        duration = int(duration)

        lambda_p = 2 * np.pi / float(periodicity)

        gamma_jt = stddev_freq_seasonal * self.prng.standard_normal(harmonics)
        gamma_star_jt = stddev_freq_seasonal * self.prng.standard_normal(harmonics)

        total_timestamps = 100 * duration
        series = np.zeros(total_timestamps)
        for t in range(total_timestamps):
            gamma_jtp1 = np.zeros_like(gamma_jt)
            gamma_star_jtp1 = np.zeros_like(gamma_star_jt)
            for j in range(1, harmonics + 1):
                cos_j = np.cos(lambda_p * j)
                sin_j = np.sin(lambda_p * j)
                gamma_jtp1[j - 1] = (
                    gamma_jt[j - 1] * cos_j
                    + gamma_star_jt[j - 1] * sin_j
                    + stddev_freq_seasonal * self.prng.standard_normal()
                )
                gamma_star_jtp1[j - 1] = (
                    -gamma_jt[j - 1] * sin_j
                    + gamma_star_jt[j - 1] * cos_j
                    + stddev_freq_seasonal * self.prng.standard_normal()
                )
            series[t] = np.sum(gamma_jtp1)
            gamma_jt = gamma_jtp1
            gamma_star_jt = gamma_star_jtp1

        return series[-duration:]

    def _simulate_exog(self, steps, exog_params):
        exog = np.zeros((steps, exog_params))
        for i in range(exog_params):
            exog[:, i] = self.prng.integers(low=0, high=2, size=steps)
        return exog


class UnobservedComponentsSimulatorResult(BaseData):
    """"""

    _trend = None
    _freq_seasonal = None
    _exog = None
    _reg = None
    _endog = None

    def __init__(self, **kwargs):
        super(UnobservedComponentsSimulatorResult, self).__init__(**kwargs)
        self.is_pandas = False

    @property
    def trend(self):
        return self._trend

    @trend.setter
    def trend(self, value):
        self._trend = value

    @property
    def freq_seasonal(self):
        return self._freq_seasonal

    @freq_seasonal.setter
    def freq_seasonal(self, value):
        self._freq_seasonal = value

    @property
    def exog(self):
        return self._exog

    @exog.setter
    def exog(self, value):
        self._exog = value

    @property
    def reg(self):
        return self._reg

    @reg.setter
    def reg(self, value):
        self._reg = value

    @property
    def endog(self):
        return self._endog

    @endog.setter
    def endog(self, value):
        self._endog = value

    @property
    def common_index(self):
        return pd.Index(range(self.nobs))

    @property
    def nobs(self):
        return len(self.endog)

    def convert_pandas(self):
        if not self.is_pandas:
            if self.trend is not None:
                self.trend = self._as_pandas_from_ndarray(
                    obj=self.trend,
                    index=self.common_index,
                    key="trend",
                    pandas_type="series",
                )

            if self.freq_seasonal is not None:
                key = [
                    "freq_seasonal_{}".format(i)
                    for i in range(self.freq_seasonal.shape[1])
                ]
                self.freq_seasonal = self._as_pandas_from_ndarray(
                    obj=self.freq_seasonal,
                    index=self.common_index,
                    key=key,
                    pandas_type="dataframe",
                )

            if self.exog is not None:
                exog_key = ["x{}".format(i) for i in range(self.exog.shape[1])]
                self.exog = self._as_pandas_from_ndarray(
                    obj=self.exog,
                    index=self.common_index,
                    key=exog_key,
                    pandas_type="dataframe",
                )

                reg_key = ["reg{}".format(i) for i in range(self.reg.shape[1])]
                self.reg = self._as_pandas_from_ndarray(
                    obj=self.reg,
                    index=self.common_index,
                    key=reg_key,
                    pandas_type="dataframe",
                )

            if self.endog is not None:
                self.endog = self._as_pandas_from_ndarray(
                    obj=self.endog,
                    index=self.common_index,
                    key="endog",
                    pandas_type="series",
                )
            self.is_pandas = True

    def convert_ndarray(self):
        if self.is_pandas:
            if self.trend is not None:
                self.trend = self._as_ndarray_from_pandas(self.trend)

            if self.freq_seasonal is not None:
                self.freq_seasonal = self._as_ndarray_from_pandas(self.freq_seasonal)

            if self.exog is not None:
                self.exog = self._as_ndarray_from_pandas(self.exog)
                self.reg = self._as_ndarray_from_pandas(self.reg)

            if self.endog is not None:
                self.endog = self._as_ndarray_from_pandas(self.endog)

            self.is_pandas = False
