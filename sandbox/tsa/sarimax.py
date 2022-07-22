""""""
import numpy as np
import pandas as pd
import pmdarima as pm

from sandbox.tsa.base import BaseTimeSeriesModel
from sandbox.utils.validation import check_2d_dataframe


# ToDo: in case of existence of regressors and set the parameters in fit method.
# ToDo: test script
class SARIMAXModel(BaseTimeSeriesModel):
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
        super(SARIMAXModel, self).__init__()
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
        self.model_fitted_ = None
        self.params_ = None

    def fit(self, X, y=None, **kwargs):

        self.X_train_ = X
        self.y_train_ = y

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = pm.auto_arima(
                y=self.y_train_,
                X=self.X_train_,
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

        self.model_fitted_ = model.arima_res_
        self.params_ = dict(
            zip(
                self.model_fitted_.param_names,
                self.model_fitted_.params,
            )
        )

        return self

    def predict(self, X, return_conf_int=False, alpha=0.95):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = check_2d_dataframe(X)
        else:
            if not isinstance(X, pd.Index):
                msg = "X must be pd.Index, pd.DataFrame or pd.Series."
                raise TypeError(msg)

        pred_is, conf_int_is = self._in_sample_predict(
            X=X, return_conf_int=return_conf_int, alpha=alpha
        )
        pred_oos, conf_int_oos = self._out_of_sample_forecast(
            X=X, return_conf_int=return_conf_int, alpha=alpha
        )

        is_in_sample = pred_is is not None
        out_of_sample = pred_oos is not None

        if is_in_sample and out_of_sample:
            pred = pd.concat([pred_is, pred_oos])
            conf_int = (
                pd.concat([conf_int_is, conf_int_oos]) if return_conf_int else None
            )
        elif is_in_sample:
            pred = pred_is.copy()
            conf_int = conf_int_is.copy() if return_conf_int else None
        else:
            pred = pred_oos.copy()
            conf_int = conf_int_oos.copy() if return_conf_int else None

        if return_conf_int:
            pred = pd.DataFrame(pred).join(conf_int)

        return pred

    @classmethod
    def _get_exog(cls, X, index=None):
        if isinstance(X, pd.Index):
            exog = None
        elif isinstance(X, (pd.DataFrame, pd.Series)):
            exog = check_2d_dataframe(X)
            if index:
                exog = exog[exog.index.isin(index)]
        else:
            msg = "X must be pd.Index, pd.DataFrame or pd.Series."
            raise TypeError(msg)
        return exog

    def _out_of_sample_forecast(self, X, return_conf_int=False, alpha=0.95):
        pred_oos = None
        conf_int_oos = None

        index_oos = self._in_out_of_sample_index(X)[1]
        if len(index_oos) > 0:
            start = len(self.y_train_)
            end = start + len(index_oos) - 1
            exog = self._get_exog(X, index=index_oos)
            predict_result_oos = self.model_fitted_.get_prediction(
                start=start, end=end, exog=exog
            )
            pred_oos = pd.Series(
                predict_result_oos.predicted_mean,
                index=index_oos,
                name="predicted_mean",
            )

            conf_int_oos = None
            if return_conf_int:
                conf_int_oos = pd.DataFrame(
                    predict_result_oos.conf_int(alpha=1 - alpha),
                    index=index_oos,
                    columns=["lower", "upper"],
                )

        return pred_oos, conf_int_oos

    def _in_sample_predict(self, X, return_conf_int=False, alpha=0.95):
        pred_is = None
        conf_int_is = None

        index_oos = self._in_out_of_sample_index(X)[0]
        if len(index_oos) > 0:
            predict_result_is = self.model_fitted_.get_prediction()
            pred_is = pd.Series(
                predict_result_is.predicted_mean,
                index=self.y_train_.index,
                name="predicted_mean",
            )
            pred_is = pred_is[pred_is.index.isin(index_oos)]

            conf_int_is = None
            if return_conf_int:
                conf_int_is = pd.DataFrame(
                    predict_result_is.conf_int(alpha=1 - alpha),
                    index=self.y_train_.index,
                    columns=["lower", "upper"],
                )
                conf_int_is = conf_int_is[conf_int_is.index.isin(index_oos)]

        return pred_is, conf_int_is

    def _in_out_of_sample_index(self, X):
        """入力されたXにおいてどこまでがin-sampleでどこからがout-of-sampleのindexかを返す."""
        if self.model_fitted_ is None:
            return None

        if isinstance(X, pd.Index):
            _index = X.copy()
        elif isinstance(X, (pd.DataFrame, pd.Series)):
            _index = X.index.copy()
        else:
            msg = "X must be pd.Index, pd.DataFrame or pd.Series."
            raise TypeError(msg)

        arr_is = np.intersect1d(self.y_train_.index.to_numpy(), _index.to_numpy())
        arr_oos = np.setdiff1d(_index.to_numpy(), arr_is)

        index_is = pd.Index(arr_is)
        index_is.name = self.y_train_.index.name
        index_oos = pd.Index(arr_oos)
        index_oos.name = self.y_train_.index.name

        return index_is, index_oos

    def score(self, X, y, type="mape"):
        from sklearn.metrics import mean_absolute_percentage_error

        y_pred = self.predict(X)
        return 1 - mean_absolute_percentage_error(y, y_pred)
