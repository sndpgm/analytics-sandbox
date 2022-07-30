"""Base classes for time series estimators."""
import functools
from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator

from sandbox.metrics import score


class _BaseModelMeta(ABCMeta):
    """Meta class for model estimators."""

    def __new__(mcs, name, bases, attrs):
        new_class = super().__new__(mcs, name, bases, attrs)

        # fit / predict methods are wrapped.
        new_class.fit = mcs.wrapper_fit(new_class.fit)
        new_class.predict = mcs.wrapper_predict(new_class.predict)

        return new_class

    @classmethod
    def wrapper_fit(mcs, child_fit):
        """Common pre-/post-process in fit method."""

        @functools.wraps(child_fit)
        def _wrapped(self, X, y=None, **kwargs):
            child_fit(self, X=X, y=y, **kwargs)
            return self

        return _wrapped

    @classmethod
    def wrapper_predict(mcs, child_predict):
        """Common pre-/post-process in predict method."""

        @functools.wraps(child_predict)
        def _wrapped(self, X, **kwargs):
            pred = child_predict(self, X, **kwargs)
            return pred

        return _wrapped


class BaseTimeSeriesModel(BaseEstimator, metaclass=_BaseModelMeta):
    """Base class for time series model."""

    @abstractmethod
    def fit(self, X, y=None):
        """Fit time series model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict using time series model."""
        pass

    def conf_int(self, X, alpha):
        """Construct confidence interval for the fitted parameters."""
        pass

    def score(self, X, y, scorer="r2", **kwargs):
        """Return the coefficient of determination of the prediction."""
        y_pred = self.predict(X)
        if scorer == "r2":
            return score.r2_score(y_true=y, y_pred=y_pred, **kwargs)
        elif scorer == "mape":
            return score.mape_score(y_true=y, y_pred=y_pred, **kwargs)
        else:
            msg = "Specified `scorer` does not support."
            raise ValueError(msg)
