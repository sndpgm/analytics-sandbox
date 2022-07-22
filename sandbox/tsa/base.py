import functools
from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator

from sandbox.utils.validation import check_X_y


class _BaseModelMeta(ABCMeta):
    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)

        # fit / predict methods are wrapped.
        new_class.fit = cls.wrapper_fit(new_class.fit)
        new_class.predict = cls.wrapper_predict(new_class.predict)

        return new_class

    @classmethod
    def wrapper_fit(cls, child_fit):
        """Common pre-/post-process in fit method."""

        @functools.wraps(child_fit)
        def _wrapped(self, X, y=None, **kwargs):
            if X is not None and y is None:
                X, y = y, X
            X, y = check_X_y(X, y)
            child_fit(self, X=X, y=y, **kwargs)
            return self

        return _wrapped

    @classmethod
    def wrapper_predict(cls, child_predict):
        """Common pre-/post-process in predict method."""

        @functools.wraps(child_predict)
        def _wrapped(self, X, **kwargs):
            pred = child_predict(self, X, **kwargs)
            return pred

        return _wrapped


class BaseTimeSeriesModel(BaseEstimator, metaclass=_BaseModelMeta):
    """Base class for time series model."""

    @abstractmethod
    def fit(self, X, y=None, **kwargs):
        """Fit time series model."""

    @abstractmethod
    def predict(self, X, **kwargs):
        """Predict using time series model."""

    def conf_int(self, X, alpha):
        """Construct confidence interval for the fitted parameters."""
        pass

    def score(self, X, y):
        """Return the coefficient of determination of the prediction."""
        pass
