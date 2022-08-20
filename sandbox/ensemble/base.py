"""Base class for ensemble-based estimators."""
import functools
from abc import ABCMeta

from sandbox.datamodel.base import BaseData, SupervisedModelDataset


# ToDo: 時系列とほぼ同じ形なので, 集約できないか検討が必要?
class _BaseEnsembleModelMeta(ABCMeta):
    """Metaclass for model estimators."""

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
        def _wrapped(self, X, y, **kwargs):

            # Both X and y must not be None.
            if (X is None) or (y is None):
                msg = "X and y must not be None."
                raise ValueError(msg)

            # ToDo: dask形式の場合を検討.
            data = SupervisedModelDataset(X, y)
            X, y = data.X.to_pandas(), data.y.to_pandas()

            child_fit(self, X, y, **kwargs)

            return self

        return _wrapped

    @classmethod
    def wrapper_predict(mcs, child_predict):
        """Common pre-/post-process in predict method."""

        @functools.wraps(child_predict)
        def _wrapped(self, X, **kwargs):

            # X must not be None.
            if X is None:
                msg = "X must not be None."
                raise ValueError(msg)

            # ToDo: dask形式の場合を検討.
            data = BaseData(X)
            X = data.to_pandas()

            pred = child_predict(self, X, **kwargs)

            return pred

        return _wrapped


class _NotInstalledModel:
    """Dummy class which is inherited in case that the required package offering
    the estimator compatible with scikit-learn cannot be imported.
    """

    _required_package = None

    def __init__(self):
        msg = "Cannot import the following required package: {}"
        raise ImportError(msg.format(self._required_package))

    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def predict(self, X, **kwargs):
        raise NotImplementedError
