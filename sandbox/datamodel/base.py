from __future__ import annotations

from abc import ABCMeta

import numpy as np
import pandas as pd

import sandbox.utils.validation as val


# ===================================
# ベースのデータクラス.
# ===================================
class BaseData:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    @staticmethod
    def _get_1d_arr(obj, default_name="y"):
        return get_1d_arr(obj=obj, default_name=default_name)

    @staticmethod
    def _get_2d_arr(obj, default_name="x"):
        return get_2d_arr(obj=obj, default_name=default_name)

    @staticmethod
    def _get_index(obj):
        if val.is_dataframe_or_series(obj):
            index = obj.index
        else:
            index = pd.Index(range(len(obj)))
        return index

    @staticmethod
    def _as_pandas_from_ndarray(obj, index, key, pandas_type="dataframe"):
        _obj = None
        if val.is_ndarray(obj):
            if pandas_type == "dataframe":
                _obj = pd.DataFrame(obj, index=index, columns=key)
            if pandas_type == "series":
                _obj = pd.Series(obj, index=index, name=key)
        else:
            _obj = obj
        return _obj

    @staticmethod
    def _as_ndarray_from_pandas(obj):
        _obj = None
        if val.is_dataframe_or_series(obj):
            _obj = obj.to_numpy()
        else:
            _obj = obj
        return _obj


# ===================================
# ベースのモデルデータクラス.
# ===================================
class BaseModelData(BaseData):
    def __init__(self, X, y, **kwargs):
        super(BaseModelData, self).__init__()
        self._check_X_or_y_is_not_none(X, y)
        self._check_X_y_length(X, y)
        self.__dict__.update(**kwargs)
        self.orig_X = X.copy() if X is not None else None
        self.orig_y = y.copy() if y is not None else None
        self.orig_is_pandas = val.is_using_padnas(self.orig_X, self.orig_y)
        self.X, self._X_name = self._get_2d_arr(X)
        self.y, self._y_name = self._get_1d_arr(y)
        self._common_index = self._get_index_from_X_and_y()
        self.is_pandas = False
        self.X_pred = None

    @staticmethod
    def _both_X_y_are_none(X, y):
        if X is None and y is None:
            return True
        else:
            return False

    def _check_X_or_y_is_not_none(self, X, y):
        if self._both_X_y_are_none(X, y):
            msg = "Both X and y must not be None."
            raise ValueError(msg)

    @staticmethod
    def _check_X_y_length(X, y):
        len_X = None
        len_y = None
        if X is not None:
            len_X = len(X)
        if y is not None:
            len_y = len(y)
        if len_X is not None and len_y is not None:
            is_same_length = len_X == len_y
            if not is_same_length:
                msg = "The length of X is not same as the one of y."
                raise ValueError(msg)

    @property
    def nobs(self):
        if self.X is not None:
            nobs = len(self.X)
        else:
            nobs = len(self.y)
        return nobs

    def _get_index_from_X_and_y(self):
        index = None
        if self.orig_X is not None:
            index = self._get_index(self.orig_X)
        if self.orig_y is not None:
            index = self._get_index(self.orig_y)
        return index

    @property
    def common_index(self) -> pd.Index:
        return self._common_index

    @common_index.setter
    def common_index(self, value):
        self._common_index = value

    @property
    def X_name(self) -> list[str] | None:
        return self._X_name

    @X_name.setter
    def X_name(self, value):
        if not isinstance(value, list):
            msg = "X_name must be list type."
            raise TypeError(msg)
        is_string = all(isinstance(d, str) for d in value)
        if not is_string:
            msg = "All elements of X_name must be str."
            raise TypeError(msg)
        self._X_name = value

    @property
    def y_name(self) -> str | None:
        return self._y_name

    @y_name.setter
    def y_name(self, value):
        if not isinstance(value, str):
            msg = "y_name must be str."
            raise TypeError(msg)
        self._y_name = value

    def _convert_pandas(self):
        X = None
        y = None
        is_pandas = self.is_pandas
        if not self.is_pandas:
            if self.X is not None:
                X = self._as_pandas_from_ndarray(
                    obj=self.X,
                    index=self.common_index,
                    key=self.X_name,
                    pandas_type="dataframe",
                )
            if self.y is not None:
                y = self._as_pandas_from_ndarray(
                    obj=self.y,
                    index=self.common_index,
                    key=self.y_name,
                    pandas_type="series",
                )
            is_pandas = True
        else:
            X, y = self.X, self.y
        return X, y, is_pandas

    def convert_pandas(self):
        self.X, self.y, self.is_pandas = self._convert_pandas()

    def _convert_ndarray(self):
        X = None
        y = None
        is_pandas = self.is_pandas
        if self.is_pandas:
            if self.X is not None:
                X = self._as_ndarray_from_pandas(self.X)
            if self.y is not None:
                y = self._as_ndarray_from_pandas(self.y)
            is_pandas = False
        else:
            X, y = self.X, self.y
        return X, y, is_pandas

    def convert_ndarray(self):
        self.X, self.y, self.is_pandas = self._convert_ndarray()


# ===================================
# 教師あり学習器のデータクラス.
# ===================================
class SupervisedModelData(BaseModelData, metaclass=ABCMeta):
    def __init__(self, X, y=None, **kwargs):
        if X is not None and y is None:
            X, y = y, X
        super(SupervisedModelData, self).__init__(X=X, y=y, **kwargs)

    def split_index_and_X_from_X_pred(self, X_pred):
        index = None
        X = None

        # If the defined model has no regression components,
        # only the index of the range to be predicted is returned
        # from input (pandas.Index or int).
        if val.is_index(X_pred):
            index = X_pred
        elif isinstance(X_pred, (int, float)):
            obs_X = int(X_pred)
            start = self.common_index.__len__()
            stop = start + obs_X
            index = pd.RangeIndex(start=start, stop=stop, step=1)

        # Otherwise, the index and the numpy array of design matrix
        # on regression are returned from input (pandas.DataFrame or numpy.ndarray).
        elif val.is_dataframe_or_series(X_pred):
            index = X_pred.index
            X = self._get_2d_arr(X_pred)[0]
            n_X = X.shape[1]
            if n_X != len(self.X_name):
                msg = "The components of X_pred does not match the ones of trained X."
                raise ValueError(msg)
        elif val.is_arraylike(X_pred):
            X = self._get_2d_arr(X_pred)[0]
            obs_X, n_X = X.shape
            if n_X != len(self.X_name):
                msg = "The components of X_pred does not match the ones of trained X."
                raise ValueError(msg)
            start = self.common_index.__len__()
            stop = start + obs_X
            index = pd.RangeIndex(start=start, stop=stop, step=1)

        else:
            msg = (
                "When your model needs the explanatory components, X_pred must be"
                " pandas.DataFrame or numpy.ndarray. Otherwise, X_pred must be"
                " pandas.Index or int to express the steps to be predicted."
            )
            raise ValueError(msg)

        return index, X


# ===================================
# 教師なし学習器のデータクラス.
# ===================================
class UnsupervisedModelData(BaseModelData, metaclass=ABCMeta):
    def __init__(self, X, **kwargs):
        y = None
        super(UnsupervisedModelData, self).__init__(X=X, y=y, **kwargs)


# ===================================
# テストデータ作成・シミュレータークラス.
# ===================================
class BaseDataSimulator:
    def __init__(self, seed=123456789, **kwargs):
        self.prng = np.random.default_rng(seed=seed)
        self.seed = seed


def get_1d_arr(obj, default_name="y"):
    obj_arr = None
    obj_name = None
    if obj is not None:
        if val.is_dataframe_or_series(obj):
            if val.is_dataframe(obj) and obj.shape[1] > 1:
                msg = "Input obj must be one variable."
                raise ValueError(msg)

            if val.is_dataframe(obj):
                obj_name = obj.columns.astype(str).to_list()[0]
            else:
                obj_name = default_name if obj.name is None else obj.name

            obj_arr = obj.values.squeeze()

        if val.is_ndarray(obj) or isinstance(obj, (list, tuple)):
            obj_arr = np.asarray(obj)
            obj_name = default_name

    return obj_arr, obj_name


def get_2d_arr(obj, default_name="x"):
    obj_arr = None
    obj_name = None
    if obj is not None:
        if val.is_dataframe_or_series(obj):
            if val.is_dataframe(obj):
                obj_arr = obj.values
                obj_name = obj.columns.astype(str).to_list()

            if val.is_series(obj):
                obj_arr = obj.values[:, None]
                n_obj = 1
                obj_name = (
                    [
                        "{default_name}{i}".format(default_name=default_name, i=i)
                        for i in range(n_obj)
                    ]
                    if obj.name is None
                    else [obj.name]
                )

        if val.is_ndarray(obj) or isinstance(obj, (list, tuple)):
            obj_arr = np.asarray(obj)
            n_obj = obj_arr.shape[1]
            obj_name = [
                "{default_name}{i}".format(default_name=default_name, i=i)
                for i in range(n_obj)
            ]

    return obj_arr, obj_name
