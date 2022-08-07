from typing import Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Index, Series
from pandas.core.indexes.api import default_index

try:
    from dask import array as da
    from dask import dataframe as dd
except ImportError:
    dd = da = None

if dd:
    StructuralDataType = Union[
        DataFrame, Series, Index, ndarray, dd.DataFrame, dd.Series, dd.Index, da.Array
    ]
else:
    StructuralDataType = Union[DataFrame, Series, Index, ndarray]


class BaseData:
    r"""Base data class.

    Parameters
    ----------
    data : StructuralDataType
        Input data. Supported format is `pandas.DataFrame`, `pandas.Series`, `pandas.Index`, `numpy.ndarray`,
        `dask.dataframe.DataFrame`, `dask.dataframe.Series`, `dask.dataframe.Index`, `dask.array.Array`.

    Warnings
    --------
    In case of `Pandas` and `NumPy` format, :py:attr:`values <sandbox.datamodel.base.BaseData.values>`
    returns the actual data. However, the format of `Dask` returns before-compute objects, and if you want
    to get the actual data, you need to :py:func:`compute <dask.dataframe.DataFrame.compute>`.

    """

    def __init__(self, data: StructuralDataType) -> None:
        # pandas, numpy, dask のみ対応
        if not self._is_incorrect_data(data):
            msg = (
                "Specified data format {} is unsupported, and must be as follows: pandas.DataFrame, "
                "pandas.Series, pandas.Index, numpy.ndarray, dask.dataframe.DataFrame, "
                "dask.dataframe.Series, dask.dataframe.Index, dask.array.Array.".format(
                    str(type(data))
                )
            )
            raise TypeError(msg)
        self.data = data
        self._names = self.names

    def __repr__(self) -> str:
        return self.data.__repr__()

    def __len__(self) -> int:
        return self.data.__len__()

    @staticmethod
    def _is_incorrect_data(data: StructuralDataType) -> bool:
        if isinstance(data, (pd.DataFrame, pd.Series, pd.Index)):
            _is = True
        elif isinstance(data, np.ndarray):
            _is = True
        elif dd and isinstance(data, (dd.DataFrame, dd.Series, dd.Index)):
            _is = True
        elif da and isinstance(data, da.Array):
            _is = True
        else:
            _is = False
        return _is

    @property
    def nobs(self) -> int:
        """Number of observations."""
        return len(self.data)

    @property
    def nparams(self) -> int:
        """Number of parameters."""
        if self.data.ndim == 1:
            return 1
        else:
            return self.shape[1]

    @property
    def values(self) -> Union[ndarray, da.Array]:
        """Return a Numpy representation of data.
        In case of Dask format, return a Dask.array.Array.
        """
        if isinstance(self.data, (np.ndarray, da.Array)):
            return self.data
        if isinstance(
            self.data,
            (pd.DataFrame, pd.Series, pd.Index, dd.DataFrame, dd.Series, dd.Index),
        ):
            return self.data.values

    @property
    def index(self) -> Union[Index, dd.Index]:
        """Return the index (row labels) of data."""
        if isinstance(self.data, (pd.Index, dd.Index, np.ndarray, da.Array)):
            return default_index(len(self.data))
        if isinstance(self.data, (pd.DataFrame, pd.Series, dd.DataFrame, dd.Series)):
            return self.data.index

    @property
    def names(self) -> Index:
        """Returns the column labels of data."""
        if isinstance(self.data, (pd.DataFrame, dd.DataFrame)):
            return self.data.columns
        if isinstance(self.data, (pd.Series, pd.Index, dd.Series, dd.Index)):
            name = self.data.name
            if name is None:
                name = "name_0"
            return Index([name])
        if isinstance(self.data, (np.ndarray, da.Array)):
            names = list()
            if self.nparams == 1:
                names.append("name_0")
                return Index(names)
            else:
                for i in range(self.nparams):
                    names.append("name_{}".format(i))
                return Index(names)

    @names.setter
    def names(self, value) -> None:
        if len(value) != self.nparams:
            msg = "Specified names length must be {0}, not {1}.".format(
                self.nparams, len(value)
            )
            raise ValueError(msg)
        if hasattr(self.data, "name"):
            self.data.name = value[0]
        if hasattr(self.data, "columns"):
            self.data.columns = Index(value)
        self._names = Index(value)

    @property
    def shape(self) -> tuple[int, int]:
        """Return a tuple representing the dimensionality of data."""
        return self.data.shape

    def to_pandas(self) -> Union[DataFrame, Series, Index]:
        """Convert the BaseData to Pandas dataframe.

        Returns
        -------
        {pandas.DataFrame, pandas.Series, pandas.Index}
        """
        if isinstance(self.data, (pd.DataFrame, pd.Series, pd.Index)):
            return self.data
        if isinstance(self.data, (dd.DataFrame, dd.Series, dd.Index)):
            return self.data.compute()
        if isinstance(self.data, np.ndarray):
            return DataFrame(self.data, index=self.index, columns=self.names)
        if isinstance(self.data, da.Array):
            return DataFrame(self.data.compute(), index=self.index, columns=self.names)

    def to_numpy(self) -> ndarray:
        """Convert the BaseData to NumPy array.

        Returns
        -------
        numpy.ndarray
        """
        if isinstance(self.data, np.ndarray):
            return self.data
        if isinstance(self.data, (pd.DataFrame, pd.Series, pd.Index)):
            return self.data.values
        if isinstance(self.data, da.Array):
            return self.data.compute()
        if isinstance(self.data, (dd.DataFrame, dd.Series, dd.Index)):
            return self.data.values.compute()

    def to_dask_dataframe(
        self, **from_pandas_kwargs
    ) -> Union[dd.DataFrame, dd.Series, dd.Index]:
        """Convert the BaseData to Dask dataframe.

        Parameters
        ----------
        from_pandas_kwargs : dict
            :py:func:`from_pandas <dask.dataframe.from_pandas>` in Dask converts data, and `from_pandas_kwargs`
            is the argument which is used in the function.

        Returns
        -------
        {dask.dataframe.DataFrame, dask.dataframe.Series, dask.dataframe.Index}

        See Also
        --------
        dask.dataframe.from_pandas
        """
        if isinstance(self.data, dd.DataFrame):
            return self.data
        if isinstance(self.data, (pd.DataFrame, pd.Series)):
            return dd.from_pandas(self.data, **from_pandas_kwargs)
        if isinstance(self.data, (np.ndarray, da.Array)):
            return dd.from_pandas(self.to_pandas(), **from_pandas_kwargs)

    def to_dask_numpy(self, **from_array_kwargs) -> da.Array:
        """Convert the BaseData to Dask array.

        Parameters
        ----------
        from_array_kwargs : dict
            :py:func:`from_array <dask.array.from_array>` in Dask converts data, and `from_array_kwargs`
            is the argument which is used in the function.

        Returns
        -------
        dask.array.Array

        See Also
        --------
        dask.array.from_array
        """
        if isinstance(self.data, da.Array):
            return self.data
        if isinstance(self.data, np.ndarray):
            return da.from_array(self.data, **from_array_kwargs)
        if isinstance(
            self.data,
            (pd.DataFrame, pd.Series, pd.Index, dd.DataFrame, dd.Series, dd.Index),
        ):
            return da.from_array(self.to_numpy(), **from_array_kwargs)


class BaseModelDataset:
    """Base class for data model of algorithm.

    Parameters
    ----------
    X : StructuralDataType
        Training data. In classification model, it is for classifying and clustering the data.
        In regression model, it is feature vectors or matrix, but can be ignored when the regression
        components are not defined in the case of time series analysis.
    y : StructuralDataType
        Target values. If algorithm is unsupervised, this should be ignored.

    """

    def __init__(self, X, y):
        self.X = BaseData(X) if X is not None else None
        self.y = BaseData(y) if y is not None else None
        self._X_name = self.X.names if X is not None else None
        self._y_name = self.y.names if y is not None else None

        if not self._has_non_None_X_y():
            msg = "X and y must not be both None."
            raise ValueError(msg)

        if self._has_both_X_y_defined():
            if not self._has_same_length_in_X_y():
                msg = "X and y must be same length."
                raise ValueError(msg)
            if not self._has_same_index_in_X_y():
                msg = "X and y must be same index."
                raise ValueError(msg)

    def __repr__(self):
        return "X:\n{0}\n\ny:\n{1}".format(self.X, self.y)

    def _has_both_X_y_defined(self):
        if self.X is not None and self.y is not None:
            return True
        else:
            return False

    def _has_non_None_X_y(self):
        if not (self.X is None and self.y is None):
            return True
        else:
            return False

    def _has_same_length_in_X_y(self):
        if len(self.X) == len(self.y):
            return True
        else:
            return False

    def _has_same_index_in_X_y(self):
        return self.X.index.equals(self.y.index)

    @property
    def nobs(self):
        """Number of observations."""
        if self.X is not None:
            return self.X.nobs
        else:
            return self.y.nobs

    @property
    def nfeatures(self):
        """Number of feature variables."""
        return self.X.nparams

    @property
    def common_index(self):
        """Common index of X and y"""
        if self.X is not None:
            return self.X.index
        else:
            return self.y.index

    @property
    def X_name(self):
        """X name columns"""
        if self.X:
            return self.X.names
        else:
            return None

    @X_name.setter
    def X_name(self, value) -> None:
        if self.X:
            if len(value) != self.nfeatures:
                msg = "Specified X_names length must be {0}, not {1}.".format(
                    self.nfeatures, len(value)
                )
                raise ValueError(msg)
            if hasattr(self.X.data, "name"):
                self.X.data.name = value[0]
            if hasattr(self.X.data, "columns"):
                self.X.data.columns = Index(value)
            self._X_name = Index(value)

    @property
    def y_name(self):
        """y name."""
        if self.y:
            return self.y.names
        else:
            return None

    @y_name.setter
    def y_name(self, value) -> None:
        if self.y:
            if len(value) != 1:
                msg = "Specified y_names length must be 1, not {}.".format(len(value))
                raise ValueError(msg)
            if hasattr(self.y.data, "name"):
                self.y.data.name = value[0]
            if hasattr(self.y.data, "columns"):
                self.y.data.columns = Index(value)
            self._y_name = Index(value)


class SupervisedModelDataset(BaseModelDataset):
    """Base class for data model for supervised model.

    Parameters
    ----------
    X : StructuralDataType
        The feature vectors or matrix. If regression is not defined, you should
        handle the position of X as the one of y.
    y : {StructuralDataType, None}, optional
        Target values. If regression is not defined, ignore that.

    """

    def __init__(self, X, y=None):
        if X is not None and y is None:
            X, y = y, X
        super(SupervisedModelDataset, self).__init__(X=X, y=y)

    def get_index_and_values_from_X_pred(self, X_pred):
        """Get index and features design matrix from X_pred
        that is assumed to be data of predictive range.

        Parameters
        ----------
        X_pred : {array_like, int}
            Data to split into index and design matrix.

        Returns
        -------
        index : pandas.Index
            Index split into.
        X: {numpy.ndarray, None}
            Design matrix split into.
        """
        if isinstance(X_pred, int):
            obs_X = int(X_pred)
            start = len(self.common_index)
            stop = start + obs_X
            index = pd.RangeIndex(start=start, stop=stop, step=1)
            values = None
        else:
            X_pred = BaseData(X_pred)
            index = X_pred.index
            values = X_pred.values
        return index, values


class UnsupervisedModelDataset(BaseModelDataset):
    ...


class BaseDataSimulator:
    """Base class for data simulator."""

    def __init__(self, seed=123456789, **kwargs):
        self.prng = np.random.default_rng(seed=seed)
        self.seed = seed
