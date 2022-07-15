"""Datasets utils modules."""
from __future__ import annotations

from os.path import abspath, dirname, join
from typing import List

from pandas import DataFrame, Series, read_csv


class Dataset(dict):
    """Dataset class."""

    def __init__(self, **kw):
        self.endog = None
        self.exog = None
        self.data = None
        self.names = None
        self.endog_name = None
        self.exog_name = None

        dict.__init__(self, kw)
        self.__dict__ = self


def load_csv(base_file, csv_name, sep=","):
    """Load standard csv file."""
    filepath = dirname(abspath(base_file))
    filename = join(filepath, csv_name)
    data = read_csv(filename, sep=sep)
    return data


def load_dataset(
    data: DataFrame | Series,
    endog_name: str,
    exog_name: List[str] | None = None,
):
    """Load dataset."""
    names = list(data.columns)
    endog = data[endog_name]
    exog = data[exog_name] if exog_name is not None else None

    dataset = Dataset(
        data=data,
        names=names,
        endog=endog,
        exog=exog,
        endog_name=endog_name,
        exog_name=exog_name,
    )
    return dataset
