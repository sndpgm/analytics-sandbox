from __future__ import annotations

from typing import Union

import pandas as pd
import yaml

from sandbox.features import ts_calculator

try:
    from dask import dataframe as dd
except ImportError:
    dd = None

if dd:
    DataFrame = Union[pd.DataFrame, dd.DataFrame]
else:
    DataFrame = pd.DataFrame


def get_features(
    data: DataFrame,
    column_values: list[str] | None = None,
    column_id: list[str] | None = None,
    sort_values: list[str] | None = None,
    func_params_list: list[dict] | None = None,
    params_path: str | None = None,
) -> DataFrame:
    """Get the dataframe with features columns.

    Parameters
    ----------
    data : {pandas.DataFrame, dask.DataFrame}
    column_values : {list [str], None}
    column_id : {list[str], None}
    sort_values : {list[str], None}
    func_params_list : {list[dict], None}
    params_path: {str, None}

    Returns
    -------
    pandas.DataFrame or dask.DataFrame
        A dataframe with features columns whose format is in accordance with input `data`.
    """
    # パラメータを受けてFeaturesFactoryに渡す形式に変換.
    fet_mgr = FeaturesManager(
        column_values=column_values,
        column_id=column_id,
        sort_values=sort_values,
        params_path=params_path,
        func_params_list=func_params_list,
    )

    # FeaturesFactoryの構築
    fet_factory = FeaturesFactory(fet_mgr.features_parameters)

    # FeaturesFactoryで特徴量生成.
    return fet_factory.process(data)


class FeaturesManager:
    """"""

    def __init__(
        self,
        column_values: list[str] | None = None,
        column_id: list[str] | None = None,
        sort_values: list[str] | None = None,
        func_params_list: list[dict] | None = None,
        params_path: str | None = None,
    ):
        self.column_values = self._check_column_values(column_values)
        self.column_id = self._check_column_id(column_id)
        self.sort_values = self._check_sort_values(sort_values)
        self.func_params_list = self._check_func_param_list(func_params_list)
        self.params_path = params_path

    @staticmethod
    def _check_column_values(
        column_values: list[str] | None = None,
    ) -> list[str] | None:
        if column_values:
            if not isinstance(column_values, list):
                msg = "The value of `column_values` must be list."
                raise ValueError(msg)
        return column_values

    @staticmethod
    def _check_column_id(column_id: list[str] | None = None) -> list[str] | None:
        if column_id:
            if not isinstance(column_id, list):
                msg = "The value of `column_id` must be list."
                raise ValueError(msg)
        return column_id

    @staticmethod
    def _check_sort_values(sort_values: list[str] | None = None) -> list[str] | None:
        if sort_values:
            if not isinstance(sort_values, list):
                msg = "The value of `sort_values` must be list."
                raise ValueError(msg)
        return sort_values

    @staticmethod
    def _check_func_param_list(
        func_params_list: list[dict] | None = None,
    ) -> list[dict] | None:
        if func_params_list:
            for d in func_params_list:
                if list(d.keys()) != ["func_name", "func_params"]:
                    msg = (
                        "All elements of the list of `func_domain` must be the dict whose "
                        "keys are ['func_name', 'func_params']."
                    )
                    raise ValueError(msg)

                if not isinstance(d["func_name"], str):
                    msg = "The value of `func_name` must be str."
                    raise ValueError(msg)

                if not (d["func_params"] is None or isinstance(d["func_params"], list)):
                    msg = "The value of `func_params` must be None or list."
                    raise ValueError(msg)
        return func_params_list

    @property
    def features_parameters(self) -> dict[list | dict]:
        """

        Returns
        -------

        """
        func_domain = self.func_params_list
        data_domain = dict()
        data_domain["column_values"] = self.column_values
        data_domain["column_id"] = self.column_id
        data_domain["sort_values"] = self.sort_values

        if self.params_path:
            with open(self.params_path, mode="r") as file:
                contents = yaml.safe_load(file)

            if "func_domain" in contents.keys():
                func_domain = self._check_func_param_list(contents["func_domain"])

            if "data_domain" in contents.keys():
                d = contents["data_domain"]
                if not set(d.keys()).issuperset(
                    {"column_id", "column_values", "sort_values"}
                ):
                    msg = (
                        "All elements of the list of `data_domain` must be the dict whose "
                        "keys are ['column_id', 'column_values', 'sort_values']."
                    )
                    raise ValueError(msg)

                data_domain["column_values"] = self._check_column_values(
                    d["column_values"]
                )
                data_domain["column_id"] = self._check_column_id(d["column_id"])
                data_domain["sort_values"] = self._check_sort_values(d["sort_values"])

        if len(func_domain) == 0:
            msg = (
                "Any argument `func_params_list` or func_domain in "
                "external setting yaml file is not defined."
            )
            raise ValueError(msg)

        if data_domain["column_values"] is None:
            msg = (
                "Arguments `column_values` or data_domain in external setting yaml file "
                "is not defined."
            )
            raise ValueError(msg)

        return {"func_domain": func_domain, "data_domain": data_domain}


class FeaturesFactory:
    """"""

    def __init__(self, features_parameters: dict):
        self.features_parameters = features_parameters
        self.input_count = None

    def validate_input(self, data) -> None:
        # input が dataframe かどうかの検証.
        if not self._input_is_dataframe(data):
            msg = "Specified data must be DataFrame or Series in Pandas/Dask, not {}".format(
                str(type(data))
            )
            raise ValueError(msg)

        # input が指定した column_values を持つかどうかの検証.
        if not self._input_includes_column_values(data):
            msg = "Specified data must have the following columns: {}".format(
                self.features_parameters["data_domain"]["column_values"]
            )
            raise ValueError(msg)

        # input が指定した column_id を持つかどうかの検証.
        if not self._input_include_column_id(data):
            msg = "Specified data must have the following columns: {}".format(
                self.features_parameters["data_domain"]["column_id"]
            )
            raise ValueError(msg)

        # input が指定した sort_values を持つかどうかの検証.
        if not self._input_include_sort_values(data):
            msg = "Specified data must have the following columns: {}".format(
                self.features_parameters["data_domain"]["sort_values"]
            )
            raise ValueError(msg)

        # 検証をクリアした場合にはデータの件数を保存する.
        self.input_count = len(data)

    @staticmethod
    def _input_is_dataframe(data):
        if isinstance(data, pd.DataFrame):
            return True
        elif dd and isinstance(data, dd.DataFrame):
            return True
        else:
            return False

    def _input_includes_column_values(self, data):
        columns = data.columns.to_list()
        if set(columns).issuperset(
            set(self.features_parameters["data_domain"]["column_values"])
        ):
            return True
        else:
            return False

    def _input_include_column_id(self, data):
        _is = True
        columns = data.columns.to_list()
        if self.features_parameters["data_domain"]["column_id"]:
            if set(columns).issuperset(
                set(self.features_parameters["data_domain"]["column_id"])
            ):
                _is = True
            else:
                _is = False
        return _is

    def _input_include_sort_values(self, data):
        _is = True
        columns = data.columns.to_list()
        if self.features_parameters["data_domain"]["sort_values"]:
            if set(columns).issuperset(
                set(self.features_parameters["data_domain"]["sort_values"])
            ):
                _is = True
            else:
                _is = False
        return _is

    def create_features(self, data):
        # データの整形 (sorting)
        # 指定された column_id / sort_values で sorting する.
        column_id = self.features_parameters["data_domain"]["column_id"]
        column_id = [] if column_id is None else column_id

        sort_values = self.features_parameters["data_domain"]["sort_values"]
        sort_values = [] if sort_values is None else sort_values

        sort_index = column_id + sort_values
        if len(sort_index) != 0:
            if isinstance(data, pd.DataFrame):
                data = data.sort_values(sort_index).reset_index(drop=False)
            else:
                npartitions = data.npartitions
                data = data.compute().sort_values(sort_index).reset_index(drop=False)
                data = dd.from_pandas(data, npartitions=npartitions)

        # 実際の特徴量生成
        column_values = self.features_parameters["data_domain"]["column_values"]
        func_domain = self.features_parameters["func_domain"]

        # 指定された column_values に対して特徴量生成を実施する.
        for value in column_values:
            # 設定されている特徴量関数を一つずつ実施する.
            for d in func_domain:
                func_name = d["func_name"]
                func_params = d["func_params"]

                if hasattr(ts_calculator, func_name):
                    func = getattr(ts_calculator, func_name)

                    # パラメータが必要な関数の場合は設定でそのパラメータの組み合わせを input しているので
                    # そのパラメータ群で for 文を実施.
                    # ToDo: getattr(func, "column_id", False) の分岐では func に入れるパラメータ dict を更新し
                    #  func 処理は if 文以降に一つにまとめられるのでは?
                    if func_params:
                        for p in func_params:
                            if (
                                getattr(func, "column_id", False)
                                and len(column_id) != 0
                            ):
                                ret, names = func(
                                    data, value=value, params=p, by=column_id
                                )
                            else:
                                ret, names = func(data, value=value, params=p)

                            # dask.DataFrame の場合には data[list] = DataFrame の形式で複数列を追加できない.
                            # よって if 文による分岐を実施,
                            if (
                                dd
                                and isinstance(data, dd.DataFrame)
                                and isinstance(names, list)
                            ):
                                for name in names:
                                    data[name] = ret[name]
                            else:
                                data[names] = ret

                    else:
                        if getattr(func, "column_id", False) and len(column_id) != 0:
                            ret, names = func(data, value=value, by=column_id)
                        else:
                            ret, names = func(data, value=value)

                        # ToDo: 一方の if 条件で同じ処理があるが, まとめることができないか...?
                        if (
                            dd
                            and isinstance(data, dd.DataFrame)
                            and isinstance(names, list)
                        ):
                            for name in names:
                                data[name] = ret[name]
                        else:
                            data[names] = ret

                else:
                    continue
        return data

    def inspect_output(self, data):
        if self.input_count != len(data):
            msg = "Processed data length through `create_features` is different from the one of original."
            raise Exception(msg)

    def process(self, data):
        self.validate_input(data)
        data = self.create_features(data)
        self.inspect_output(data)
        return data
