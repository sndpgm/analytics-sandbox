import pandas as pd
import pytest
from dask import dataframe as dd

from sandbox.features import ts_calculator as tc

x_pd = pd.DataFrame(
    {
        "x": pd.to_datetime(
            ["2020-01-01 12:00:00", "2020-06-15 23:00:00", "2021-02-12 01:00:00"]
        )
    }
)
x_dd = dd.from_pandas(x_pd, npartitions=1)


class TestTsCalculator:
    def test_hour(self):
        exp_name = "x__hour"
        exp_ret = pd.Series([12, 23, 1], name=exp_name)

        pd.testing.assert_series_equal(exp_ret, tc.hour(x_pd, value="x")[0])
        assert exp_name == tc.hour(x_pd, value="x")[1]

        pd.testing.assert_series_equal(exp_ret, tc.hour(x_dd, value="x")[0].compute())
        assert exp_name == tc.hour(x_dd, value="x")[1]

    def test_dayofweek(self):
        exp_name = "x__dayofweek"
        exp_ret = pd.Series([2, 0, 4], name=exp_name)

        pd.testing.assert_series_equal(exp_ret, tc.dayofweek(x_pd, value="x")[0])
        assert exp_name == tc.dayofweek(x_pd, value="x")[1]

        pd.testing.assert_series_equal(
            exp_ret, tc.dayofweek(x_dd, value="x")[0].compute()
        )
        assert exp_name == tc.dayofweek(x_dd, value="x")[1]

    def test_quarter(self):
        exp_name = "x__quarter"
        exp_ret = pd.Series([1, 2, 1], name=exp_name)

        pd.testing.assert_series_equal(exp_ret, tc.quarter(x_pd, value="x")[0])
        assert exp_name == tc.quarter(x_pd, value="x")[1]

        pd.testing.assert_series_equal(
            exp_ret, tc.quarter(x_dd, value="x")[0].compute()
        )
        assert exp_name == tc.quarter(x_dd, value="x")[1]

    def test_month(self):
        exp_name = "x__month"
        exp_ret = pd.Series([1, 6, 2], name=exp_name)

        pd.testing.assert_series_equal(exp_ret, tc.month(x_pd, value="x")[0])
        assert exp_name == tc.month(x_pd, value="x")[1]

        pd.testing.assert_series_equal(exp_ret, tc.month(x_dd, value="x")[0].compute())
        assert exp_name == tc.month(x_dd, value="x")[1]

    def test_year(self):
        exp_name = "x__year"
        exp_ret = pd.Series([2020, 2020, 2021], name=exp_name)

        pd.testing.assert_series_equal(exp_ret, tc.year(x_pd, value="x")[0])
        assert exp_name == tc.year(x_pd, value="x")[1]

        pd.testing.assert_series_equal(exp_ret, tc.year(x_dd, value="x")[0].compute())
        assert exp_name == tc.year(x_dd, value="x")[1]

    def test_dayofyear(self):
        exp_name = "x__dayofyear"
        exp_ret = pd.Series([1, 167, 43], name=exp_name)

        pd.testing.assert_series_equal(exp_ret, tc.dayofyear(x_pd, value="x")[0])
        assert exp_name == tc.dayofyear(x_pd, value="x")[1]

        pd.testing.assert_series_equal(
            exp_ret, tc.dayofyear(x_dd, value="x")[0].compute()
        )
        assert exp_name == tc.dayofyear(x_dd, value="x")[1]

    def test_dayofmonth(self):
        exp_name = "x__dayofmonth"
        exp_ret = pd.Series([1, 15, 12], name=exp_name)

        pd.testing.assert_series_equal(exp_ret, tc.dayofmonth(x_pd, value="x")[0])
        assert exp_name == tc.dayofmonth(x_pd, value="x")[1]

        pd.testing.assert_series_equal(
            exp_ret, tc.dayofmonth(x_dd, value="x")[0].compute()
        )
        assert exp_name == tc.dayofmonth(x_dd, value="x")[1]

    def test_weekofyear(self):
        exp_name = "x__weekofyear"
        exp_ret = pd.Series([1, 25, 6], name=exp_name)

        pd.testing.assert_series_equal(exp_ret, tc.weekofyear(x_pd, value="x")[0])
        assert exp_name == tc.weekofyear(x_pd, value="x")[1]

        pd.testing.assert_series_equal(
            exp_ret, tc.weekofyear(x_dd, value="x")[0].compute()
        )
        assert exp_name == tc.weekofyear(x_dd, value="x")[1]

    def test_lag(self):
        # pandas.Series, dask.dataframe.Series
        params = {"lag": 1}
        exp_name = "x__lag_lag_1"
        exp_series = pd.Series(
            [
                None,
                pd.Timestamp("2020-01-01 12:00:00"),
                pd.Timestamp("2020-06-15 23:00:00"),
            ],
            name=exp_name,
        )

        pd.testing.assert_series_equal(
            exp_series, tc.lag(x_pd, value="x", params=params)[0]
        )
        assert exp_name == tc.lag(x_pd, value="x", params=params)[1]

        pd.testing.assert_series_equal(
            exp_series, tc.lag(x_dd, value="x", params=params)[0].compute()
        )
        assert exp_name == tc.lag(x_dd, value="x", params=params)[1]

        #
        exp_gb = pd.Series(
            [
                None,
                pd.Timestamp("2022-07-01"),
                None,
                pd.Timestamp("2022-07-03"),
                None,
                pd.Timestamp("2022-07-05"),
                None,
                pd.Timestamp("2022-07-07"),
            ],
            name=exp_name,
        )
        x_gb_pd = pd.DataFrame(
            {
                "cat_id": [1, 1, 1, 1, 2, 2, 2, 2],
                "subcat_id": [1, 1, 2, 2, 3, 3, 4, 4],
                "x": pd.date_range(start="2022-07-01", end="2022-07-08"),
            }
        )
        pd.testing.assert_series_equal(
            exp_gb,
            tc.lag(x_gb_pd, value="x", params=params, by=["cat_id", "subcat_id"])[0],
        )
        assert (
            exp_name
            == tc.lag(x_gb_pd, value="x", params=params, by=["cat_id", "subcat_id"])[1]
        )

        x_gb_dd = dd.from_pandas(x_gb_pd, npartitions=2)
        pd.testing.assert_series_equal(
            exp_gb,
            tc.lag(x_gb_dd, value="x", params=params, by=["cat_id", "subcat_id"])[0],
        )
        assert (
            exp_name
            == tc.lag(x_gb_dd, value="x", params=params, by=["cat_id", "subcat_id"])[1]
        )

    def test_rolling(self):
        # Test data and expected results.
        # For test of pandas data sorting, x_pd is shuffled.
        test_data = pd.read_csv("tests/features/data/testdata4features.csv")
        x_pd = (
            test_data[["category_id", "subcategory_id", "date", "sales"]]
            .sample(frac=1)
            .copy()
        )

        # Expected.
        exp_name = [
            "sales__rolling_max_lag_1_window_2",
            "sales__rolling_mean_lag_1_window_2",
            "sales__rolling_median_lag_1_window_2",
            "sales__rolling_min_lag_1_window_2",
            "sales__rolling_sum_lag_1_window_2",
            "sales__rolling_std_lag_1_window_2",
            "sales__rolling_var_lag_1_window_2",
        ]
        exp_ret = test_data[
            [
                "sales__max_lag_1_window_2",
                "sales__mean_lag_1_window_2",
                "sales__median_lag_1_window_2",
                "sales__min_lag_1_window_2",
                "sales__sum_lag_1_window_2",
                "sales__std_lag_1_window_2",
                "sales__var_lag_1_window_2",
            ]
        ].copy()
        exp_ret.columns = exp_name
        exp_ret_gb = test_data[
            [
                "sales__max_lag_1_window_2_by_partition",
                "sales__mean_lag_1_window_2_by_partition",
                "sales__median_lag_1_window_2_by_partition",
                "sales__min_lag_1_window_2_by_partition",
                "sales__sum_lag_1_window_2_by_partition",
                "sales__std_lag_1_window_2_by_partition",
                "sales__var_lag_1_window_2_by_partition",
            ]
        ].copy()
        exp_ret_gb.columns = exp_name

        # Parameters.
        params = {
            "lag": 1,
            "window": 2,
            "stats": ["max", "mean", "median", "min", "sum", "std", "var"],
        }
        value = "sales"
        by = ["category_id", "subcategory_id"]
        sort = ["date"]
        x_pd["date"] = pd.to_datetime(x_pd["date"])

        # Test for pandas.
        pd.testing.assert_frame_equal(
            exp_ret, tc.rolling(x_pd, value=value, params=params, sort=sort)[0]
        )
        assert exp_name == tc.rolling(x_pd, value=value, params=params, sort=sort)[1]

        # Test for pandas which is partitioned by.
        pd.testing.assert_frame_equal(
            exp_ret_gb,
            tc.rolling(x_pd, value=value, params=params, by=by, sort=sort)[0],
        )
        assert (
            exp_name
            == tc.rolling(x_pd, value=value, params=params, by=by, sort=sort)[1]
        )

        # Test for dask.
        x_dd = dd.from_pandas(x_pd.sort_index(), npartitions=2)
        pd.testing.assert_frame_equal(
            exp_ret, tc.rolling(x_dd, value=value, params=params)[0].compute()
        )
        assert exp_name == tc.rolling(x_dd, value=value, params=params)[1]

        # Test for dask which is partitioned by (dask.groupby.SeriesGroupBy).
        pd.testing.assert_frame_equal(
            exp_ret_gb, tc.rolling(x_dd, value=value, params=params, by=by)[0]
        )
        assert exp_name == tc.rolling(x_dd, value=value, params=params, by=by)[1]

    def test__sort_and_groupby(self):
        # Test data and expected results.
        # For test of pandas data sorting, x_pd is shuffled.
        test_data = pd.read_csv("tests/features/data/testdata4features.csv")
        x_pd = (
            test_data[["category_id", "subcategory_id", "date", "sales"]]
            .sample(frac=1)
            .copy()
        )
        # x_pd = exp.sample(frac=1).copy()

        # Parameters.
        value = "sales"
        by = ["category_id", "subcategory_id"]
        sort = ["date"]

        # pandas
        pd.testing.assert_series_equal(x_pd[value], tc._sort_and_groupby(x_pd, value))
        pd.testing.assert_series_equal(
            x_pd.sort_values(sort)[value],
            tc._sort_and_groupby(x_pd, value=value, sort=sort),
        )

        assert isinstance(
            tc._sort_and_groupby(x_pd, value=value, by=by),
            pd.core.groupby.generic.SeriesGroupBy,
        )
        pd.testing.assert_series_equal(
            x_pd[value], tc._sort_and_groupby(x_pd, value=value, by=by).obj
        )

        assert isinstance(
            tc._sort_and_groupby(x_pd, value=value, by=by, sort=sort),
            pd.core.groupby.generic.SeriesGroupBy,
        )
        pd.testing.assert_series_equal(
            x_pd.sort_values(sort)[value],
            tc._sort_and_groupby(x_pd, value=value, by=by, sort=sort).obj,
        )

        # dask
        x_dd = dd.from_pandas(x_pd.sort_index(), npartitions=2)
        pd.testing.assert_series_equal(
            x_dd[value].compute(), tc._sort_and_groupby(x_dd, value).compute()
        )

        assert isinstance(
            tc._sort_and_groupby(x_dd, value=value, by=by), dd.groupby.SeriesGroupBy
        )

    def test__raise_sort_and_groupby(self):
        x_dd = dd.read_csv("tests/features/data/testdata4features.csv")

        # Parameters.
        value = "sales"
        sort = ["date"]

        msg = "In dask format, sorting the order of rows is unsupported, then please sort your dask data in advance."
        with pytest.raises(NotImplementedError, match=msg):
            tc._sort_and_groupby(x_dd, value=value, sort=sort)
