import pandas as pd

import sandbox.datasets.utils as utl

test_base_file = "sandbox/datasets/air_passengers/air_passengers.csv"
test_csv_name = "air_passengers.csv"

# test data
data = pd.DataFrame(
    {
        "endog": [1] * 10,
        "exog1": [10] * 10,
        "exog2": [float(1.5)] * 10,
        "exog3": [str("str")] * 10,
    }
)
endog_name = "endog"
exog_name = list(["exog1", "exog2", "exog3"])
endog = data[endog_name]
exog = data[exog_name]
names = list(data.columns)


class TestUtils:
    def test_dataset(self):
        """Test for Dataset class"""
        dataset = utl.Dataset(
            data=data,
            names=names,
            endog=endog,
            exog=exog,
            endog_name=endog_name,
            exog_name=exog_name,
        )

        # test
        pd.testing.assert_frame_equal(data, dataset.data)
        assert names == dataset.names

        pd.testing.assert_series_equal(endog, dataset.endog)
        assert endog_name == dataset.endog_name

        pd.testing.assert_frame_equal(exog, dataset.exog)
        assert exog_name == dataset.exog_name

    def test_load_csv(self):
        """"""
        data = utl.load_csv(test_base_file, test_csv_name)
        assert isinstance(data, pd.DataFrame)

    def test_load_dataset(self):
        """"""
        dataset = utl.load_dataset(
            data=data, endog_name=endog_name, exog_name=exog_name
        )

        # test
        pd.testing.assert_frame_equal(data, dataset.data)
        assert names == dataset.names

        pd.testing.assert_series_equal(endog, dataset.endog)
        assert endog_name == dataset.endog_name

        pd.testing.assert_frame_equal(exog, dataset.exog)
        assert exog_name == dataset.exog_name
