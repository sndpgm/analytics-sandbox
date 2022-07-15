import sandbox.datasets.utils as du

COPYRIGHT = """This data is public domain."""

TITLE = """Wholesale hardware data"""

DESCRIPTION = """
The monthly record of wholesale hardware data. (January 1967 - November 1979).
"""

SOURCE = """
http://www.mi.u-tokyo.ac.jp/mds-oudan/lecture_document_2019_math7/時系列データ/whard_new.csv
"""

NOTE = """
Number of Observations - 155
Number of Variables - 1
    whard - wholesale hardware
"""


def load():
    dataset = _process_data()
    return dataset


def _get_data():
    return du.load_csv(__file__, "whard.csv")


def _process_data():
    data = _get_data()

    import pandas as pd

    data.set_index(
        pd.to_datetime(data["date"], format="%Y-%m-%d"), drop=True, inplace=True
    )
    data.index.freq = "MS"
    data.drop(columns=["date"], inplace=True)
    data = data.astype(float)

    names = list(data.columns)
    endog_name = "whard"
    endog = data[endog_name]
    exog_name = None
    exog = None
    dataset = du.Dataset(
        data=data,
        names=names,
        endog=endog,
        endog_name=endog_name,
        exog=exog,
        exog_name=exog_name,
    )
    return dataset
