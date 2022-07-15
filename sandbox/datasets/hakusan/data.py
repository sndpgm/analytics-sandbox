import sandbox.datasets.utils as du

COPYRIGHT = """This data is public domain."""

TITLE = """HAKUSAN: Ship's Navigation Data"""

DESCRIPTION = """
A multivariate time series of a ship's yaw rate, rolling, pitching and rudder angles
which were recorded every second while navigating across the Pacific Ocean.
"""

SOURCE = """
http://www.mi.u-tokyo.ac.jp/mds-oudan/lecture_document_2019_math7/時系列データ/hakusan_new.csv
"""

NOTE = """
Number of Observations - 1000
Number of Variables - 4
    YawRate - yaw rate
    Rolling - rolling
    Pitching - pitching
    Rudder - rudder angle
"""


def load():
    dataset = _process_data()
    return dataset


def _get_data():
    return du.load_csv(__file__, "hakusan.csv")


def _process_data():
    data = _get_data()
    data = data.astype(float)
    names = list(data.columns)
    endog_name = "YawRate"
    endog = data[endog_name]
    exog_name = ["Rolling", "Pitching", "Rudder"]
    exog = data[exog_name]
    dataset = du.Dataset(
        data=data,
        names=names,
        endog=endog,
        endog_name=endog_name,
        exog=exog,
        exog_name=exog_name,
    )
    return dataset
