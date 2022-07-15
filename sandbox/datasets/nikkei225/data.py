import sandbox.datasets.utils as du

COPYRIGHT = """This data is public domain."""

TITLE = """Nikkei 225 stock price on market close"""

DESCRIPTION = """
The stock price on market close in business day from 1988/1/4 to 1993/12/30.
"""

SOURCE = """
http://www.mi.u-tokyo.ac.jp/mds-oudan/lecture_document_2019_math7/時系列データ/nikkei225_new.csv
"""

NOTE = """
Number of Observations - 1480
Number of Variables - 1
    nikkei225 - stock price on market close
"""


def load():
    dataset = _process_data()
    return dataset


def _get_data():
    return du.load_csv(__file__, "nikkei225.csv")


def _process_data():
    data = _get_data()
    data = data.astype(float)
    names = list(data.columns)
    endog_name = "nikkei225"
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
