import sandbox.datasets.utils as du

COPYRIGHT = """This data is public domain."""

TITLE = """Seismic Data"""

DESCRIPTION = """
The time series of East-West components of seismic waves, recorded every 0.02 seconds.
"""

SOURCE = """
http://www.mi.u-tokyo.ac.jp/mds-oudan/lecture_document_2019_math7/時系列データ/mye1f_new.csv

Takanami, T. (1991), "ISM data 43-3-01: Seismograms of foreshocks of 1982 Urakawa-Oki earthquake",
Ann. Inst. Statist. Math., 43, 605.
"""

NOTE = """
Number of Observations - 2600
Number of Variables - 1
    mye1f - spectrum strength on seismic waves
"""


def load():
    dataset = _process_data()
    return dataset


def _get_data():
    return du.load_csv(__file__, "mye1f.csv")


def _process_data():
    data = _get_data()
    data = data.astype(float)
    names = list(data.columns)
    endog_name = "mye1f"
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
