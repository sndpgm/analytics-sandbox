import sandbox.datasets.utils as du

COPYRIGHT = """This data is public domain."""

TITLE = """Road traffic fatalities in Norway and Finland"""

DESCRIPTION = """
The annual numbers of road traffic fatalities in Norway and Finland as observed for the years 1970 through 2003
"""

SOURCE = """
Commandeur, J. J., & Koopman, S. J. (2007). An introduction to state space time series analysis.
Oxford University Press.
"""

NOTE = """
Number of Observations - 34
Number of Variables - 2
    Norway - the annual numbers of road traffic fatalities in Norway
    Finland - the annual numbers of road traffic fatalities in Finland
"""


def load():
    dataset = _process_data()
    return dataset


def _get_data():
    return du.load_csv(__file__, "rtf.csv")


def _process_data():
    data = _get_data()

    import pandas as pd

    data.set_index(
        pd.to_datetime(data["date"], format="%Y") + pd.offsets.YearEnd(),
        drop=True,
        inplace=True,
    )
    data.index.freq = "Y"
    data.drop(columns=["date"], inplace=True)

    data = data.astype(float)
    names = list(data.columns)
    endog_name = "Finland"
    endog = data[endog_name]
    exog_name = ["Norway"]
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
