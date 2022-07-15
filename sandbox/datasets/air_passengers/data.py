import sandbox.datasets.utils as du

COPYRIGHT = """This data is public domain."""

TITLE = """AirPassengers: Monthly Airline Passenger Numbers 1949-1960"""

DESCRIPTION = """
Monthly Airline Passenger Numbers 1949-1960.
"""

SOURCE = """
Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976)
Time Series Analysis, Forecasting and Control. Third Edition. Holden-Day. Series G.
"""

NOTE = """
Number of Observations - 144 (Monthly 1949 - 1960)
Number of Variables - 1
    #Passengers - the number of airline passengers
"""


def load():
    dataset = _process_data()
    return dataset


def _get_data():
    return du.load_csv(__file__, "air_passengers.csv")


def _process_data():
    data = _get_data()

    import pandas as pd

    data.set_index(
        pd.to_datetime(data["Month"], format="%Y-%m"), drop=True, inplace=True
    )
    data.index.freq = "MS"
    data.drop(columns=["Month"], inplace=True)
    data = data.astype(float)

    names = list(data.columns)
    endog_name = "#Passengers"
    endog = data[endog_name]
    dataset = du.Dataset(
        data=data,
        names=names,
        endog=endog,
        endog_name=endog_name,
        exog=None,
        exog_name=None,
    )

    return dataset
