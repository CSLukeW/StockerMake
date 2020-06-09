import pandas as pd
from alpha_vantage.timeseries import TimeSeries

""" Training functions inc."""

def daily(symbol, key, compact=True):
    """ Returns data frame of queried data

        symbol -- symbol of desired stock
        key -- user's API key
        compact -- True -> last 100 results
                   False -> all past results
    """

    ts = TimeSeries(key=key, output_format='pandas')
    if compact:
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
    else:
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')

    return data










if __name__ == "__main__":
    key = '8X10BAT7HUYHPS8Y'
    symbols = ['AAPL', 'TSLA', 'MSFT']
    data = []

    for symbol in symbols:
        hist = daily(symbol, key, compact=False)
        data.append(hist)
        print(hist)
        print()