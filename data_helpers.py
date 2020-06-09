from alpha_vantage.timeseries import TimeSeries

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