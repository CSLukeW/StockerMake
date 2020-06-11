from alpha_vantage.timeseries import TimeSeries
import numpy as np
from scipy.ndimage.interpolation import shift

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

def array_to_supervised(array, input_index):
    """ 
        Converts given numpy array to supervised learning input.

        Shifts given to t-1 and moves it to end of the array. 
        End column represents the recursive input value.
    """

    temp = []

    for row in array:
        temp.append(row[input_index])

    array = np.delete(array, input_index, axis=1)

    temp = np.roll(temp, 1)
    temp[-1] = temp[-2]
    temp = np.asarray([temp])

    con = np.concatenate((array, temp.T), axis=1)

    return con

if __name__ == '__main__':
    array = [[1.0,2.0,3.0],\
            [4.0, 5.0, 6.0]]

    print(array_to_supervised(array, 0))