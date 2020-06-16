from alpha_vantage.timeseries import TimeSeries
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler

def daily_adjusted(symbol, key, compact=True):
    """ Returns data frame of queried data

        symbol -- symbol of desired stock
        key -- user's API key
        compact -- True -> last 100 results
                   False -> all past results
    """

    ts = TimeSeries(key=key, output_format='pandas')
    if compact:
        data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='compact')
    else:
        data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='full')

    return data

def single_step_data(data, target, start, end, history_size, target_size, step):

    """
    Converts dataframe to numpy arrays with correct LSTM input format.
    """

    dataset=[]
    labels=[]

    start = start + history_size
    if end is None:
        end = len(data) - target_size
    
    for i in range(start, end):
        indices = range(i-history_size, i, step)
        dataset.append(data[indices])

        labels.append(target[i+target_size])

    print(np.array(dataset).shape)

    return np.array(dataset), np.array(labels)

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir