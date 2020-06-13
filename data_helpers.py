from alpha_vantage.timeseries import TimeSeries
import numpy as np
from scipy.ndimage.interpolation import shift
import tensorflow as tf

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

def standardize(data, split):
    """ Standardize data using z-value """
    values = data.values
    mean = values[:split].mean(axis=0)
    std = values[:split].mean(axis=0)

    return (values-mean)/std

def to_dataset(data, split):
    """Returns 2-tuple of training dataset and validation dataset for input into LSTM"""
    past = 30
    future = 7
    step = 1
    buffer = 60
    batch = 60

    train_in, train_out = single_step_data(data, data[:, 1], 0, split, past, future, step)
    val_in, val_out = single_step_data(data, data[:, 1], split, None, past, future, step)

    return tf.data.Dataset.from_tensor_slices((train_in, train_out)).cache().shuffle(buffer).batch(batch).repeat(), \
            tf.data.Dataset.from_tensor_slices((val_in, val_out)).batch(batch).repeat(), train_in.shape

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

    return np.array(dataset), np.array(labels)