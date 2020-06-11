""" training script """

import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
from matplotlib import pyplot

import data_helpers as dh

class Stocker:
    def __init__(self, training, test, loss='mse', optimizer='adam'):
        """ Creating Stocker instance immediately creates model 

            Model (WIP) is a two-layer LSTM. Defaults to Mean Squared Error
            loss function and ADAM optimizer function.
        """
        self.training_data = training
        self.test_data = test

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(100, activation='tanh', recurrent_activation='sigmoid', \
                                            input_shape=(training.shape[1], training.shape[2])))
        self.model.add(tf.keras.layers.Dense(5))
        self.model.compile(loss=loss, optimizer=optimizer)
        print(self.model.summary())

    """ Training WIP

    def train(self):
        self.fit = self.model.fit(self.training_data, epochs=50, \
                            batch_size=100, \
                            validation_data=self.test_data, verbose=2, shuffle=False)

        pyplot.plot(self.fit['loss'], label='train')
        pyplot.plot(self.fit['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()
        
    """


if __name__ == '__main__':

    """ Test/Demo of Stocker module """

    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument('key', help='User API Key')
    parser.add_argument('-outdir', metavar='out', default='/models/', help="Directory for stored model(s) (one for each symbol).")
    parser.add_argument('symbols', nargs=argparse.REMAINDER, help="List of symbols to train (Place all at end of command)")
    parse = parser.parse_args()

    data = {}

    for symbol in parse.symbols:

        # read historical daily data from alpha_vantage
        # store in python dict
        hist = dh.daily(symbol, parse.key, compact=False)
        data[symbol] = hist
        print(hist)
        #print()

        """ Data Preprocessing """ 

        # turn dataframe to numpy array
        tmp = hist.to_numpy()

        # split into training and testing sets 90-10
        split = round(tmp.shape[0]*1/10)
        test, training = tmp[:split], tmp[split:]

        test = dh.array_to_supervised(test, 3)
        test = test.reshape((test.shape[0], 1, test.shape[1]))

        training = dh.array_to_supervised(training, 3)
        training = training.reshape((training.shape[0], 1, training.shape[1]))

        # convert numpy arrays to tensors and reshape for LSTM
        training_tensor = tf.convert_to_tensor(training, np.float32)
        test_tensor = tf.convert_to_tensor(test, np.float32)
        
        """ -------------------------------- """
        #print(training_tensor)
        #print(test_tensor)

        model = Stocker(training_tensor, test_tensor)
        model.train()