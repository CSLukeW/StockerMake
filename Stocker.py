""" training script """

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
from matplotlib import pyplot

import data_helpers as dh

class Stocker:
    def __init__(self, training, training_shape, test, loss='mse', optimizer=tf.keras.optimizers.Adam()):
        """ Creating Stocker instance immediately creates model 

            Model (WIP) is a two-layer LSTM. Defaults to Mean Squared Error
            loss function and ADAM optimizer function.
        """
        self.training_data = training
        self.test_data = test

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(100, activation='tanh', recurrent_activation='sigmoid', \
                                            input_shape=training_shape[-2:]))
        self.model.add(tf.keras.layers.Dense(5))
        self.model.compile(loss=loss, optimizer=optimizer)
        print(self.model.summary())

    def train(self):
        self.fit = self.model.fit(self.training_data, epochs=50, \
                            batch_size=100, steps_per_epoch = 50, \
                            validation_data=self.test_data, validation_steps = 50)

        pyplot.plot(self.fit.history['loss'], label='train')
        pyplot.plot(self.fit.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()


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
        hist.head()
        data[symbol] = hist
        print(hist)
        #print()

        hist.plot(subplots=True)
        pyplot.savefig('input.png')

        """ Data Preprocessing """
        
        split = round(len(hist.index)*7/10)

        # standardize data
        data = hist.values
        mean = data[:split].mean(axis=0)
        std = data[:split].mean(axis=0)

        data = (data-mean)/std

        # split into training and test datasets
        past = 7
        future = 1
        step = 1
        buffer = 100
        batch = 100

        train_in, train_out = dh.single_step_data(data, data[:, 1], 0, split, past, future, step)
        val_in, val_out = dh.single_step_data(data, data[:, 1], split, None, past, future, step)
        train_shape = train_in.shape
        print(train_shape)

        # convert to tf Datasets
        train_data_set = tf.data.Dataset.from_tensor_slices((train_in, train_out))
        train_data_set = train_data_set.cache().shuffle(buffer).batch(batch).repeat()

        val_data_set = tf.data.Dataset.from_tensor_slices((val_in, val_out))
        val_data_set = val_data_set.batch(batch).repeat()
        
        """ -------------------------------- """

        model = Stocker(train_data_set, train_shape, val_data_set)
        model.train()