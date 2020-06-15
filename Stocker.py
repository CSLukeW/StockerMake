""" training script """

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
from matplotlib import pyplot
import os

import data_helpers as dh

class Stocker:
    def __init__(self, symbol, data, split, feature_labels, row_labels, \
                    loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=.0001)):
        """ Creating Stocker instance immediately creates model 

            Model (WIP) is a two-layer LSTM. Defaults to Mean Squared Error
            loss function and ADAM optimizer function.
        """

        past = 60
        future = 1
        step = 1
        buffer = 50


        self.all_data_df = data
        data_numpy = data.to_numpy()

        batch = 50

        # store data in numpy format
        self.train_in, self.train_out = dh.single_step_data(data_numpy, data_numpy[:, 4], 0, split, past, future, step)
        self.val_in, self.val_out = dh.single_step_data(data_numpy, data_numpy[:, 4], split, None, past, future, step)

        # store data attributes
        self.symbol = symbol
        self.train_shape = self.train_in.shape
        self.test_shape = self.val_in.shape
        self.features = feature_labels
        self.samples = row_labels
        self.batch = batch

        # create and store model
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(60, activation='tanh', recurrent_activation='sigmoid', \
                                                input_shape=self.train_shape[-2:], return_sequences=True, name='Input'))
        self.model.add(tf.keras.layers.Dropout(.2, name='Drop1'))
        self.model.add(tf.keras.layers.LSTM(5, activation='tanh', recurrent_activation='sigmoid', \
                                                input_shape=self.train_shape[-2:], return_sequences=True, name='Hidden'))
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
        self.model.compile(loss=loss, optimizer=optimizer)
        print(self.model.summary())

    def train(self, EPOCHS=10):
        """ Trains model in data given during Stocker's init.

            WIP
        """
        #early = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1, mode='min')
        self.history = self.model.fit(x=self.train_in, y=self.train_out, epochs=EPOCHS, \
                            validation_split=.3, batch_size=self.batch)

        # plot losses
        pyplot.figure()
        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='test')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Error')
        pyplot.legend()
        pyplot.suptitle('Error')
        pyplot.savefig('./plots/error.png')
        print()

    def evaluate(self):
        """ Evalate model and output loss """
        self.loss = self.model.evaluate(x=self.val_in, y=self.val_out, batch_size=self.batch)

    def save_model(self, dir='./models/'):
        """ Save model to given folder. models folder is default """

        if not os.path.exists(dir):
            os.mkdir(dir)

        dir += self.symbol+'.h5'

        self.model.save(dir)

    def predict_data(self, data_in, sample_size, future_steps=1):
        """ Method predicts 1 step ahead given data 
            Sample number must be greater than batch size
        """

        predictions = self.model.predict(data_in, verbose=1)

        return predictions

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
        hist = dh.daily_adjusted(symbol, parse.key, compact=False)
        hist = hist.drop(['6. volume', '7. dividend amount', '8. split coefficient'], axis=1)
        hist = hist.reindex(index=hist.index[::-1])
        data[symbol] = hist
        print(hist)
        print()

        pyplot.figure()
        hist.plot(subplots=True)
        pyplot.suptitle('Input Features')
        pyplot.savefig('./plots/input.png')

        """ Data Preprocessing """
        
        split = round(len(hist.index)*7/10)

        # standardize data
        standard, mean, std = dh.standardize(hist, split)

        print(standard)

        pyplot.figure()
        standard.plot(subplots=True)
        pyplot.suptitle('Standardized Features')
        pyplot.savefig('./plots/standardized.png')
        
        """ -------------------------------- """

        # test Stocker methods
        model = Stocker(symbol, standard, split, hist.columns, hist.index)
        model.train(70)
        model.evaluate()
        model.save_model()
        predictions = model.predict_data(model.val_in, model.test_shape[0])

        standard_numpy = standard[split:]['5. adjusted close'].to_numpy()
        pyplot.figure()
        pyplot.plot(standard_numpy, label='True Values')
        pyplot.plot(predictions[:, 0], label='Predictions')
        pyplot.xlabel('Time Step')
        pyplot.ylabel('Adjusted Close')
        pyplot.suptitle('Predictions')
        pyplot.legend()
        pyplot.savefig('./plots/predictions.png')