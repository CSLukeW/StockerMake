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
                    loss='mse', optimizer=tf.keras.optimizers.Adam()):
        """ Creating Stocker instance immediately creates model 

            Model (WIP) is a two-layer LSTM. Defaults to Mean Squared Error
            loss function and ADAM optimizer function.
        """

        past = 60
        future = 1
        step = 1
        buffer = 365
        batch = 365

        data_numpy = data.to_numpy()

        # store data in numpy format
        self.train_in, self.train_out = dh.single_step_data(data_numpy, data_numpy[:, 1], 0, split, past, future, step)
        self.val_in, self.val_out = dh.single_step_data(data_numpy, data_numpy[:, 1], split, None, past, future, step)

        # store tf.Dataset format
        self.training_data = tf.data.Dataset.from_tensor_slices((self.train_in, self.train_out)).cache().shuffle(buffer).batch(batch).repeat()
        self.test_data = tf.data.Dataset.from_tensor_slices((self.val_in, self.val_out)).batch(batch).repeat()

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
                                            input_shape=self.train_shape[-2:]))
        self.model.add(tf.keras.layers.Dense(5))
        self.model.compile(loss=loss, optimizer=optimizer)
        print(self.model.summary())

    def train(self, EPOCHS=10):
        """ Trains model in data given during Stocker's init.

            WIP
        """
        self.history = self.model.fit(self.training_data, epochs=EPOCHS, \
                            steps_per_epoch=10, \
                            validation_data=self.test_data)

        # plot losses
        pyplot.figure()
        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='test')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Error')
        pyplot.legend()
        pyplot.suptitle('Error')
        pyplot.savefig('./plots/error.png')

    def evaluate(self):
        """ Evalate model and output loss """
        self.loss= self.model.evaluate(self.test_data, steps=int(self.test_shape[0]/60))
        print()
        print('Test LOSS: ', self.loss)

    def save_model(self, dir='./models/'):
        """ Save model to given folder. models folder is default """

        if not os.path.exists(dir):
            os.mkdir(dir)

        dir += self.symbol+'.h5'

        self.model.save(dir)

    def predict_data(self, data_in, sample_size, future_steps=2):
        """ Method predicts 1 step ahead given data 
            Sample number must be greater than batch size
        """

        predictions = pd.DataFrame(self.model.predict(data_in, steps=1), columns=self.features)

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
        hist = dh.daily(symbol, parse.key, compact=False)
        hist.head()
        data[symbol] = hist
        #print(hist)
        #print()

        pyplot.figure()
        hist.plot(subplots=True)
        pyplot.suptitle('Input Features')
        pyplot.savefig('./plots/input.png')

        """ Data Preprocessing """
        
        split = round(len(hist.index)*7/10)

        # standardize data
        standard = dh.standardize(hist, split)
        
        """ -------------------------------- """

        # test Stocker methods
        model = Stocker(symbol, standard, split, hist.columns, hist.index)
        model.train(5)
        model.evaluate()
        model.save_model()
        predictions = model.predict_data(model.test_data, model.test_shape[0])

        predictions = pd.DataFrame(np.asarray(predictions), columns=hist.columns)

        print(standard)
        pyplot.figure()
        standard[:split][:-1].plot(subplots=True)
        predictions[:][:-1].plot(subplots=True)
        pyplot.legend()
        pyplot.suptitle('Predictions')
        pyplot.savefig('./plots/predictions.png')