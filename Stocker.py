""" training script """

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
from matplotlib import pyplot
import os
from sklearn.preprocessing import MinMaxScaler

import helpers as helper

class Stocker:
    def __init__(self, symbol, data, depth, batch, test_size, loss, learning_rate, inpath):
        """ Creating Stocker instance immediately creates model 

            Model (WIP) is a two-layer LSTM. Defaults to Mean Squared Error
            loss function and ADAM optimizer function.
        """

        self.batch = batch
        self.symbol = symbol

        if inpath != None:
            self.model = tf.keras.models.load_model(dir)
            return

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        data_numpy = data.to_numpy()
        split = int(data_numpy.shape[0]*(1-test_size))

        # store data in numpy format
        self.train_in, self.train_out = helper.single_step_data(data_numpy, data_numpy[:, 4], 0, split, 60, 1, 1)
        self.val_in, self.val_out = helper.single_step_data(data_numpy, data_numpy[:, 4], split, None, 60, 1, 1)

        print('Constructing model...')
        # create and store model
        self.model = tf.keras.Sequential()

        for i in range(depth):
            self.model.add(tf.keras.layers.LSTM(max(1, 100-i*30), activation='tanh', recurrent_activation='sigmoid', \
                                                    input_shape=self.train_in.shape[-2:], return_sequences=True, name='Input'))
            self.model.add(tf.keras.layers.Dropout(.3))

        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
        self.model.compile(loss=loss, optimizer=optimizer)
        print('Model Constructed!\n')
        print(self.model.summary())

    def train(self, EPOCHS, early_stopping, plot):
        """ Trains model in data given during Stocker's init.
        """
        early = None
        if early_stopping:
            early = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1, mode='min')
            self.history = self.model.fit(x=self.train_in, y=self.train_out, epochs=EPOCHS, \
                                            validation_split=.2, batch_size=self.batch, callbacks=[early])
        else:
            self.history = self.model.fit(x=self.train_in, y=self.train_out, epochs=EPOCHS, \
                                            validation_split=.2, batch_size=self.batch)

        if plot:
            # plot losses
            pyplot.figure()
            pyplot.plot(self.history.history['loss'], label='train')
            pyplot.plot(self.history.history['val_loss'], label='test')
            pyplot.xlabel('Epoch')
            pyplot.ylabel('Error')
            pyplot.legend()
            pyplot.suptitle('Error')
            pyplot.savefig(helper.make_dir('./plots/' + self.symbol) + '/error.png')
            print()

    def evaluate(self, data=None):
        """ Evalate model and output loss """

        if data != None:
            self.val_in, self.val_out = dh.single_step_data(data, data[:, 4], 0, None, 60, 1, 1)
            return

        self.loss = self.model.evaluate(x=self.val_in, y=self.val_out, batch_size=self.batch)

    def save_model(self, dir='./models/'):
        """ Save model to given folder. models folder is default """

        if not os.path.exists(dir):
            os.mkdir(dir)

        dir += self.symbol+'.h5'

        self.model.save(dir)

    def predict_data(self, data_in):
        """ Method predicts 1 step ahead given data 
            Sample number must be greater than batch size
        """

        predictions = self.model.predict(data_in, verbose=1)

        return predictions

if __name__ == '__main__':

    """ Test/Demo of Stocker module """

    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument('key', help='User API Key')
    parser.add_argument('-depth', default=1, help='Depth of LSTM neural net, (default 1)')
    parser.add_argument('-batch', default=50, help='Batch size (default 50)')
    parser.add_argument('-test_size', default=.2, help='Percentage of samples to use for testing (decimal form)')
    parser.add_argument('-loss', default='mse', help='Loss function for Neural Net')
    parser.add_argument('-learning_rate', default=.001, help='Learning rate of Neural Net (Default .001)')
    parser.add_argument('-epochs', default=100, help='Epoch count for training (Default 100)')
    parser.add_argument('-model_in', default=None, help='Path of pre-made model to load')
    parser.add_argument('--early_stop', action='store_true', default=False, help='Apply early stopping to model training (Patience 10)')
    parser.add_argument('--plots', action='store_true', help='Saves all plots to plots folder')
    parser.add_argument('symbol', help="List of symbols to train (Place all at end of command)")
    parse = parser.parse_args()

    data = {}
    symbol = parse.symbol

    # read historical daily data from alpha_vantage
    # store in python dict
    hist = helper.daily_adjusted(symbol, parse.key, compact=False)
    hist = hist.drop(['6. volume', '7. dividend amount', '8. split coefficient'], axis=1)
    hist = hist.reindex(index=hist.index[::-1])
    data[symbol] = hist
    print(hist)
    print()

    if parse.plots:
        pyplot.figure()
        hist.plot(subplots=True)
        pyplot.suptitle('Input Features')
        pyplot.savefig(helper.make_dir('./plots/' + symbol) + '/input.png')

    # test Stocker methods
    model = Stocker(symbol, hist, parse.depth, parse.batch, parse.test_size, parse.loss, \
                        parse.learning_rate, parse.model_in)
    model.train(parse.epochs, parse.early_stop, parse.plots)
    model.evaluate()
    model.save_model()
    predictions = model.predict_data(model.val_in)

    standard_numpy = hist[int(hist.shape[0]*(1-parse.test_size)):]['5. adjusted close'].to_numpy()

    if parse.plots:
        pyplot.figure()
        pyplot.plot(standard_numpy, label='True Values')
        pyplot.plot(predictions[:, 0], label='Predictions')
        pyplot.xlabel('Time Step')
        pyplot.ylabel('Adjusted Close')
        pyplot.suptitle('Predictions')
        pyplot.legend()
        pyplot.savefig(helper.make_dir('./plots/' + symbol) + '/predictions.png')