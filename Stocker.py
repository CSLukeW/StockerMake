""" 
Author: Luke Williams

License: LGPLv3

Stocker module for easy, modular prototyping of LSTM neural networks. 

Coming soon:
    Support for more layer types
    Multi-step future models
    Optimizer customization
    Customizable past-window (how far back to look)
"""

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
    def __init__(self, symbol, data, depth=1, node_counts=[100], batch=50, test_size=.2, loss='mse', learning_rate=.001, inpath=None):
        """ Creating Stocker instance immediately creates model 

            Args:
                symbol ---- ticker symbol of desired stock
                data ---- full set of training and testing data (split can be specified) (must be dataframe)

            Kwargs:
                depth ---- number of computational layers to be added to neural network
                node_counts ---- list of node counts for specified layers (len(node_counts) must be equal to depth)
                batch ---- batch size of data
                test_size ---- proportion of data to be used as validation set
                loss ---- loss function to be used in training (must be supported by tf.keras)
                learning_rate ---- learning rate to be used by the optimizer
                inpath ---- filepath of existing model to be loaded instead of training a new one

        """

        self.batch = batch
        self.symbol = symbol

        # load model if specified
        if inpath != None:
            self.model = tf.keras.models.load_model(dir)
            return

        # assign optimizer (support for optimizer customization coming)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        data_numpy = data.to_numpy()
        split = int(data_numpy.shape[0]*(1-test_size))

        # store data in numpy format given data in dataframe
        self.train_in, self.train_out = helper.single_step_data(data_numpy, data_numpy[:, 4], 0, split, 60, 1, 1)
        self.val_in, self.val_out = helper.single_step_data(data_numpy, data_numpy[:, 4], split, None, 60, 1, 1)

        print('Constructing model...')
        # create and store model
        self.model = tf.keras.Sequential()

        for i in range(depth):
            self.model.add(tf.keras.layers.LSTM(node_counts[i], activation='tanh', recurrent_activation='sigmoid', \
                                                    input_shape=self.train_in.shape[-2:], return_sequences=True, name='LSTM'+str(i)))
            self.model.add(tf.keras.layers.Dropout(.3))

        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
        self.model.compile(loss=loss, optimizer=optimizer)
        print('Model Constructed!\n')
        print(self.model.summary())

    def train(self, EPOCHS, early_stopping, plot):
        """ Trains model in data given during Stocker's init.

            args:
                EPOCHS ---- max number of epochs to run training on
                early_stopping ---- flag deciding whether or not to implement early stopping (patience=5)
                plot ---- flag deciding whether or not to save plots of error
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
        """ Evalate model and output loss 
        
            args
                data ---- if specified, use as validation data instead of data stored by Stocker object (must be pd.Dataframe)
        """

        if data != None:
            self.val_in, self.val_out = dh.single_step_data(data, data[:, 4], 0, None, 60, 1, 1)
            return

        self.loss = self.model.evaluate(x=self.val_in, y=self.val_out, batch_size=self.batch)

    def save_model(self, dir='./models/'):
        """ Save model to given folder. models folder is default 
        
            args:
                dir ---- folder where models are to be stored
        """

        if not os.path.exists(dir):
            os.mkdir(dir)

        dir += self.symbol+'.h5'

        self.model.save(dir)

    def predict_data(self, data_in):
        """ Method makes single-step prediction given at least 60 prior data points
            
            args:
                data_in ---- data on which to perform a prediction (numpy array)

            returns:
                predictions ---- numpy array of predicted values
        """

        predictions = self.model.predict(data_in, verbose=1)

        return predictions

if __name__ == '__main__':

    """ Test/Demo of Stocker module """

    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument('key', help='User API Key')
    parser.add_argument('-depth', default=1, type=int, help='Depth of LSTM neural net, (default 1)')
    parser.add_argument('-node_counts', default=[100], type=int, nargs='*', help='Node counts for each LSTM layer. (number of args must be equal to depth')
    parser.add_argument('-batch', default=50, type=int, help='Batch size (default 50)')
    parser.add_argument('-test_size', default=.2, type=float, help='Percentage of samples to use for testing (decimal form)')
    parser.add_argument('-loss', default='mse', help='Loss function for Neural Net')
    parser.add_argument('-learning_rate', default=.001, type=float, help='Learning rate of Neural Net (Default .001)')
    parser.add_argument('-epochs', default=100, type=int, help='Epoch count for training (Default 100)')
    parser.add_argument('-model_in', default=None, help='Path of pre-made model to load')
    parser.add_argument('--early_stop', action='store_true', default=False, help='Apply early stopping to model training (Patience 10)')
    parser.add_argument('--plots', action='store_true', help='Saves all plots to plots folder')
    parser.add_argument('symbols', nargs='*', help="List of symbols to train")
    parse = parser.parse_args()

    data = {}
    symbols = parse.symbols

    for symbol in symbols:

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
        model = Stocker(symbol, hist, parse.depth, parse.node_counts, parse.batch, parse.test_size, parse.loss, \
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