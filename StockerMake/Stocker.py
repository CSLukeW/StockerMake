""" 
Author: Luke Williams

License: LGPLv3

Stocker module for easy, modular prototyping of LSTM neural networks. 

Coming soon:
    Support for more layer types
    Multi-step future models
    Optimizer customization
    Customizable past-window (how far back to consider)
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

from . import helpers as helper

class Stocker:
    def __init__(self, symbol, data, depth=1, node_counts=[100], batch=50, test_size=.2, loss='mse', learning_rate=.001, inpath=None, normalize=True):
        """ Creating Stocker instance and model

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

        # swap data to numpy for easier manipulation
        data_numpy = data.to_numpy()
        split = int(data_numpy.shape[0]*(1-test_size))

        # store data in numpy format given data in dataframe
        self.train_in, self.train_out = helper.single_step_data(data_numpy, data_numpy[:, 4], 0, split, 60, 1, 1, normalize)
        self.val_in, self.val_out = helper.single_step_data(data_numpy, data_numpy[:, 4], split, None, 60, 1, 1, normalize)

        print('Constructing model...')

        self.model = tf.keras.Sequential()
        # build model based on user inputs
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
                data ---- if specified, use as validation data instead of data stored by Stocker object (must be pd.Dataframe).
                          Must specify if model was loaded from file.
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