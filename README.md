# StockerMake: Modular Prototyping Tool for Stock Market Prediction Models

## Installation

Install using [pip](https://pip.pypa.io/en/stable/) installer.

```bash
pip install StockerMake
```

## Usage

### The StockerMake Script

The StockerMake script combines the Stocker and helper modules into one to create an easy, convenient, and modular neural network prototyping tool designed for stock market prediction. The script will take the user's desired parameters and create, train, and evaluate a model fitting said parameters. This allows the user to quickly analyze model prototypes, make adjustments, and iterate on model designs.

#### Arguments:
    Positional:
        key ---- User's Alpha_Vantage API key
        symbols ---- Ticker symbols to create models for

    Optional:
        depth ---- number of layers to include in the neural network (def: 1)
        node_counts ---- list of node counts for layers (len(node_counts) must equal depth)
        batch ---- batch size of input data set (def: [100])
        test_size ---- proportion of dataset to use as validation (def: .2)
        loss ---- identifier string of keras-supported loss function to be used in training (def: mse)
        learning_rate ---- learning rate to be used by the Adam optimizer
        epochs ---- maximum number of epochs to train the model (def: 100)
        model_in ---- file path of pre-made model to load
        early_stop ---- flag deciding whether to apply early stopping (patience 5) to the training phase
        plots ---- flag deciding whether to save loss, input, and prediction graphs

#### Usage Example:
    StockerMake APIKEY FORD MSFT --early_stop --plots

### The Stocker Module

If you would like to use your own data pipelines as inputs, the Stocker and data helper modules can be used separately from the main script.

```Python
from Stocker import *

""" Data operations (assign data, pred_data)
______________________________________________
"""

stkr = Stocker('FORD', data)
stkr.train(25, True, True))
stkr.evaluate()
stkr.save_model('./models/')
stkr.predict_data(pred_data)
```
The above code will take prepared data, create a stocker instance for the FORD ticker, train for 25 epochs, save the model to the models folder as FORD.h5 and predict a data point based on the user's prepared prediction data.

### The Helpers module

This module includes the data operation helper functions used by Stocker.

daily_adjusted() returns a pandas dataframe of historical daily adjusted stock data.

single_step_data takes a full dataset and creates a single-step timeseries dataset from it for input into an LSTM model.

make_dir() is a filepath helper to assist with saving models.

## Coming soon!

Stocker:
- Support for more layer types
- Multi-step future models
- Optimizer customization
- Customizable past-window (how far back to consider)

Helpers:
- Data for multi-step prediction
- Data normalization option
- Compact (last 60) data

## Contributions

Please send pull requests! I am a full-time student, so development and support for Stocker will likely be slow with me working alone. I welcome any and all efforts to contribute!

## License

[GNU LGPLv3](https://choosealicense.com/licenses/lgpl-3.0/)