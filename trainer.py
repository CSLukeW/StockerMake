""" training script """

import pandas as pd
import argparse

import data_helpers as dh

parser = argparse.ArgumentParser(description="Model Training Script")
parser.add_argument('key', nargs='1', help='User API Key', default='8X10BAT7HUYHPS8Y')
parser.add_argument('outdir', nargs='1', default='/models/1/', help="Directory for stored model (if more than one symbol, symbol will be added at end of path for each model")
parser.add_argument('symbols', nargs=argparse.REMAINDER, help="List of symbols to train (All at end of command)")

symbols = ['AAPL', 'TSLA', 'MSFT']
data = []

for symbol in symbols:
    hist = dh.daily(symbol, key, compact=False)
    data.append(hist)
    print(hist)
    print()