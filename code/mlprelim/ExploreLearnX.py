import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from CreateDataSet import create_dataset_helper
from CreateDataSet import read_dataset
from CreateDataSet import write_dataset

Xtrain, ytrain, Xtest, ytest = read_dataset(['Xtrainpoly.csv', 'ytrainpoly.csv', 'Xtestpoly.csv', 'ytestpoly.csv'])

model = Sequential()
model.add(Dense(32, input_dim=7, activation='tanh'))

