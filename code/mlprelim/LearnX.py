import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from CreateDataSet import create_dataset_helper
from CreateDataSet import read_dataset
from CreateDataSet import write_dataset

Xtrain, ytrain, Xtest, ytest = read_dataset(['Xtrainpoly.csv', 'ytrainpoly.csv', 'Xtestpoly.csv', 'ytestpoly.csv'])

#nn = MLPRegressor(hidden_layer_sizes=(100,100), activation='tanh',)
#nn.fit(Xtrain,ytrain)
#maybe use keras/tf so can use l1 penalty? or logreg/svm

model = Sequential()
model.add(Dense(32, input_dim=7, activation='tanh'))
#model.add(Dense(32, activation='tanh'))
# so far worked better with single layer and polynomial feature trans on ytrain but maybe
# that is probably unfair?
model.add(Dense(5, activation='tanh'))
model.compile(loss='mean_absolute_error')
model.fit(Xtrain, ytrain, epochs=10)

print(model.evaluate(Xtest, ytest))
ypreds = model.predict(Xtest)
write_dataset("ypreds", ypreds)