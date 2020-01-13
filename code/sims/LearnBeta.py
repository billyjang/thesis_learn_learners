import numpy as np
import pandas as pd
from DSSynthesizer import DSSynthesizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model

num_samples = 10000
low = -100
high = 100
n = 1000
d = 1
num_updates = 10
intercept = False
func = "learnBeta"
pwd = "/Users/william/Documents/2019-2020/Thesis/code/sims"
datasets = ['Xtrain', 'ytrain', 'Xtest', 'ytest']

def constructPath(num_samples, low, high, n, d, num_updates, intercept, func, pwd, dataset):
    return pwd+"/"+func+"/"+"Processed/"+dataset+"_"+str(num_samples)+"_"+str(low)+"_"+str(high)+"_"+str(n)+"_"+str(d)+"_"+str(num_updates)+"_"+str(intercept)+".csv"

paths = [constructPath(num_samples, low, high, n, d, num_updates, intercept, func, pwd, dataset) for dataset in datasets]

print(paths)
ds_manager = DSSynthesizer()

Xtrain, ytrain, Xtest, ytest = ds_manager.read_dataset(paths)

print(Xtrain.shape)
print(ytrain.shape)
print(Xtest.shape)
print(ytest.shape)
input_size = Xtrain.shape[1]
output_size = 1
if len(ytrain.shape) > 1:
    output_size = ytrain.shape[1]

print(input_size)
print(output_size)
# TODO: implement cross validation with keras

model = Sequential()
model.add(Dense(100, activation='tanh', input_dim=input_size))
model.add(Dense(100, activation='tanh'))
model.add(Dense(output_size, activation='tanh'))
model.compile(loss='mean_absolute_error', optimizer='sgd')
model.fit(Xtrain, ytrain, epochs=100)

model_pwd = pwd+"/"+func+"/"+"Models/"+str(num_samples)+"_"+str(low)+"_"+str(high)+"_"+str(n)+"_"+str(d)+"_"+str(num_updates)+"_"+str(intercept)+".h5"
model.save(model_pwd)

loss = model.evaluate(Xtest, ytest)

ypreds = model.predict(Xtest, verbose=1)
ypreds_pwd = pwd+"/"+func+"/"+"Predictions/"+str(num_samples)+"_"+str(low)+"_"+str(high)+"_"+str(n)+"_"+str(d)+"_"+str(num_updates)+"_"+str(intercept)+".csv"
ds_manager.write_dataset(ypreds, ypreds_pwd)

print("Trying to learn: " + func)
print("number samples: " + str(num_samples))
print("range: " + "["+str(low)+", "+str(high)+"]")
print("number of points in original ds: " + str(n))
print("dim: " + str(d))
print("number of updates:" + str(num_updates))
print("intercept: " + str(intercept))
print("loss on test set: " + str(loss))
