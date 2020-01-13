import numpy as np
import pandas as pd
from DSSynthesizer import DSSynthesizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error

from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

num_samples = 10000
low = -100
high = 100
n = 1
d = 1
num_updates = 20
intercept = False
func = "LearnN"
pwd = "/Users/william/Documents/2019-2020/Thesis/code/sims"
datasets = ['Xtrain', 'ytrain', 'Xtest', 'ytest']

def constructPath(num_samples, low, high, n, d, num_updates, intercept, func, pwd, dataset):
    return pwd+"/"+func+"/"+"Processed/"+dataset+"_"+str(num_samples)+"_"+str(low)+"_"+str(high)+"_"+str(n)+"_"+str(d)+"_"+str(num_updates)+"_"+str(intercept)+".csv"

paths = [constructPath(num_samples, low, high, n, d, num_updates, intercept, func, pwd, dataset) for dataset in datasets]

print(paths)
ds_manager = DSSynthesizer()

Xtrain, ytrain, Xtest, ytest = ds_manager.read_dataset(paths)
ytrain = ytrain
ytest = ytest

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

def regressor():
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=input_size))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(output_size))
    model.compile(loss='mean_absolute_error', optimizer='sgd')
    return model
#model = LogisticRegression(penalty='l1', dual=True, verbose=3)
#model = SVR(kernel='poly', degree=5, max_iter=10, verbose=True)
#model = KNeighborsRegressor(n_neighbors=5)
model = KerasRegressor(build_fn=regressor, batch_size=32, epochs=200)
#model = MLPRegressor(hidden_layer_sizes=(86,100,100,10), n_iter_no_change=20, max_iter=300, verbose=True, tol=.00000001, activation='relu')
#model_pwd = pwd+"/"+func+"/"+"Models/"+str(num_samples)+"_"+str(low)+"_"+str(high)+"_"+str(n)+"_"+str(d)+"_"+str(num_updates)+"_"+str(intercept)+".h5"
#model.save(model_pwd)
model.fit(Xtrain, ytrain)
#loss = model.evaluate(Xtest, ytest)
#loss = model.score(Xtrain,ytrain)
#ypreds = model.predict(Xtest, verbose=1)
ypreds = model.predict(Xtest)
ypreds_pwd = pwd+"/"+func+"/"+"Predictions/"+str(num_samples)+"_"+str(low)+"_"+str(high)+"_"+str(n)+"_"+str(d)+"_"+str(num_updates)+"_"+str(intercept)+".csv"
ds_manager.write_dataset(ypreds, ypreds_pwd)

print("Trying to learn: " + func)
print("number samples: " + str(num_samples))
print("range: " + "["+str(low)+", "+str(high)+"]")
print("number of points in original ds: " + str(n))
print("dim: " + str(d))
print("number of updates:" + str(num_updates))
print("intercept: " + str(intercept))
#print("loss on test set: " + str(loss))
actual_loss = mean_absolute_error(ytest, ypreds)
print("actual?: " + str(actual_loss))
# quick thing, shouldn't it just tell from the beta? like that shouldn't be too hard right?