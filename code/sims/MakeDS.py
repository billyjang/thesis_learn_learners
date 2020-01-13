import numpy as np
import pandas as pd
from DSSynthesizer import DSSynthesizer

num_samples = 10000
low = -100
high = 100
n = 1
d = 1
num_updates = 20
intercept = False
std_dev = 0.0
func = "LearnN"
pwd = "/Users/william/Documents/2019-2020/Thesis/code/sims"

ds_manager = DSSynthesizer()

dataset_dict = ds_manager.create_dataset_learnN(num_samples=num_samples, d=d, num_updates=num_updates)
for data in dataset_dict:
    print(dataset_dict[data])
#dataset_dict = ds_manager.create_dataset_learnX(num_samples=num_samples, low=low, high=high, n=n, d=d, num_updates=num_updates, intercept=intercept, std_dev=std_dev)
#dataset_dict = ds_manager.create_dataset_learnX(num_samples=num_samples)

for data in dataset_dict:
    path = pwd+"/"+func+"/"+"Raw/"+data+"_"+str(num_samples)+"_"+str(low)+"_"+str(high)+"_"+str(n)+"_"+str(d)+"_"+str(num_updates)+"_"+str(intercept)+".csv"
    ds_manager.write_dataset(dataset_dict[data], path)

polycols = 0
if func == "learnBeta":
    polycols = (d+1)*num_updates
if func == "LearnX":
    polycols = (d+1)*(num_updates)
if func == "LearnN":
    polycols = (d+1)*(num_updates)
#try scaling before feature extr
#.9588 w/o scaling
#

if func != "LearnFunc" or func != "LearnN":
    dataset_dict['Xtrain'] = ds_manager.poly_features(dataset_dict['Xtrain'], polycols)
    dataset_dict['Xtest'] = ds_manager.poly_features(dataset_dict['Xtest'], polycols)

dataset_dict['Xtrain'] = ds_manager.scaler(dataset_dict['Xtrain'])
#dataset_dict['ytrain'] = ds_manager.scaler(dataset_dict['ytrain'])

dataset_dict['Xtest'] = ds_manager.scaler(dataset_dict['Xtest'])
#dataset_dict['ytest'] = ds_manager.scaler(dataset_dict['ytest'])

# Processed y sets are the same as raw
for data in dataset_dict:
    path = pwd+"/"+func+"/"+"Processed/"+data+"_"+str(num_samples)+"_"+str(low)+"_"+str(high)+"_"+str(n)+"_"+str(d)+"_"+str(num_updates)+"_"+str(intercept)+".csv"
    ds_manager.write_dataset(dataset_dict[data], path)
