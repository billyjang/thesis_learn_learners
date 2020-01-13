import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

def create_learn_n_dataset_helper(num_samples, low, high, n, d, num_updates):
    
    for i in range(num_samples):
        original_ds_size = np.random.randint(0, 1000)
        original_ds = np.random.randint(low, high+1, size=(original_ds_size, d+1))
        updates = np.random.randint(low, high+1, size=(num_updates, d+1))
        betas = np.zeros((num_updates+1, d))
        lm = LinearRegression(fit_intercept=False)
        lm.fit(original_ds[:,:-1], original_ds[:,-1])
        betas[i,0] = lm.coef_
        for j in range(num_updates):
            ds = np.append(original_ds, updates[i], axis=0)
            lm.fit(ds[:,:-1], ds[:,-1])
            betas[i,j] = lm.coef_

def create_learn_n_dataset(num_samples=1000, low=-100, high=100, n=1, d=1, num_updates=1, save=False):
    Xtrain,ytrain = create_dataset_helper(num_samples,low,high,n,d,num_updates)
    Xtest,ytest = create_dataset_helper((int)(.2*num_samples), low,high,n,d,num_updates)
    dataset = {"Xtrain": Xtrain, "ytrain": ytrain,
               "Xtest": Xtest, "ytest": ytest}
    if save:
        for data in dataset:
            df = pd.DataFrame(dataset[data])
            df.to_csv(data+".csv", index=None, header=None)
    return dataset

def create_dataset_helper(num_samples,low,high,n,d,num_updates):
    #assumes n = 1 for original dataset
    #probably move to keras/tf backend as n grows
    #flatten array for now but maybe keep structure (similar to cnns)
    original_ds = np.random.randint(low,high+1,size=(num_samples,n,d+1))
    updates = np.random.randint(low,high+1,size=(num_samples,num_updates,d+1))
    betas = np.zeros((num_samples,num_updates+1,d))
    lm = LinearRegression(fit_intercept=False)
    for i in range(num_samples):
        if i % 10000 == 0:
            print("i:",i)
        ds = original_ds[i]
        lm.fit(ds[:,:-1], ds[:,-1])
        #might need to change this line hahaha
        betas[i,0] = lm.coef_
        for j in range(num_updates):
            ds = np.append(ds,[updates[i,j]],axis=0) #careful with pointers
            lm.fit(ds[:,:-1],ds[:,-1])
            betas[i,j+1] = lm.coef_
            
    original_ds_reshape = np.reshape(original_ds, (num_samples, n*(d+1)))
    updates_reshape = np.reshape(updates, (num_samples, num_updates*(d+1)))
    X = np.append(updates_reshape, betas.reshape(num_samples,d*(num_updates+1)),axis=1)
    #normalize(X, copy=False)
    #MinMaxScaler(copy=False)
    y = original_ds_reshape
    return X,y

def create_dataset(num_samples=1000,low=-100,high=100,n=1,d=1,num_updates=1,save=False):
    Xtrain,ytrain = create_dataset_helper(num_samples,low,high,n,d,num_updates)
    Xtest,ytest = create_dataset_helper((int)(.2*num_samples), low,high,n,d,num_updates)
    dataset = {"Xtrain": Xtrain, "ytrain": ytrain,
               "Xtest": Xtest, "ytest": ytest}
    if save:
        for data in dataset:
            df = pd.DataFrame(dataset[data])
            df.to_csv(data+".csv", index=None, header=None)
    return dataset

def create_other_dataset_helper(num_samples,low,high,n,d,num_updates):
    original_ds = np.random.randint(low,high+1,size=(num_samples,n,d+1))
    updates = np.random.randint(low,high+1,size=(num_samples,num_updates,d+1))
    betas = np.zeros((num_samples, num_updates, d))
    y = np.zeros((num_samples, d))
    lm = LinearRegression(fit_intercept=False)
    for i in range(num_samples):
        if i % 10000 == 0:
            print("i: ",i)
        ds = original_ds[i]
        lm.fit(ds[:,:-1], ds[:,-1])
        betas[i,0] = lm.coef_
        for j in range(num_updates):
            ds = np.append(ds,[updates[i,j]],axis=0)
            lm.fit(ds[:,:-1], ds[:,-1])
            if j == num_updates-1:
                y[i] = lm.coef_
            else:
                betas[i,j+1] = lm.coef_
    updates_reshape = np.reshape(updates, (num_samples, num_updates*(d+1)))
    betas_reshape = np.reshape(betas, (num_samples,d*(num_updates)))
    X = np.append(updates_reshape, betas_reshape,axis=1)
    return np.reshape(original_ds, (num_samples, n*(d+1))),X,y

def create_other_dataset(num_samples=1000, low=-100, high=100, n=1, d=1, num_updates=1, save=False):
    ods, Xtrain, ytrain = create_other_dataset_helper(num_samples,low,high,n,d,num_updates)
    ods, Xtest,ytest = create_other_dataset_helper((int)(.2*num_samples), low,high,n,d,num_updates)
    dataset = {"Xtrain": Xtrain, "ytrain": ytrain,
               "Xtest": Xtest, "ytest": ytest}
    if save:
        for data in dataset:
            df = pd.DataFrame(dataset[data])
            df.to_csv(data+"h.csv", index=None, header=None)
    print(pd.DataFrame(ods).head())
    return dataset

def read_dataset(paths):
    files = []
    if type(paths) == list:
        for path in paths:
            files.append(np.genfromtxt(path, delimiter=','))
    else:
        files.append(np.genfromtxt(paths, delimiter=','))
    return tuple(files)

def write_dataset(paths, data, add=""):
    if type(paths) == list:
        for i in range(len(paths)):
            if ".csv" in paths[i]:
                paths[i] = paths[i].replace(".csv","")
            df = pd.DataFrame(data)
            df.to_csv(paths[i]+add+".csv", index=None, header=None)
    else:
        df = pd.DataFrame(data)
        if ".csv" in paths:
            paths = paths.replace(".csv","")
        df.to_csv(paths+add+".csv", index=None, header=None)

if __name__ == "__main__":
    dataset = create_dataset(num_samples=1000000, low=0, high=20, n=1, d=1, num_updates=10, save=False)
    for data in dataset:
        df = pd.DataFrame(dataset[data])
        df.to_csv(data+"extended.csv", index=None, header=None)
