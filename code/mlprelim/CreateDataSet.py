import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

def create_dataset_helper(num_samples,low,high,n,d,num_updates):
    #assumes n = 1 for original dataset
    #probably move to keras/tf backend as n grows
    #flatten array for now but maybe keep structure (similar to cnns)
    original_ds = np.random.randint(low,high+1,size=(num_samples,n,d+1))
    updates = np.random.randint(low,high+1,size=(num_samples,num_updates,d+1))
    betas = np.zeros((num_samples,num_updates+1,d))
    lm = LinearRegression(fit_intercept=False)
    for i in range(num_samples):
        if i % 1000 == 0:
            print("i:",i)
        ds = original_ds[i]
        lm.fit(ds[:,:-1], ds[:,-1])
        betas[i,0] = lm.coef_
        for j in range(num_updates):
            ds = np.append(ds,[updates[i,j]],axis=0) #careful with pointers
            lm.fit(ds[:,:-1],ds[:,-1])
            betas[i,j+1] = lm.coef_
            
    original_ds_reshape = np.reshape(original_ds, (num_samples, n*(d+1)))
    updates_reshape = np.reshape(updates, (num_samples, num_updates*(d+1)))
    X = np.append(updates_reshape, betas.reshape(num_samples,d*(num_updates+1)),axis=1)
    #normalize(X, copy=False)
    MinMaxScaler(copy=False)
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
            df.to_csv(data+".csv",index=None,header=None)
    return dataset