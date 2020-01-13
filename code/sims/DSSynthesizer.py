import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression

# needs commenting and testing
# needs learn n fix
# needs preprocessing
# would also like some sort of file structure for this

class DSSynthesizer:
    def __init__(self):
        pass
    
    def learnFunc(self, num_samples):
        Xtrain,ytrain = self.learnFunc_helper(num_samples)
        Xtest,ytest = self.learnFunc_helper((int)(.2*num_samples))
        dataset = {"Xtrain": Xtrain, "ytrain": ytrain,
                "Xtest": Xtest, "ytest": ytest}
        return dataset

    def learnFunc_helper(self, num_samples):
        #scaling, poly features, negative numbers, more complex
        X = np.random.randint(-5, 5, (num_samples, 2))
        y = X[:,0] * X[:,1] + X[:,0]
        return X,y

    def poly_features(self, data, polycols):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        print(data)
        print(polycols)
        print(data[:,:polycols])
        return np.append(data[:,polycols:], poly.fit_transform(data[:,:polycols]), axis=1)

    def scaler(self, data):
        scaler = MinMaxScaler(copy=False)
        scaler.fit(data)
        scaler.transform(data)
        return data

    # TODO: figure out generators for keras maybe, so we can do multiprocessing?
    def read_dataset(self, paths):
        files = []
        if type(paths) == list:
            for path in paths:
                files.append(np.genfromtxt(path, delimiter=','))
        else:
            files.append(np.genfromtxt(paths, delimiter=','))
        return tuple(files)
    
    def write_dataset(self, data, paths):
        if type(data) == list:
            print('hit')
            print(data)
            for i in range(len(data)):
                df = pd.DataFrame(data[i])
                df.to_csv(paths[i], index=None, header=None)
        else:
            print("also hit")
            print(data)
            df = pd.DataFrame(data)
            df.to_csv(paths, index=None, header=None)

    def test_call(self, num_samples=1000, low=-100, high=100, n=1, d=1, num_updates=1, intercept=False, std_dev=1.0):
        Xtrain,ytrain = self.test(num_samples,low,high,n,d,num_updates,intercept,std_dev)
        Xtest,ytest = self.test((int)(.2*num_samples),low,high,n,d,num_updates,intercept,std_dev)
        dataset = {"Xtrain": Xtrain, "ytrain": ytrain,
                "Xtest": Xtest, "ytest": ytest}
        return dataset

    def test(self, num_samples, low, high, n, d, num_updates, intercept, std_dev):
        original_ds = np.zeros((num_samples, n, d+1))
        for i in range(num_samples):
            r = make_regression(n_samples=n, n_features=d, n_informative=1, coef=True, noise=std_dev, n_targets=1)
            X, y = r[0], r[1]
            original_ds[i] = np.insert(X, d, y, axis=1)
        print("original")
        print(original_ds)
        betas = np.zeros((num_samples,d))
        lm = LinearRegression(fit_intercept=intercept)
        for i in range(num_samples):
            ds = original_ds[i]
            lm.fit(ds[:,:-1], ds[:,-1])
            betas[i] = lm.coef_
        
        original_ds_reshape = np.reshape(original_ds, (num_samples, n*(d+1)))
        #updates_reshape = np.reshape(updates, (num_samples, num_updates*(d+1)))
        #X = np.append(updates_reshape, betas.reshape(num_samples,d*(num_updates+1)),axis=1)
        #normalize(X, copy=False)
        #MinMaxScaler(copy=False)
        y = betas
        return original_ds_reshape, y

    def create_dataset_learnX(self, num_samples=1000,low=-100,high=100,n=1,d=1,num_updates=1,intercept=False,std_dev=1.0):
        Xtrain,ytrain = self.create_dataset_learnX_helper(num_samples,low,high,n,d,num_updates,intercept,std_dev)
        Xtest,ytest = self.create_dataset_learnX_helper((int)(.2*num_samples),low,high,n,d,num_updates,intercept, std_dev)
        dataset = {"Xtrain": Xtrain, "ytrain": ytrain,
                "Xtest": Xtest, "ytest": ytest}
        return dataset

    # what if we add the same updates to every point. Would that be totally off base?
    def create_dataset_learnX_helper(self, num_samples, low, high, n, d, num_updates, intercept, std_dev):
        original_ds = np.zeros((num_samples, n, d+1))
        for i in range(num_samples):
            r = make_regression(n_samples=n, n_features=d, n_informative=d, coef=True, noise=std_dev, n_targets=1)
            X, y = r[0], r[1]
            original_ds[i] = np.insert(X, d, y, axis=1)

        updates = np.random.randint(low,high+1,size=(num_samples,num_updates,d+1))
        betas = np.zeros((num_samples,num_updates+1,d))
        lm = LinearRegression(fit_intercept=intercept)
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
    '''
    def create_dataset_learnX_helper(self, num_samples,low,high,n,d,num_updates,intercept):
        #assumes n = 1 for original dataset
        #probably move to keras/tf backend as n grows
        #flatten array for now but maybe keep structure (similar to cnns)
        original_ds = np.random.randint(low,high+1,size=(num_samples,n,d+1))
        updates = np.random.randint(low,high+1,size=(num_samples,num_updates,d+1))
        betas = np.zeros((num_samples,num_updates+1,d))
        lm = LinearRegression(fit_intercept=intercept)
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
    '''

    def create_dataset_learnBeta(self, num_samples=1000, low=-100, high=100, n=1, d=1, num_updates=1, intercept=False):
        ods, Xtrain, ytrain = self.create_dataset_learnBeta_helper(num_samples,low,high,n,d,num_updates,intercept)
        print(Xtrain.shape)
        print(ytrain.shape)
        ods, Xtest,ytest = self.create_dataset_learnBeta_helper((int)(.2*num_samples),low,high,n,d,num_updates,intercept)
        dataset = {"Xtrain": Xtrain, "ytrain": ytrain,
                "Xtest": Xtest, "ytest": ytest}
        print(pd.DataFrame(ods).head())
        return dataset

    '''
    def create_dataset_learnBeta_helper(self, num_samples,low,high,n,d,num_updates,intercept):
        original_ds = np.random.randint(low,high+1,size=(num_samples,n,d+1))
        updates = np.random.randint(low,high+1,size=(num_samples,num_updates,d+1))
        betas = np.zeros((num_samples, num_updates, d))
        y = np.zeros((num_samples, d))
        lm = LinearRegression(fit_intercept=intercept)
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
    '''

    # these still need to be worked on
    def create_dataset_learnN(self, num_samples=1000, d=1, num_updates=1, save=False):
        Xtrain,ytrain = self.learnN(num_samples,d,num_updates)
        Xtest,ytest = self.learnN((int)(.2*num_samples),d,num_updates)
        dataset = {"Xtrain": Xtrain, "ytrain": ytrain,
                "Xtest": Xtest, "ytest": ytest}
        if save:
            for data in dataset:
                df = pd.DataFrame(dataset[data])
                df.to_csv(data+".csv", index=None, header=None)
        return dataset

    def learnN(self, num_samples, d, num_updates):
        ys = [0] * num_samples
        betas = np.zeros((num_samples, num_updates+1, d))
        updates = np.zeros((num_samples, num_updates, d+1))
        lm = LinearRegression(fit_intercept=False)
        for i in range(num_samples):
            n = np.random.randint(2,1000)
            ys[i] = n
            X,y = make_regression(n_samples=n, n_features=d, n_informative=d, bias=False)
            lm.fit(X,y)
            betas[i,0] = lm.coef_
            for j in range(num_updates):
                updatex, updatey = make_regression(n_samples=1, n_features=d, n_informative=d)
                updates[i,j] = np.insert(updatex, d, updatey, axis=1)
                X = np.append(X, updatex, axis=0)
                y = np.append(y, [updatey], axis=0)
                lm.fit(X,y)
                betas[i,j+1] = lm.coef_
        updates_reshape = np.reshape(updates, (num_samples, num_updates*(d+1)))
        X = np.append(updates_reshape, betas.reshape(num_samples, d*(num_updates+1)), axis=1)
        return X,np.asarray(ys)


    '''
    def write_dataset(self, paths, data, add=""):
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
    '''
