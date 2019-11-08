import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from CreateDataSet import read_dataset
from CreateDataSet import write_dataset
from sklearn.preprocessing import MinMaxScaler

def create_poly_features(X):
    #obviously make this more generalizable
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X = np.append(X, poly.fit_transform(X[:,0:2]), 1)[:,2:]
    scaler = MinMaxScaler(copy=True)
    scaler.fit(X)
    X = scaler.transform(X)
    return X



datasets_names = ['Xtrainextended.csv', 'Xtestextended.csv']
datasets = read_dataset(datasets_names)

for i in range(len(datasets)):
    X = create_poly_features(datasets[i])
    write_dataset(datasets_names[i], X, 'poly')


