import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

from CreateDataSet import create_dataset

dataset = create_dataset(100000,0,10,1,1,1,save=True)
print("done getting data")
Xtrain, ytrain, Xtest, ytest = dataset["Xtrain"], dataset["ytrain"], dataset["Xtest"], dataset["ytest"]

params = [
    {'hidden_layer_sizes': [(100,), (10,10), (10,10,10), (100,100)], 'activation': ['identity', 'tanh', 'logistic', 'relu']}
]

grid_search = GridSearchCV(MLPRegressor(max_iter=400), params, scoring='neg_mean_squared_error', cv=5, n_jobs=7, verbose=7)
grid_search.fit(Xtrain, ytrain)

df = pd.DataFrame.from_dict(grid_search.cv_results_)
df.to_csv('gridsearchresults.csv', index=False)

print(grid_search.best_estimator_.get_params())

grid_search.score(Xtest,ytest)