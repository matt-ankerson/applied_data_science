# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 09:08:00 2015

@author: matt
"""

from sklearn.datasets import load_boston
import numpy as np
from sklearn.utils import shuffle
from sklearn import linear_model

boston = load_boston()
print("Keys: ", boston.keys())
print("Shape: ", boston.data.shape)
print("Fetaure names: ", boston.feature_names)

data = boston.data
target = boston.target

# split the data into two halves and carry out linear regression 
# on the training data

X = data[:]        # copy the data, these are the features
Y = target[:]         # copy the data, this is our target

# shuffle the data and target
X, Y = shuffle(X, Y, random_state=1)

# split the data into training sets and test sets
train_set_size = X.shape[0] / 2
print('Train set size: ', train_set_size)
X_train = X[:train_set_size, :] # select first train_set_size rows
X_test= X[train_set_size:, :] # select from row train_set_size until end
print('X train shape: ', X_train.shape)
print('X test shape: ', X_test.shape)

# split the targets into train and test in a similar fashion
Y_train = Y[:train_set_size]    # selectd first train_set_size rows for train set
Y_test = Y[train_set_size:] # select from row train_set_size until end for test set
print('Y train shape: ', Y_train.shape)
print('Y test shape: ', Y_test.shape)

# Linear Regression
regr = linear_model.LinearRegression()

# fit the model using the training set
regr.fit(X_train, Y_train)

# how can we evaluate our results?
# this score reaches it's maximum value of 1 when the model perfectly
# predicts all the test target values.
print('Regression Score: ', regr.score(X_train, Y_train))

# This means that our model explains almost 78% of the variance
# contained in the training data
# We have 13 coefficients and one intercept
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# Now we calculate the mean square error 
# This is the sum of all differences.
print("Training Error: ", np.mean((regr.predict(X_train) - Y_train) ** 2))
print("Testing Error: ", np.mean((regr.predict(X_test) - Y_test) ** 2))