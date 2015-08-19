# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 09:21:51 2015

@author: matt
"""

# Linear Regression with one variable.
from sklearn import datasets
from sklearn import linear_model
from sklearn.utils import shuffle
import pylab as plt
import numpy as np

# load the diabetes dataset
diabetes = datasets.load_diabetes()

x = diabetes.data
y = diabetes.target
print(x.shape)
print(y.shape)

x, y = shuffle(x, y, random_state=1)

# use only one column from the data
x = x[:, 2:3]
print(x.shape)

# split the data into training/test sets
train_set_size = 250
x_train = x[:train_set_size]
x_test = x[train_set_size:]
print(x_train.shape)
print(x_test.shape)

# split the data into training/test sets
y_train = y[:train_set_size]
y_test = y[train_set_size:]
print(y_train.shape)
print(y_test.shape)

# now we can look at our training data, we see that the examples 
# have linear relation
plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test)
plt.xlabel('Data')
plt.ylabel('Target')

# plot feature 4, is there a linear relationship in the data?
# pull out the column:
# change line 24 to select column 4 **x = x[:, 4:5]

# create a linear regression object 
regr = linear_model.LinearRegression()

# now we are going to fit the model with the training set.
# this function carries out an estimation of the parameters of the lineal
# model (theta 0 and theta 1)
regr.fit(x_train, y_train)

# now we have the coefficient (theta 1) and the bias or intercept (theta 0)
print(regr.coef_)    # theta 1
print(regr.intercept_)   # theta 0

# now we calculate the mean square error on the test set
# (the distance from the model to the test set instances)
# mean square error
print("Training Error: ", np.mean((regr.predict(x_train) - y_train) ** 2))
print("Test     Error: ", np.mean((regr.predict(x_test) - y_test) ** 2))

# Plotting and linear model
#   now we want to plot the train data and targets.
plt.scatter(x_train, y_train, color='black')
# plots the linear model
plt.plot(x_train, regr.predict(x_train), color='blue', linewidth=3)
plt.xlabel('Data')
plt.ylabel('Target')

#   now we want to plot the test data and targets.
#plt.scatter(x_test, y_test, color='black')
# plots the linear model
#plt.plot(x_test, regr.predict(x_test), color='blue', linewidth=3)
#plt.xlabel('Data')
#plt.ylabel('Target')