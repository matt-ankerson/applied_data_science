# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 08:39:01 2015

@author: matt
"""

# We're working with a reduced dataset containing just three features.
# (Voice feature 1, voice feature 2, and a clinition score)

import numpy as np
from sklearn.utils import shuffle
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model

data = np.loadtxt("data/artificial_lin.txt")

# we want to work with 2d data, so we select two attributes of 
# the possible three.

X = data[:, :-1]
Y = data[:, -1]     # the target vector
print(X[:10, :])
print(Y[:10])

# shuffle the examples
X, Y= shuffle(X, Y, random_state=1)
print('X Shape: ', X.shape)
print('Y Shape: ', Y.shape)

# split the data into train and test sets
train_set_size = X.shape[0] / 2
print('Train set size: ', train_set_size)
X_train = X[:train_set_size, :] # select first train_set_size rows
X_test= X[train_set_size:, :] # select from row train_set_size until end
print('X train shape: ', X_train.shape)
print('X test shape: ', X_test.shape)

# split the targets into train and test in a similar fashion
Y_train = Y[:train_set_size]    # selectd first 15 rows for train set
Y_test = Y[train_set_size:] # select from row 250 until end for test set
print('Y train shape: ', Y_train.shape)
print('Y test shape: ', Y_test.shape)

# let's plot the data in 3D.
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(X_train[:500, 0], X_train[:500, 1], Y_train[:500])
ax.view_init(6,-20)
plt.show()

# Linear Regression
regr = linear_model.LinearRegression()

# fit the model using the training set
regr.fit(X_train, Y_train)

# how can we evaluate our results?
# this score reaches it's maximum value of 1 when the model perfectly
# predicts all the test target values.
print(regr.score(X_train, Y_train))

# This means that our model explains almost 82% of the variance
# contained in the training data
# We have two coefficients and one intercept
print('Coefficient: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# Now we calculate the mean square error 
# This is the sum of all differences.
print("Training Error: ", np.mean((regr.predict(X_train) - Y_train) ** 2))
print("Training Error: ", np.mean((regr.predict(X_test) - Y_test) ** 2))

# Plotting data and linear model.
fig = plt.figure()
ax = Axes3D(fig)

# plots 3D points, 500 is number of points which are visualised.
ax.scatter3D(X_train[:500, 0], X_train[:500, 1], Y_train[:500])

# here we create the plane which we want to plot.
range_x = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), num=10)
range_y = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), num=10)
xx, yy = np.meshgrid(range_x, range_y)
zz = np.vstack([xx.ravel(), yy.ravel()]).T
pred = regr.predict(zz)
pred = pred.reshape(10, 10)

ax.plot_surface(xx, yy, pred, alpha=.3) # plots the plane
ax.view_init(6,-20)
plt.show()

# now we plot the test data and plane in a similar way

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter3D(X_test[:500, 0], X_test[:500, 1], Y_test[:500])

# here we create the plane which we want to plot.
range_x = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), num=10)
range_y = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), num=10)
xx, yy = np.meshgrid(range_x, range_y)
zz = np.vstack([xx.ravel(), yy.ravel()]).T
pred = regr.predict(zz)
pred = pred.reshape(10, 10)

ax.plot_surface(xx, yy, pred, alpha=.3) # plots the plane
ax.view_init(6,-20)
plt.show()


