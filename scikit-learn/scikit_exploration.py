# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 09:14:17 2015

@author: matt
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab as plt_2
from sklearn.datasets import load_digits
from sklearn.datasets import make_s_curve
from mpl_toolkits.mplot3d import Axes3D

print("Iris Dataset: ")

# load the iris dataset
from sklearn.datasets import load_iris

# this is a Bunch object, which is basically an enhanced dictionary
iris = load_iris()

n_samples, n_features = iris.data.shape

# print the first instance in the data set and the correspondong target
print(iris.data[0])
print(iris.target[0])

# print the entire target/label vector
print(iris.target)

# print the names associated with each target vector
print(iris.target_names)

# the iris dataset is four dimensional, but we can visualise two on a scatter plot
def plot_iris_projection(x_index, y_index):
    # this formatter will label the colour bar with the target names
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
    
    plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c = iris.target)
    plt.colorbar(ticks=[0, 1, 2], format = formatter)
    plt.xlabel(iris.feature_names[x_index])
    plt.ylabel(iris.feature_names[y_index])
    
plot_iris_projection(2, 3)
    
print("Another Dataset - Digits: ")  

digits = load_digits()
print(digits.keys())

n_samples, n_features = digits.data.shape
print(n_samples, n_features)

print(digits.data[0])
# the target here is just the digit represented by the data
print(digits.target)
# we've got two versions of the data array, data and images
print(digits.data.shape)
print(digits.images.shape)
# we see that the two versions differ only in shape
print(digits.data.__array_interface__['data'])
print(digits.images.__array_interface__['data'])

# we can visualise this.
fig = plt_2.figure(figsize = (6,6)) # figure size in inches
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

# plot the digits, each is 8x8 pixels 
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt_2.cm.binary, interpolation = 'nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))
# each feature is a real-valued quantity indicating the darkness of a particular
# pixel in an 8x8 image

print("Non linear dataset - The S-Curve: ")
data, colors = make_s_curve(n_samples=1000)
print(data.shape)
print(colors.shape)
# let's visualise this
ax = plt_2.axes(projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = colors)
ax.view_init(10, -60)