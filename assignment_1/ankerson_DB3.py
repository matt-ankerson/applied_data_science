# Assignment 1 - Linear Regression

# Author: Matthew Ankerson
# Date: 28 August 2015

# This assignment works with advertising data for sales of a certain product. 
# There are three features:
# - Dollars spent on TV advertising.
# - Dollars spent on Radio advertising.
# - Dollars spent on Newspaper advertising.
# The target vector is the sales revenue generated for the certain product.

import numpy as np
import pandas as pd         # Like numpy, gives us useful operations.
from sklearn.utils import shuffle
from sklearn import linear_model
import pylab as plt

data = pd.read_csv('Advertising.csv', index_col=0)  # Read in data from .csv

print(data.head())          # View the top 5 instances.

#      TV  Radio  Newspaper  Sales
# 1  230.1   37.8       69.2   22.1
# 2   44.5   39.3       45.1   10.4
# 3   17.2   45.9       69.3    9.3
# 4  151.5   41.3       58.5   18.5
# 5  180.8   10.8       58.4   12.9

# Extract the feature matrix (X).
feature_cols = [ 'TV', 'Radio', 'Newspaper' ]
x = np.array(data[feature_cols])

# Extract the target vector (Y).
y = np.array(data.Sales)

# Inspect the shape of our data
print('x shape: ' + str(x.shape))
print('y shape: ' + str(y.shape))

# Shuffle the dataset
x, y = shuffle(x, y, random_state=1)

# Get the sales data
sales = y[:]

# ---------------------------------------------------------
# Exercise 1 - Plot expenditure on TV ads against Sales.

# Get the tv expenditure exclusively.
tv_expenditure = x[:, 0]
plt.scatter(tv_expenditure, sales)
plt.xlabel('TV Expenditure $ (thousands)')
plt.ylabel('Sales $ (thousands)')
plt.show()

# ---------------------------------------------------------
# Exercise 2 - Plot expenditure on Radio ads against Sales.

# Get the radio expenditure exclusively.
radio_expenditure = x[:, 1]
plt.scatter(radio_expenditure, sales)
plt.xlabel('Radio Expenditure $ (thousands)')
plt.ylabel('Sales $ (thousands)')
plt.show()

# ---------------------------------------------------------
# Exercise 3 - Plot expenditure on Newspaper ads against Sales.

# Get the newspaper expenditure exclusively.
newspaper_expenditure = x[:, 2]
plt.scatter(newspaper_expenditure, sales)
plt.xlabel('Newspaper Expenditure $ (thousands)')
plt.ylabel('Sales $ (thousands)')
plt.show()

# ---------------------------------------------------------
# Exercise 4 - Are the TV and radio features positively correlated with sales?
# Yes.
# Correlation is positive when the values increase together

# ---------------------------------------------------------
# Evercise 5 - Is the Newspaper features positively or negatively correlated with sales?
# There is a very weak positive correlation.

# ---------------------------------------------------------
# Exercise 6 - Split the data in 2 halves: training set and test set

train_set_size = 100
x_train = x[:train_set_size]
x_test = x[train_set_size:]
print('x_train shape: ' + str(x_train.shape))
print('x test shape: ' + str(x_test.shape))

y_train = y[:train_set_size]
y_test = y[train_set_size:]
print('y train shape: ' + str(y_train.shape))
print('y test shape: ' + str(y_test.shape))

# ---------------------------------------------------------
# Exercise 7 - Fit a multivariate linear regression model on the training data using all the features available

# Create a linear regression object 
regr = linear_model.LinearRegression()

# Fit the model using all of the training set
regr.fit(x_train, y_train)

# ---------------------------------------------------------
# Exercise 8 - What are the intercepts and coefficients of the model?

# We have 3 coefficients and one intercept.
print('Coefficients: ' + str(regr.coef_))
print('Intercept: ' +  str(regr.intercept_))

# Coefficients: [ 0.04684413  0.19437815 -0.00619286]
# Intercept: 2.79401745058

# ---------------------------------------------------------
# Exercise 9 - What is the R squared score (i.e. the coefficient of determination that measures 
# the proportion of the outcomes variation explained by the model) for the training data? and for the test data?
print('Training R squared score: ' + str(regr.score(x_train, y_train)))
print('Testing R squared score:  ' + str(regr.score(x_test, y_test)))

# Training R squared score: 0.916518861995
# Testing R squared score:  0.876145253062

# This means that our model explains about 91% of the variance contained in the training data,
# and about 87% of the variance in the testing data.

# ---------------------------------------------------------
# Exercise 10 - Let's say you are trying to decide how much to spend on a different advertisement media before
# entering a new market. Your budget is 300 (Keep in mind that means $300,000). Remember,
# that with the model you have created, you can predict how much sales you're likely to achieve.

# The question is, if your goal is to maximize sales, should you spend (in thousands of dollars):
# - 200 on TV, 30 on Radio and 70 on Newspaper OR 
# - 200 on TV, 90 on Radio and 10 in Newspaper OR 
# - 100 on TV, 100 on Radio and 100 in Newspaper? 
# What predictions do you get from the model for each case?

prediction_1 = [200, 30, 70]
prediction_2 = [200, 90, 10]
prediction_3 = [100, 100, 100]

print('Prediction 1: ' + str(regr.predict(prediction_1)))
print('Prediction 2: ' + str(regr.predict(prediction_2)))
print('Prediction 3: ' + str(regr.predict(prediction_3)))

# Predictions:
# Prediction 1: [ 17.56068773]
# Prediction 2: [ 29.59494817]
# Prediction 3: [ 26.29695901]

# Based on these predictions I would opt for the second strategy, as this yeilds the highest sales
# results. (approx $29,594.95)