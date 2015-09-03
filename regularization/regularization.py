# Author: Matt Ankerson
# Date: 3 September

# This is a curve fitting regression problem, we will generate a 1d dataset with some noise.

import numpy as np
import scipy.stats as st
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

# We need to define a deterministic function underlying our generative model.
# The function is f(x) = e^3x
f = lambda x: np.exp(3 * x)

# Next generate 200 values along the x axis in the interval [0,2] and the corresponding
# value of f in order to make a nice plot.
x_tr = np.linspace(0., 2, 200)
y_tr = f(x_tr)

# Now generate data points within [0,1]. We use function f and add some gaussian noise.
x = np.array([0, .1, .2, .5, .8, .9, 1])
y = f(x) + 2 * np.random.randn(len(x))  # y will be determined by the function f plus some random.

# Plot this:
plt.figure(figsize=(6, 3))
plt.plot(x_tr[:100], y_tr[:100], '--k')
plt.plot(x, y, 'ok', ms=10)
plt.show()

# Next we will use linear regression on just the black dots to try to approximate the hidden Law f of 
# our universe.

# Create the model
lr = lm.LinearRegression()
# Train the model on our training dataset.
lr.fit(x[:, np.newaxis], y)
# Predict the points with the trained model.
y_lr = lr.predict(x_tr[:, np.newaxis])

# Now we plot the result of the trained linear model. We obtain a regression line in green.
plt.figure(figsize=(6, 3))
plt.plot(x_tr, y_tr, '--k')     # plot dashed line in black.
plt.plot(x_tr, y_lr, 'g')       # plot in green
plt.plot(x, y, 'ok', ms = 10)
plt.xlim(0, 1)
plt.ylim(y.min() - 1, y.max() + 1)
plt.title('Linear Regression')
plt.show()

# Next we will try to fit a non-linear model to the data, a polynomial function.

lrp = lm.LinearRegression()
plt.figure(figsize=(6, 3))
plt.plot(x_tr, y_tr, '--k')

for deg, s in zip([2, 5], ['-', '.']):
    lrp.fit(np.vander(x, deg + 1), y)
    y_lrp = lrp.predict(np.vander(x_tr, deg + 1))
    plt.plot(x_tr, y_lrp, s, label='degree ' + str(deg))
    plt.legend(loc=2)
    plt.xlim(0, 1.4)
    plt.ylim(-10, 40)
    # Print the model's coefficients.
    print(' '.join(['%.2f' % c for c in lrp.coef_]))
plt.plot(x, y, 'ok', ms=10)
plt.title("Linear Regression")
plt.show()

# The green line is overfitted and does a poor job at predicting new data.

# We will now use a different learning model, called ridge regression. It prevents the 
# polynimials coefficients from exploding.

ridge = lm.Ridge(alpha = 0.1)
plt.figure(figsize=(6, 3))
plt.plot(x_tr, y_tr, '--k')

for deg, s in zip([2, 5], ['-', '.']):
    ridge.fit(np.vander(x, deg + 1), y)
    y_ridge = ridge.predict(np.vander(x_tr, deg + 1))
    plt.plot(x_tr, y_ridge, s, label='degree ' + str(deg))
    plt.legend(loc=2)
    plt.xlim(0, 1.5)
    plt.ylim(-5, 80)
    # Print the model's coefficients.
    print(' '.join(['%.2f' % c for c in ridge.coef_]))
plt.plot(x, y, 'ok', ms=10)
plt.title("Ridge Regression")
plt.show()