# Author: Matt Ankerson
# Date: 26 September 2015

# Validation and Model Selection
# This script explores the tradeoff between bias and variance,
# and why there is a tradeoff.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

# Generate an un-balanced 2D dataset
np.random.seed(0)
x = np.vstack([np.random.normal(0, 1, (950, 2)),
               np.random.normal(-1.8,  0.8, (50, 2))])
y = np.hstack([np.zeros(950), np.ones(50)])

plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='none',
            cmap=plt.cm.Accent)
plt.show()

# Cross Validation - the simplest cross validation involves running two
# where you split the data in two parts, training on both sets separately.
x1, x2, y1, y2 = cross_validation.train_test_split(x, y, test_size=0.5)
print x1.shape
print x2.shape

y2_pred = LogisticRegression().fit(x1, y1).predict(x2)
y1_pred = LogisticRegression().fit(x2, y2).predict(x1)

print np.mean([metrics.precision_score(y1, y1_pred),
              metrics.precision_score(y2, y2_pred)])

# This is known as two fold cross validation.
print cross_val_score(LogisticRegression(), x, y, cv=5, scoring='precision')

# Overfitting, underfitting and model selection.
# If our estimator is underperforming what do we do?
# - use a simpler or more colpex model?
# - add more features?
# - add more training samples?


# Illustration of the Bias/Variance tradeoff:
def test_func(x, err=0.5):
    y = 10 - 1. / (x + 0.1)
    if err > 0:
        y = np.random.normal(y, err)
    return y


def make_data(n=40, error=1.0, random_seed=1):
    # randomly sample the data
    np.random.seed(1)
    x = np.random.random(n)[:, np.newaxis]
    y = test_func(x.ravel(), error)
    return x, y

# Let's plot the data
x, y = make_data(40, error=1)
plt.scatter(x.ravel(), y)
plt.show()

# Compute a fit with linear regression.
x_test = np.linspace(-0.1, 1.1, 500)[:, None]
model = LinearRegression()
model.fit(x, y)
y_test = model.predict(x_test)

plt.scatter(x.ravel(), y)
plt.plot(x_test.ravel(), y_test)
plt.show()
print 'mean square error: ', metrics.mean_squared_error(model.predict(x), y)

# The linear model is clearly not a good choice for this data, it is what
# we'd call biased, or that it underfits the data.
# Let's try a more complicated model. (PolynomialRegression)


class PolynomialRegression(LinearRegression):
    '''Simple polynomial regression to 1D data'''
    def __init__(self, degree=1, **kwargs):
        self.degree = degree
        LinearRegression.__init__(self, **kwargs)

    def fit(self, x, y):
        if x.shape[1] != 1:
            raise ValueError('Only 1D data valid here.')
        xp = x ** (1 + np.arange(self.degree))
        return LinearRegression.fit(self, xp, y)

    def predict(self, x):
        xp = x ** (1 + np.arange(self.degree))
        return LinearRegression.predict(self, xp)

model = PolynomialRegression(degree=2)
model.fit(x, y)
y_test = model.predict(x_test)

plt.scatter(x.ravel(), y)
plt.plot(x_test.ravel(), y_test)
plt.show()
print 'mean squared error: ', metrics.mean_squared_error(model.predict(x), y)

# What happens when we use an even higher degree polynomial?
model = PolynomialRegression(degree=30)
model.fit(x, y)
y_test = model.predict(x_test)

plt.scatter(x.ravel(), y)
plt.plot(x_test.ravel(), y_test)
plt.ylim(-4, 14)
plt.show()
print 'mean squared error: ', metrics.mean_squared_error(model.predict(x), y)

# The mean squared error has been significantly reduced, but now we have a
# high variance model that captures the noise in the data.
# Clearly, computing the error on the training data is not enough.

# Plot the training and test error on the data in x as we increse the degrees.
degrees = np.arange(1, 30)

x, y = make_data(100, error=1.0)
x_train, x_test, y_train, y_test = \
    cross_validation.train_test_split(x, y, test_size=0.3)
training_error = []
test_error = []
mse = metrics.mean_squared_error

for d in degrees:
    model = PolynomialRegression(d).fit(x_train, y_train)
    training_error.append(mse(model.predict(x_train), y_train))
    test_error.append(mse(model.predict(x_test), y_test))

# Note that the test error can also be computed via cross validation.
plt.plot(degrees, training_error, label='training')
plt.plot(degrees, test_error, label='test')
plt.legend()
plt.xlabel('degree')
plt.ylabel('MSE')
plt.show()

# This is a typical bias/variance plot

# Illustration of learning curves:
# The exact turning point of the tradeoff between bias and variance is highly
# dependent on the number of training points used.

x, y = make_data(200, error=1.0)
degree = 3
x_train, x_test, y_train, y_test = \
    cross_validation.train_test_split(x, y, test_size=0.3)
n_range = np.linspace(15, x_train.shape[0], 20).astype(int)


def plot_learning_curve(degree=3):
    training_error = []
    test_error = []
    mse = metrics.mean_squared_error
    for n in n_range:
        xn = x_train[:n]
        yn = y_train[:n]
        model = PolynomialRegression(degree).fit(xn, yn)
        training_error.append(mse(model.predict(xn), yn))
        test_error.append(mse(model.predict(x_test), y_test))
    plt.plot(n_range, training_error, label='training')
    plt.plot(n_range, test_error, label='test')
    plt.plot(n_range, np.ones_like(n_range), ':k')
    plt.legend()
    plt.title('degree = {0}'.format(degree))
    plt.xlabel('num. training points')
    plt.ylabel('MSE')
    plt.show()

plot_learning_curve(3)

# This shows a typical learning curve. For very few training points there
# is a large separation between the training and test error, which indicates
# over fitting. For a larger set of training points, the training and
# test errors converge, which indicates potential underfitting.

# Adding more samples won't bring the MSE down in this case, let's try
# a different degree.
plot_learning_curve(2)

# What about the other extreme?
plot_learning_curve(5)

# For d=5 it is converged, to a better value than d=3.
# Thus you can bring the curves closer together by adding more points.
# You can bring the convergence level down only by adding complexity
# to the model.
