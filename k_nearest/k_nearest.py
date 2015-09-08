# Author: Matt Ankerson
# Date: 8 September 2015

# K NEarest Neighbors.

import numpy as np
from sklearn.datasets import make_circles
import pylab as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Create a new dataset of circle distributions
x, y = make_circles(noise=.1, factor=.5)
print('x shape: ' + str(x.shape))
print('Unique labels: ' + str(np.unique(y)))

# Now let's plot the data.
plt.prism()     # this sets a nice colour map.
plt.scatter(x[:, 0], x[:, 1], c=y)
#plt.show()

# Split the data into training and test sets.
x_train = x[:50]
y_train = y[:50]
x_test = x[50:]
y_test = y[50:]

# Now we fit a logistic regression model to the training data.
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

# Evauate the logistic regression by plotting the decision surface and predictions on 
# the test data.
def plot_decision_boundary(clf, X):
    w = clf.coef_.ravel()
    a = -w[0] / w[1]
    xx = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]))
    yy = a * xx - clf.intercept_ / w[1]
    plt.plot(xx, yy)
    plt.xticks(())
    plt.yticks(())
plt.prism()
y_pred_test = logreg.predict(x_test)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred_test, marker='^')
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plot_decision_boundary(logreg, x)
plt.xlim(-1.5, 1.5)
plt.show()
print('Accuracy of logistic regression on test set: ' + str(logreg.score(x_test, y_test)))
# In short, the accuracy sucks. This is a classical illustration of a non-linearly seperable problem.
# So let's look at how k-nearest neighbor performs here...

knn = KNeighborsClassifier(n_neighbors=5)   # we specify that it should use 5 neighbors.

# Fit the KNN model to the training data.
knn.fit(x_train, y_train)

# Now we can carry out predictions using the test set.
y_pred_test = knn.predict(x_test)

# The following visualisation shows the test data as triangles.
plt.prism()
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.scatter(x_test[:, 0], x_test[:, 1], c = y_pred_test, marker='^')
plt.scatter(x_train[:, 0], x_train[:, 1], c = y_train)
plt.show()
print('Accuracy of KNN test set: ' + str(knn.score(x_test, y_test)))
# This is a much higher degreee of accuracy.























