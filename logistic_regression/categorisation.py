# Author: Matt Ankerson
# Date: 26 August 2015

# Logistic Regression.
# (where the dependent variable is categorical)

from sklearn.datasets import make_blobs
import pylab as plt
from sklearn.linear_model import LogisticRegression
import numpy as np

# Generate dataset consisting of two Gaussian clusters
X, Y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=0.5)

print("X shape: " + str(X.shape))
print("Y: " + str(Y))

#plt.prism()
#plt.scatter(X[:, 0], X[:, 1], c=Y)
#plt.show()

# Carry out logistic regression by constructing a classification object
logreg = LogisticRegression()

# Split the data into a training set and a test set.
X_train = X[:50]
Y_train = Y[:50]
X_test = X[50:]
Y_test = Y[50:]

# The test points are plotted as white triangles
#plt.prism()
#plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train)
#plt.scatter(X_test[:, 0], X_test[:, 1], c='white', marker='^')

logreg.fit(X_train, Y_train)

print(str(logreg.intercept_))    # theta_0
print(str(logreg.coef_))         # theta_1 and theta_2

def plot_decision_boundary(clf, X):
    w = clf.coef_.ravel()
    a = -w[0] / w[1]
    xx = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]))
    yy = a * xx - clf.intercept_ / w[1]
    plt.plot(xx, yy)
    plt.xticks(())
    plt.yticks(())
    
y_pred_train = logreg.predict(X_train)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train)
plot_decision_boundary(logreg, X)
    
