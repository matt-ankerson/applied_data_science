import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn import linear_model

# Author: Matt Ankerson
# Date: 19 October 2015
# This script contains answers to questions outlined in assignment 2.

# -----------------------------------------------------------------------------
# Exercise 1 - K Means
# Given the following dataset which is unlabeled:
# Use the K-means algorithm included in scikit-learn using 3 custers to fit the
# data and plot the data colour coded for the estimated cluster each data point
# belongs to.
x, y = make_blobs(n_samples=300, centers=3, random_state=0, cluster_std=0.60)
kmeans = KMeans(n_clusters=3, random_state=8)
y_hat = kmeans.fit(x).labels_   # y_hat contains the estimated group assignments
# Plot the data, colour coded according to our k-means preditions
plt.scatter(x[:, 0], x[:, 1], c=y_hat)
plt.show()

# -----------------------------------------------------------------------------
# Exercise 2 - K-Nearest Neighbors
# Let's move to a classification task. This is a dataset with 50 instances
# of customer behaviour in a video store. The data contains 8 columns:
# 1. customerID
# 2. gender
# 3. income
# 4. age
# 5. rentals
# 6. average per visit
# 7. genre
# 8. incidentals
# # Clean and arrange the data:
vs = np.genfromtxt("video_store_2.csv", delimiter=",", names=True,
                   dtype=(int, "|S1", float, int, int, float, "|S10", "|S3"))
vs_records = vs[['Gender', 'Income', 'Age', 'Rentals', 'Avg_Per_Visit',
                 'Genre']]
vs_names = vs_records.dtype.names
vs_dict = [dict(zip(vs_names, record)) for record in vs_records]
vs_vec = DictVectorizer()
x = vs_vec.fit_transform(vs_dict).toarray()
y = vs['Incidentals']
# We now move onto splitting the data into training and testing.
# (80% for training)
tpercent = 0.8
tsize = tpercent * len(x)
x_train = x[:tsize, :]
x_test = x[tsize:, :]
y_train = y[:tsize]
y_test = y[tsize:]
# Before we carry out classification on the training data we need to normalise
# the values in order to optimise the algorithm's performance. This process is
# called feature scaling.
min_max_scaler = preprocessing.MinMaxScaler()
x_train_norm = min_max_scaler.fit_transform(x_train)
x_test_norm = min_max_scaler.fit_transform(x_test)
# Now use the K nearest neighbors classifier to fit the data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_norm, y_train)
# Now we can carry out predictions with the test set.
y_pred_test = knn.predict(x_test_norm)
print('Accuracy of KNN classifier on test data: ' + str(knn.score(x_test_norm,
                                                                  y_test)))
# 0.5

# -----------------------------------------------------------------------------
# Exercise 3 - Error Analysis: Precision and Recall
# Print out a detailed analysis of precision versus recall for the KNN algorithm
# above.
# Generate predictions with KNN Classifier
y_pred = knn.predict(x_test_norm)
# Get a classification summary report:
print('KNN Classification Report:')
print metrics.classification_report(y_test, y_pred,
                                    target_names=['background', 'foreground'])
#             precision    recall  f1-score   support
#
#  background      0.50      0.60      0.55         5
#  foreground      0.50      0.40      0.44         5
#
#  avg / total     0.50      0.50      0.49        10

# -----------------------------------------------------------------------------
# Exercise 4 - Support Vector Machines
# Now use a support vector machine classifier with a no linear kernel to build
# a model using x_train_norm, y_train and to carry out prediction on
# x_test_norm. Print out a detailed analysis of precision vs recall.
svm_clf = SVC(kernel='rbf')
svm_clf.fit(x_train_norm, y_train)
y_pred = svm_clf.predict(x_test_norm)
# Get a classification summary report
print('SVM Classification Report:')
print metrics.classification_report(y_test, y_pred,
                                    target_names=['background', 'foreground'])
#             precision    recall  f1-score   support
#
#  background      0.75      0.60      0.67         5
#  foreground      0.67      0.80      0.73         5
#
#  avg / total     0.71      0.70      0.70        10

# -----------------------------------------------------------------------------
# Exercise 5 - Logistic Regression
# Now use a logistic regression classifier to build a model using x_train_norm,
# y_train to carry out prediction on x_test_norm. Print out a detailed analysis
# of precision vs recall.
logreg = LogisticRegression()
logreg.fit(x_train_norm, y_train)
y_pred = logreg.predict(x_test_norm)
# Get a classification summary report:
print('Logistic Regression Classification Report:')
print metrics.classification_report(y_test, y_pred,
                                    target_names=['background', 'foreground'])
#             precision    recall  f1-score   support
#
#  background      0.75      0.60      0.67         5
#  foreground      0.67      0.80      0.73         5
#
#  avg / total     0.71      0.70      0.70        10

# -----------------------------------------------------------------------------
# Exercise 6 - Principal Component Analysis
# Create 200 random 2D data points, sampled from a multinomial normal
# distribution.
cov = np.array([[2.9, -2.2], [-2.2, 6.5]])
x = np.random.multivariate_normal([1, 2], cov, size=200)
plt.figure(figsize=(4, 4))
plt.scatter(x[:, 0], x[:, 1])
plt.axis('equal')   # equal scaling on both axis
plt.show()
# Use PCA to find the two principal components of the data. Print out those
# two components and the percentage of variance explained by each.
pca = PCA(n_components=2)
pca.fit(x)
print('PCA Components:')
print(pca.components_)
print('Percentage variance explained by each one of the two components:')
print(pca.explained_variance_ratio_)
# PCA Components:
#    [[-0.48828435  0.87268459]
#     [ 0.87268459  0.48828435]]
# Percentage variance explained by each one of the two components:
#    [ 0.83848037  0.16151963]

# -----------------------------------------------------------------------------
# Exercise 7 - Artificial Neural Networks
# What is the output of this (logical OR) artificial neural network if the
# input vector is x = (1, 0) ?
# g(-10 + 20 + 0) = 1
# The answer is: 1

# -----------------------------------------------------------------------------
# Exercise 8 - Validation
# In your own words, explain what these two lines of code do.
# cv = cross_val_score(KNeighborsClassifier(1), X, y, cv=10)
# cv.mean()
# The first line computes a 10-fold cross validation on the data. The data
# is split into 'n' portions, in this case n=10. Each 'fold' is a process
# wherin 9 of the 10 portions are used for training, and the remaining 1 portion
# is used for testing. Each fold uses a different 1/10 slice for testing and
# subsequently a different 9/10 slice for training - yielding a collection of
# slightly differing accuracies. It is using the
# K-Nearest Neighbors classification algorithm to build the models.
# The second line takes an average of all accuracies now stored in 'cv'.
# The cross validation score is a means of estimating the accuracy of a
# classification or regression model.

# -----------------------------------------------------------------------------
# Exercise 9 - Regularization
# Now let's load the boston housing dataset.
# Examine the code below, explain why the coefficients (params) of the model
# "ridgeRegr" are smaller than those from model "regr".
boston = load_boston()
# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    train_test_split(boston.data,  boston.target, test_size=0.5, random_state=0)
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
print('Linear Regression Coef:')
print regr.coef_
print "-------------------"
ridgeRegr = linear_model.SGDRegressor(loss='squared_loss', penalty='l2')
ridgeRegr = linear_model.Ridge(alpha=10000)
ridgeRegr.fit(x_train, y_train)
print('Ridge Regression Coef:')
print ridgeRegr.coef_
# [ -1.22697052e-01   5.76835439e-02   7.42047961e-02   3.38949970e+00
#   -1.65391519e+01   3.57730248e+00  -2.98033902e-03  -1.55759520e+00
#    2.50098217e-01  -9.73850155e-03  -1.12527834e+00   6.85073312e-03
#   -5.92410460e-01]
# -------------------
# [ -8.13191022e-02   7.88595689e-02  -3.45432534e-02   8.42665464e-03
#    3.40347049e-05   6.72188436e-02   1.92473899e-02  -7.62272258e-02
#    7.11550968e-02  -1.25711659e-02  -1.30120768e-01   1.06545857e-02
#   -4.02238603e-01]
# Answer: The coefficients of the model 'ridgeRegr' are smaller than those from
# the model 'regr' because the regr model is overfitted. Overfitting occurs
# when a model matches the training set very well (too closely),
# but doesn't generalise on test data. (ie. it learns noise in the data, instead
# of just the trend or general curve.) Large coefficients are generally a sign
# of overfitting. 'ridgeRegr' uses regularisation to lower
# the coefficients. Regularisation is a means of constraining or
# reducing the size of the coefficients towards zero. (Though Ridge Regression
# ensures they never actually reach zero.) Ridge Regression adds a parameter
# 'alpha' to the loss function to enforce a penalty on the size of the
# coefficients. A larger alpha indicates a larger penalty.
