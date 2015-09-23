import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Author: Matt Ankerson
# Date: 23 September 2015

# How can we evaluate the performance of a model?

# Generate an unbalanced 2-D dataset.
np.random.seed(0)
x = np.vstack([np.random.normal(0, 1, (950, 2)), np.random.normal(-1.8, 0.8, (50, 2))])
y = np.hstack([np.zeros(950), np.ones(50)])

plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='none', cmap=plt.cm.Accent)
plt.show()

# The problem with simple validation:
# We might not care how well we can classify the background (non-cancer), but we might instead 
# be concerned with suceessfully pulling out an uncontaminated set of foreground (cancer) sources.
# We get this by computing the precision, recall and F1 score.

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
clf = LogisticRegression()
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print 'Accuracy:  ', metrics.accuracy_score(y_test, y_pred)
print 'Precision: ', metrics.precision_score(y_test, y_pred)
print 'Recall:    ', metrics.recall_score(y_test, y_pred)
print 'F1 Score:  ', metrics.f1_score(y_test, y_pred)

# Accuracy:   0.968
# Precision:  0.833333333333
# Recall:     0.625
# F1 Score:   0.714285714286

# What do these mean?
# These are ways of taking into account not just the classification results, but the results 
# relative to the true category.

# accuracy = correct labels / total samples
# precision = true positives / (true positives + false positives)
# recall = true positives / (true positives + false negatives)
# F1 = 2 * ((precision * recall) / (precision + recall))

# All the above scores range from 0 to 1, 1 being optimal.

# A summary can be obtained with the following command:
print metrics.classification_report(y_test, y_pred, target_names=['background', 'foreground'])

#              precision    recall  f1-score   support

#  background       0.97      0.99      0.98       234
#  foreground       0.83      0.62      0.71        16

# avg / total       0.97      0.97      0.97       250

# This tells us that, though the overall correct classification rate is 97%, we only correctly identify
# 62% of the desired samples, and those that we label as positives are only 83% correct! Theoretically this means
# that we were only able to detect 62% of the people in the population who actually had cancer, that is, we 
# missed 38% of true cancer cases and diagnosed them as not having cancer. Among the people that the algorithm
# classifies as having cancer, ony 83% of them actually do have cancer.