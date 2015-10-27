import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Principal Component Analysis.
# A powerful unsupervised method for dimensionality reduction in data.
# Lets visualise the concept with some sample data.
np.random.seed(1)
x = np.dot(np.random.random(size=(2, 2)), np.random.normal(size=(2, 200))).T
plt.plot(x[:, 0], x[:, 1], 'og')
plt.axis('equal')
plt.show()
# We begin by carrying out PCA on the artificial dataset created. The
# argument n_components indicating the number of components to extract,
# n_components plays the role of k.
pca = PCA(n_components=2)
pca.fit(x)
# The computations required to extract the components have been carried out.
# We can now print out the percentage of variance explained by each one of the
# two components.
print(pca.explained_variance_ratio_)
# You can see that the 1st component explained almost 97% of variance in the
# data. The 2nd component on the other hand explains less than 2% of the
# variance in the data. This means that we can condense most of the information
# in the original data (using two features) into a projection that uses only
# one feature and still captures 97% of the information contained in the
# original data.
# We can print the coordinates of those two vectors. (The two principal
# components of the data)
print(pca.components_)
# To visualize how these two components relate to the variability of the data,
# lets plot the vector components on top of the scatter plot data.
plt.plot(x[:, 0], x[:, 1], 'og', alpha=0.3)
plt.axis('equal')
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    plt.plot([0, v[0]], [0, v[1]], 'k', lw=3)
plt.show()
# It is important to note that one direction is very important, while the other
# is not. This shows us that the second principal could be completely ignored,
# without much loss of information.
clf = PCA(0.97)
x_trans = clf.fit_transform(x)
print(x.shape)
print(x_trans.shape)
# By specifying that we want to throw away 3% of the variance, the data is now
# compressed by a factor of 50%. Let's visualize how the principal component
# analysis is just a projection of data points onto the principal component
# vector (axis) that still manages to capture most of the variance in the data.
x_new = clf.inverse_transform(x_trans)
plt.plot(x[:, 0], x[: 1], 'og', alpha=0.2)
plt.plot(x_new[:, 0], x_new[:, 1], 'og', alpha=0.8)
plt.axis('equal')
plt.show()
