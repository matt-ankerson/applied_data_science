import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as datasets
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
plt.jet()   # set the colour map
# Clustering ins the process o partitioning a group of data points into a
# smaller number of clusters.
# Let's start by generating some 2D data.
x, y = datasets.make_blobs(centers=4, cluster_std=0.5, random_state=0)
# Plot the data
plt.scatter(x[:, 0], x[:, 1])
plt.show()
# The data appears to contain four diffrent types of data point. Infact, this
# is how the data was generated, add colour to visualise these groups.
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()
# Once again, under normal circumstances you do not know the information in the
# y category of each instance in the dataset. Unsupervised clustering is all
# bout trying to estimate y from the data alone. This is what the k means
# algorithm does
kmeans = KMeans(n_clusters=4, random_state=8)   # we guess there are 4 groups
y_hat = kmeans.fit(x).labels_   # y_hat containes the estimated group
# belonging to each data point.
# Now the label assignments should be quite similar to y, up to a different
# ordering for the colours.
plt.scatter(x[:, 0], x[:, 1], c=y_hat)
plt.show()
# Sometimes in a given problem you are intersted in the assignment of each
# data point to a centroid. Other times you're interested in the centroids
# themselves. In the next figure we plot the centroid of each group as a big
# dot colour coded as the group they represent.
plt.scatter(x[:, 0], x[:, 1], c=y_hat, alpha=0.4)
mu = kmeans.cluster_centers_
plt.scatter(mu[:, 0], mu[:, 1], s=100, c=np.unique(y_hat))
print mu
plt.show()
# Let's increase the std deviation of our clusters and see the results.
x, y = datasets.make_blobs(centers=4, cluster_std=0.9, random_state=0)
kmeans = KMeans(n_clusters=4, random_state=8)   # we guess there are 4 groups
y_hat = kmeans.fit(x).labels_   # y_hat containes the estimated group
plt.scatter(x[:, 0], x[:, 1], c=y_hat)
plt.show()
# Let's give an incorrect value for the number or clusters and plot the results.
x, y = datasets.make_blobs(centers=4, cluster_std=0.5, random_state=0)
kmeans = KMeans(n_clusters=5, random_state=8)   # we guess there are 4 groups
y_hat = kmeans.fit(x).labels_   # y_hat containes the estimated group
plt.scatter(x[:, 0], x[:, 1], c=y_hat)
plt.show()
# Change the number of clusters generated.
x, y = datasets.make_blobs(centers=5, cluster_std=0.5, random_state=0)
kmeans = KMeans(n_clusters=4, random_state=8)   # we guess there are 4 groups
y_hat = kmeans.fit(x).labels_   # y_hat containes the estimated group
plt.scatter(x[:, 0], x[:, 1], c=y_hat)
plt.show()
# Now we will use the k-means algorithm on the classical MNIST dataset.
# Fetch from internet:
x_digits, _, _, y_digits = fetch_mldata('MNIST Original').values()
x_digits, y_digits = shuffle(x_digits, y_digits)
x_digits = x_digits[-5000]  # take the last instances, to shorted runtime.
# Let's have a look at some of the instances in the dataset we just loaded.
plt.rc('image', cmap='binary')
for i in xrange(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_digits[i].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())
plt.tight_layout()
