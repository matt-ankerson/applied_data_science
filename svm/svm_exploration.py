import numpy as np
from mpl_toolkits import mplot3d
from sklearn.datasets.samples_generator import make_circles
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from scipy import stats
from sklearn.svm import SVC
# Use seaborn plotting defaults.
import seaborn as sns
sns.set()

# Support Vector Machines are a powerful supervised learning algorithm
# used for classification and regression. They are a discriminative classifier,
# meaning they draw a boundary between classes of data.
# Create a sample dataset:
x, y = make_blobs(n_samples=50, centers=2, random_state=0,
                  cluster_std=0.60)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='spring')
plt.show()

# There are many possible separations of this data.
xfit = np.linspace(-1, 3.5)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='spring')
for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')

plt.xlim(-1, 3.5)
plt.show()

# SVMs, maximising the margin.
xfit = np.linspace(-1, 3.5)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='spring')

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                     color='#AAAAAA', alpha=0.4)
plt.xlim(-1, 3.5)
plt.show()
# Note that if we want to maximise this width, the middle fit is the best.
# This is the intuition of support vector machines.
# Now we'll fit a svm classifier to these points.
clf = SVC(kernel='linear')
clf.fit(x, y)
# To visualise what's going on here, let's create a function that will plot
# SVM boundary decisions for us.


def plot_svc_decision_function(clf, ax=None):
    '''Plot the decision function for a 2D SVC'''
    if ax is None:
        ax = plt.gca()
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
    Y, X = np.meshgrid(y, x)
    P = np.zeros_like(X)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            P[i, j] = clf.decision_function([xi, yj])
    # Plot the margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
# We now plot the optimal separation line together with the margins supported
# by some of the training data at the boundary.
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf)
plt.show()
# Note that a couple of the points touch the lines, these are known as our
# support vectors.
print clf.support_vectors_
# Visually check the concordance between the coordinates printed above
# and the highlighted points in the figure.
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=200, facecolors='none')
plt.show()
# Only the support vectors matter with an SVM. Moving points wihtout letting
# them cross the decision boundaries, it would have no effect.
# The SVM becomes more powerful in conjunction with kernels. Let's look
# at some data which is not linearly seperable.
x, y = make_circles(100, factor=.1, noise=.1)
clf = SVC(kernel='linear').fit(x, y)

plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf)
plt.show()
# Clearly no linear separation is going to work on this data.
# One way we can adjust to this data is to apply a kernel, which is some
# transformation of the input data. We could use the radial basis function.
r = np.exp(-(x[:, 0] ** 2 + x[:, 1] ** 2))
# if we plot this alongside our data, we can see the effect.


def plot_3D(elev=30, azim=30):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(x[:, 0], x[:, 1], r, c=y, s=50, cmap='spring')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
plot_3D()
plt.show()
# We can see that with this additional dimension, the data becomes trivially
# linearly seperable.
clf = SVC(kernel='rbf')
clf.fit(x, y)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=200, facecolors='none')
plt.show()
# Here there are effectively N basis functions. One centered at each point.
# Through a clever mathematical trick, this computation executes very
# efficiently using the 'kernel trick', and cleanly seperates the non-linearly
# seperable problem into the two distinct classes.
