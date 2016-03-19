from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
import scipy.stats as stats
import random
from sklearn.neighbors import KNeighborsClassifier
import math
import pandas as pd
from sklearn.lda import LDA


iris = datasets.load_iris()

#3. Petal Length X Sepal Width for all 3
plt.scatter(iris.data[:, 1], iris.data[:, 2], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])

#The first 100 observations correspond to setosa and 
#a = iris.data[0:50].append(iris.data[100:150])

plt.scatter(iris.data[0:150, 1], iris.data[0:150, 2], c=iris.target[0:150])
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])

from sklearn import svm
svc = svm.SVC(kernel='linear', C=10) #The > C, the wider the margin between groups
from sklearn import datasets
X = iris.data[0:150, 1:3]
y = iris.target[0:150]
svc.fit(X, y)

#Adapted from https://github.com/jakevdp/sklearn_scipy2013
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()

plot_estimator(svc, X, y)
plt.show()
plt.clf()

###LDA###
sklearn_lda = LDA(n_components=2)
X_lda_sklearn = sklearn_lda.fit_transform(X, y)
print X_lda_sklearn

#4. Petal Length X Sepal Width for all 3
plt.scatter(iris.data[:, 1], iris.data[:, 2], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])

plt.scatter(iris.data[0:150, 1], iris.data[0:150, 2], c=iris.target[0:150])
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])

from sklearn import svm
svc = svm.SVC(kernel='linear', C=10) #The > C, the wider the margin between groups
from sklearn import datasets
X = X_lda_sklearn #reassigning to LDA
y = iris.target[0:150]
svc.fit(X, y)

#Adapted from https://github.com/jakevdp/sklearn_scipy2013
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()

plot_estimator(svc, X, y)
plt.show()
