from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import decomposition
import scipy.stats as stats
import random
from sklearn.neighbors import KNeighborsClassifier
import math
import pandas as pd


iris = datasets.load_iris()
'''
#1. Petal Length X Sepal Width for Setosa and Versicolor
plt.scatter(iris.data[:, 1], iris.data[:, 2], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])

#The first 100 observations correspond to setosa and versicolor
plt.scatter(iris.data[0:100, 1], iris.data[0:100, 2], c=iris.target[0:100])
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])

from sklearn import svm
svc = svm.SVC(kernel='linear', C=1) #The > C, the wider the margin between groups
from sklearn import datasets
X = iris.data[0:100, 1:3]
y = iris.target[0:100]
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

#2. Petal Length X Sepal Width for Versicolor and Virginica
plt.scatter(iris.data[:, 1], iris.data[:, 2], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])

#The first 100 observations correspond to setosa and versicolor
plt.scatter(iris.data[50:150, 1], iris.data[50:150, 2], c=iris.target[50:150])
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])

from sklearn import svm
svc = svm.SVC(kernel='linear', C=1) #The > C, the wider the margin between groups
from sklearn import datasets
X = iris.data[50:150, 1:3]
y = iris.target[50:150]
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
'''
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

###PCA###
pca = decomposition.PCA(n_components=2)
pca.fit(X)
xxx = pca.transform(X)
#print xxx

#4. Petal Length X Sepal Width for all 3
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
X = xxx #reassigning to PCA
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


####KNN####

data = pd.read_csv('iris.data', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

plt.suptitle('Petal Length x Width', fontsize=18)
plt.xlabel('Petal Length (cm)', fontsize=16)
plt.ylabel('Sepal Width (cm)', fontsize=16)
plt.scatter(xxx[:,0], xxx[:,1], alpha=0.5) #how to print all values, not just records 0 and 1? and how to compare w/original knn?
#plt.show() 

#random point
xx = random.uniform(4, 9)
yy = random.uniform(2, 5)
point = [(xx, yy)]
print point

y = data['class'] #class 
Xk = xxx #design matrix (normally)
#X = data[collist] = data[['sepal length', 'sepal width']] #design matrix (normally)

clf = KNeighborsClassifier(n_neighbors=10, weights = 'distance')
clf.fit(Xk, y)
preds = clf.predict(point)
print preds
plt.show()
