#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:16:51 2018

@author: ashwin
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets
from sklearn.decomposition import TruncatedSVD

def plot_embedding(X, y,title=None):
    # method for visualizing the MNIST representations
    digits = datasets.load_digits(n_class=10)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()
       
def visualize(X,labels,data='cifar'):
    # method to visualize the t-SNE
    tsne = manifold.TSNE(n_components=2, init='pca',random_state=0, perplexity=40, verbose=2)
    Y = tsne.fit_transform(X)
    if data == 'cifar':
        plt.scatter(Y[:, 0], Y[:, 1], c=labels)
        plt.colorbar()
        plt.show()
    else:
        plot_embedding(Y,labels, "t-SNE embedding of the digits ")

if __name__ == '__main__':
    # load both the datasets 
    X = np.load(open('data/cifar_10_x1d_train.npy'))
    y = np.load(open('data/cifar_10_y_train.npy'))
    X = X[:2000]
    y = y[:2000]

    # Do SVD to reduce dimensionality of CIFAR and then t-SNE
    svd = TruncatedSVD(n_components=32)
    U = svd.fit_transform(X)
    digits = datasets.load_digits(n_class=10)
    visualize(U,y,data='cifar')
    
    # t-SNE on MNIST raw data
    digits = datasets.load_digits(n_class=10)
    X = digits.data
    y = digits.target
    visualize(X,y,data='mnist')
    