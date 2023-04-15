import re
from turtle import distance
import numpy as np


def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''

    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    rand_indices = np.random.randint(0, X.shape[0], k)
    centroids = X[rand_indices]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float)


def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    centroids_tensor = np.expand_dims(centroids, 1)
    distances = ((np.abs(X - centroids_tensor)**p).sum(axis=2)**(1/p))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances


def kmeans(X, k, p, max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = np.array([])
    centroids = get_random_centroids(X, k)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    iter = 0
    classes_hist = [np.ones(X.shape[0])]
    while iter < max_iter and np.any(classes != classes_hist[-1]):
        classes_hist.append(classes)
        distances = lp_distance(X, centroids, p)
        classes = distances.argmin(axis=0)
        for c in range(len(centroids)):
            centroids[c] = X[classes == c].mean(axis=0)
        iter += 1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes


def kmeans_pp(X, k, p, max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = np.array([])
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # probs = np.ones(filtered_data.shape[0])
    filtered_data = X.copy()
    centroid = X[np.random.choice(X.shape[0], 1, replace=False), :]
    centroids.append(centroid.flatten())
    for _ in range(k):
        centroid = centroids[-1]
        filtered_data = filtered_data[np.all(
            filtered_data != centroid, axis=1), :]
        distances = lp_distance(filtered_data, centroids, p).min(axis=0)
        probs = distances**2 / (distances**2).sum()
        centroid = filtered_data[np.random.choice(
            filtered_data.shape[0], 1, replace=False, p=probs.flatten()), :]
        centroids.append(centroid.flatten())

    # continue with Kmeans
    iter = 0
    classes_hist = [np.ones(X.shape[0])]
    while iter < max_iter and np.any(classes != classes_hist[-1]):
        classes_hist.append(classes)
        distances = lp_distance(X, centroids, p)
        classes = distances.argmin(axis=0)
        for c in range(len(centroids)):
            centroids[c] = X[classes == c].mean(axis=0)
        iter += 1
    centroids = np.array(centroids)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes
