"""
Module for doing the training
"""

from __future__ import division
import numpy as np
import math
import plot as plot
from numpy import matrix
from numpy import linalg


def trainUnweighted (train_X, train_Y):
    """
    Trains the unweighted linear regression model
    Theta = (X.T * X)-1 * X.T * Y
    """

    X_matrix = np.matrix(train_X).T
    Y_matrix = np.matrix(train_Y).T

    X_ones = np.ones((X_matrix.shape[0], 1))
    X_matrix = np.hstack((X_ones, X_matrix))

    return np.linalg.inv(X_matrix.T * X_matrix) * X_matrix.T * Y_matrix


def displayWeighted (train_X, train_Y, tau=0.8):
    """
    For each point trains the weighted model
    and plots a point
    """
    train_X.sort()
    h_theta = []
    for x_i in train_X:
        prediction = float(trainWeighted(train_X, train_Y, x_i, tau))
        h_theta.append(prediction)

    plot.linePlot(train_X, train_Y, h_theta, fileName=("plots/weighted-" + str(tau).replace('.','_')), title=("Weighted - Tau=" + str(tau)))


def trainWeighted (train_X, train_Y, x, tau):
    """
    Trains the weighted linear regression model, for a single query point
    Theta = (X.T * W * X)-1 * X.T * W * Y
    """

    X_matrix = np.matrix(train_X).T
    Y_matrix = np.matrix(train_Y).T
    x_matrix = np.matrix([1,x]).T

    X_ones = np.ones((X_matrix.shape[0], 1))
    X_matrix = np.hstack((X_ones, X_matrix))

    W_matrix = generateW (train_X, x, tau)

    # print (X_matrix.shape, W_matrix.shape, Y_matrix.shape)
    Theta = np.linalg.inv(X_matrix.T * W_matrix * X_matrix) * X_matrix.T * W_matrix * Y_matrix
    prediction = Theta.T * x_matrix
    return prediction


def generateW (train_X, x, tau):
    """
    Generate the diagonal weight matrix W for the training set train_X and query x, with parameter tau
    """

    weights = []
    for x_i in train_X:
        expo =( (x - x_i)**2) / (-2.0 * tau**2)
        weights.append(math.exp(expo))
    return np.diag(weights)
