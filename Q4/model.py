"""
Module for doing the training for logistic regression
"""

from __future__ import division
import numpy as np
import math
import plot as plot
from numpy import matrix
from numpy import linalg


def trainGDA (train_X, train_Y):
    """
    Trains the logistic regression model, using Newton's method
    Theta = Theta - H-1(Theta) * grad(Theta)
    """

    X_matrix = np.matrix(train_X)
    Y_matrix = np.matrix(train_Y).T
    print (X_matrix.shape, Y_matrix.shape)

    phi = computePhi (Y_matrix)
    mu0 = computeMu (X_matrix, Y_matrix, classLabel=0)
    mu1 = computeMu (X_matrix, Y_matrix, classLabel=1)
    Sigma = computeSig (X_matrix, Y_matrix, mu0, mu1)

    print (mu0, mu1, Sigma)
    plot.linearGDAPlot(train_X, train_Y, np.matrix(mu0).T, np.matrix(mu1).T, np.matrix(Sigma), np.matrix(Sigma))


def computePhi (train_Y):
    """
    Computes the phi of the GDA
    """
    m = train_Y.shape[0]
    class_1 = np.sum(train_Y == 1)
    return class_1 / m


def computeMu (train_X, train_Y, classLabel):
    """
    Computes the means of the GDA
    """
    m = train_Y.shape[0]
    classIndicator = np.array(train_Y == classLabel).flatten()
    classCount = np.sum(classIndicator == True)
    train_X = np.array(train_X)

    classSum = [0, 0]
    for (indicator, x) in zip(classIndicator, train_X):
        if indicator:
            classSum = classSum + x

    mu = classSum / classCount
    return np.array(mu)


def computeSig (train_X, train_Y, mu0, mu1):
    """
    Computes the means of the GDA
    """
    m = train_Y.shape[0]
    classIndicator = np.array(train_Y == 0).flatten()

    Sigma = np.matrix([[0, 0], [0, 0]])
    for (indicator, x) in zip(classIndicator, train_X):
        if indicator:
            # Class 0
            # print (indicator, x-mu0, Sigma, (x - mu0).T * ((x - mu0)))
            Sigma = Sigma + (x - mu0).T * ((x - mu0))
        else:
            # Class 1
            # print (indicator, x-mu1, Sigma,  (x - mu1).T * ((x - mu1)))
            Sigma = Sigma + (x - mu1).T * (x - mu1)

    Sigma = Sigma / m
    print (Sigma)
    return np.array(Sigma)
