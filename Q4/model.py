"""
Module for doing the training for logistic regression
"""

from __future__ import division
import numpy as np
import math
import plot as plot
from numpy import matrix
from numpy import linalg


def trainGDA (train_X, train_Y, mode=1):
    """
    Trains the logistic regression model, using Newton's method
    Theta = Theta - H-1(Theta) * grad(Theta)
    """

    X_matrix = np.matrix(train_X)
    Y_matrix = np.matrix(train_Y).T

    phi = float(computePhi (Y_matrix))
    mu0 = computeMu (X_matrix, Y_matrix, classLabel=0)
    mu1 = computeMu (X_matrix, Y_matrix, classLabel=1)

    if mode == 0:
        # Common Sigma
        Sigma = computeSig (X_matrix, Y_matrix, np.matrix(mu0).T, np.matrix(mu1).T)
        ## Print the result
        sigma = np.round(Sigma, decimals=3).tolist()
        print ("PHI: %.2f | Mu0: %s | Mu1: %s | Sigma: %s" % (phi, mu0, mu1, sigma) )
        ## Plot the result
        plot.GDAPlot(train_X, train_Y, phi, mu0, mu1, Sigma, Sigma, fileName="Q4/plots/linearGDA.png", title="GDA Linear Boundary")
    else:
        # Different Sigma
        Sigma0 = computeDiffSig (X_matrix, Y_matrix, np.matrix(mu0).T, 0)
        Sigma1 = computeDiffSig (X_matrix, Y_matrix, np.matrix(mu1).T, 1)
        ## Print the result
        sigma0 = np.round(Sigma0, 3).tolist()
        sigma1 = np.round(Sigma1, 3).tolist()
        print ("PHI: %.3f | Mu0: %s | Mu1: %s | Sigma0: %s | Sigma1: %s" % (phi, mu0, mu1, sigma0, sigma1) )
        ## Plot the result
        plot.GDAPlot(train_X, train_Y, phi, mu0, mu1, Sigma0, Sigma1, fileName="Q4/plots/quadraticGDA.png", title="GDA Quadratic Boundary")


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
        x = x.T
        if indicator:
            # Class 0
            Sigma = Sigma + (x - mu0) * (x - mu0).T
        else:
            # Class 1
            Sigma = Sigma + (x - mu1) * (x - mu1).T

    Sigma = Sigma / m
    return np.matrix(Sigma)


def computeDiffSig (train_X, train_Y, mu, classLabel):
    """
    Computes the means of the GDA
    """
    m = train_Y.shape[0]
    classIndicator = np.array(train_Y == classLabel).flatten()
    classCount = np.sum(classIndicator == True)

    Sigma = np.matrix([[0, 0], [0, 0]])
    for (indicator, x) in zip(classIndicator, train_X):
        x = x.T
        if indicator:
            # Class 0
            Sigma = Sigma + (x - mu) * (x - mu).T

    Sigma = Sigma / classCount
    return np.matrix(Sigma)