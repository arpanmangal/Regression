"""
Module for doing the training for logistic regression
"""

from __future__ import division
import numpy as np
import math
import plot as plot
from numpy import matrix
from numpy import linalg


def trainLogistic (train_X, train_Y):
    """
    Trains the logistic regression model, using Newton's method
    Theta = Theta - H-1(Theta) * grad(Theta)
    """

    X_matrix = np.matrix(train_X)
    Y_matrix = np.matrix(train_Y).T

    X_ones = np.ones((X_matrix.shape[0], 1))
    X_matrix = np.hstack((X_ones, X_matrix))
    # print (X_matrix.shape, Y_matrix.shape)

    Theta = np.matrix([0, 0, 0]).T # (0, 0, 0).T
    # for itr in range(1):
    #     # Do one iteration of Newton's Method
    #     print (itr)
    #     grad = gradient(X_matrix, Y_matrix, Theta)
    #     if (np.linalg.norm(grad) == 0):
    #         break
    #     H_inverse = np.linalg.inv( hessian(X_matrix, Y_matrix, Theta) )
    #     Theta = Theta - H_inverse * grad

    # Do one iteration of Newton's Method
    grad = gradient(X_matrix, Y_matrix, Theta)
    H_inverse = np.linalg.inv( hessian(X_matrix, Y_matrix, Theta) )
    Theta = Theta - H_inverse * grad
    
    return Theta


def gradient (X, Y, Theta):
    """
    Find the gradient (first derivative) of L(Theta) at Theta
    """
    return X.T * (Y - sigma(X*Theta))


def hessian (X, Y, Theta):
    """
    Computes the Hessian (second derivative) of L(Theta) at Theta
    Vectorised form: X.T * D * X
    where D = diag( sig(X*Theta) * (1 - sig(X*Theta)) )
    """
    sigXTheta = np.array(sigma (X * Theta))[:,0]
    D = np.matrix( np.diag( sigXTheta * (1 - sigXTheta) ))
    return X.T * D *  X


def sigma (Z):
    """
    Computes sigma(Z) vectorised
    """
    return 1 / (1 + np.exp(-Z))
