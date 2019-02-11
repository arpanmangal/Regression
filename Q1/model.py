"""
Module for doing the training
"""

from __future__ import division
import numpy as np
import plot as plot
from numpy import matrix
from numpy import linalg


def train (train_X, train_Y, learning_rate=1, delay=0.2, type="curve"):
    """
    Trains the linear regression model on training data
    Sets the hyperparameter according to the validation data
    and final accuracy on test data
    """

    Theta = np.matrix([0.0, 0.0]).T
    X_matrix = np.matrix(train_X).T
    Y_matrix = np.matrix(train_Y).T

    X_ones = np.ones((X_matrix.shape[0], 1))
    X_matrix = np.hstack((X_ones, X_matrix))

    # normalisation (approximate)
    X_matrix[:,1] = X_matrix[:,1] / 10

    epoch = 0
    eta = learning_rate
    costData = [] # for plotting real time data
    old_cost = 0
    while (True):
        Theta = SGD(X_matrix, Y_matrix, Theta, eta, 20)
        cost = compute_cost(X_matrix, Y_matrix, Theta)
        # print ('Epoch: %d | Cost: %.7f | Theta: %f, %f' % (epoch, cost, Theta[0,0], Theta[1,0]) )
        epoch = epoch + 1
        costData.append( (float(cost), float(Theta[0, 0]), float(Theta[1,0]) / 10.0) )
        # Stopping condition
        if (epoch > 20 and cost < 1e-5 and abs(old_cost - cost) / old_cost < 0.01):
            # Change in cost is less than 1% => Stop
            break
        old_cost = cost

    Theta[1,:] = Theta[1,:] / 10
    X_matrix[:,1] = X_matrix[:,1] * 10
    
    # Print result
    print("Theta0: %.6f | Theta1: %.6f | #Epochs: %d" % (Theta[0], Theta[1], epoch))

    plot.regressionPlot(train_X, train_Y, Theta[1,0], Theta[0,0], Xlabel="Acidity", Ylabel="Density of Wine", marker="bx", fileName="Q1/plots/curve.png")
    animatedDesent(X_matrix, Y_matrix, np.array(costData), delay)
    return Theta


def animatedDesent (X_matrix, Y_matrix, costData, delay=0.2):
    """
    Plots the J(Theta) curve in 3D space and contours
    Shows the real time gradient descent
    """
    theta0s = np.linspace(0.4, 1.6, 100)
    theta1s = np.linspace(-0.050, 0.050, 105)

    costMatrix = np.zeros((len(theta1s), len(theta0s)))
    for i in range(len(theta1s)):
        for j in range(len(theta0s)):
            Theta = np.matrix([theta0s[j], theta1s[i]]).T
            # compute cost
            costMatrix[i][j] = compute_cost(X_matrix, Y_matrix, Theta)

    plot.costPlot(theta0s, theta1s, costMatrix, costData, delay=delay, Xlabel="Theta 0", Ylabel="Theta 1", Zlabel="Cost")
    plot.contourPlot(theta0s, theta1s, costMatrix, costData, delay=delay, Xlabel="Theta 0", Ylabel="Theta 1", Zlabel="Cost")


def SGD (X, Y, Theta, eta, batch_size=20):
    """
    Computes one epoch of the batch gradient decent algorithm
    """

    (m, n) = X.shape
    start = 0
    while (start < m):
        end = start + batch_size
        Theta = Theta - eta * compute_gradient(X[start:end,:], Y[start:end,:], Theta)
        start = end

    return Theta


def compute_gradient (X, Y, Theta):
    """
    Computes the cost gradient
    X = m*n
    Y = m*1
    Theta = n*1
    gradient = (1/m) * X_transpose * (X*Theta - Y) 
    """
    (m, n) = X.shape
    return (1.0/m) * (X.T) * (X*Theta - Y)


def compute_cost (X, Y, Theta):
    """
    Computes the J(Theta)
    X = m*n
    Y = m*1
    Theta = n*1
    Cost = (1/2m) * (Y-X*Theta)_tran * (Y-X*Theta)
    """

    (m, n) = X.shape
    error = Y - X*Theta
    return (0.5 / m) * error.T * error
