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
        print ('Epoch: %d | Cost: %.7f | Theta: %f, %f' % (epoch, cost, Theta[0,0], Theta[1,0]) )
        epoch = epoch + 1
        costData.append( (float(cost), float(Theta[0, 0]), float(Theta[1,0]) / 10.0) )
        # Stopping condition
        if (epoch > 20 and cost < 1e-5 and abs(old_cost - cost) / old_cost < 0.01):
            # Change in cost is less than 1% => Stop
            break
        old_cost = cost

    # print (costData)
    Theta[1,:] = Theta[1,:] / 10
    X_matrix[:,1] = X_matrix[:,1] * 10
    # plot.regressionPlot(train_X, train_Y, Theta[1,0], Theta[0,0], "Acidity", "Density of Wine", "bx", fileName="Q1/plots/curve.png")
    bowlCurve(X_matrix, Y_matrix, np.array(costData), delay)
    # if (type == 'curve'):
    # elif (type == 'bowl'):
    return Theta


def bowlCurve (X_matrix, Y_matrix, costData, delay=0.2):
    """
    Plots the J(Theta) curve in 3D space 
    Shows the real time gradient descent
    """

    # theta0s 0-1.2 ~100 points
    # theta0s = ((np.arange(20) / 2.0)) / 100
    # theta0s = np.arange(200) * 2.4 / 200 - 2.2
    # theta0s = np.arange(0.8, 1.2, 0.0001)
    # theta1s = np.arange(0.08, 0.11, 0.00001)
    # -0.1 0.1
    # 0.9 1.1
    # theta0s = np.linspace(0.9, 1.1, 100)
    # theta1s = np.linspace(-0.1, 0.1, 105)
    theta0s = np.linspace(0.4, 1.6, 100)
    theta1s = np.linspace(-0.050, 0.050, 105)
    # theta0s = theta0s + 0.94
    # theta1s 0.000-0.03 ~200 points
    # theta1s = np.arange(200) * 0.06 / 200 #+ .01
    # theta1s = ((np.arange(20) / 2.0) + 5) / 1000
    print (theta0s.shape, theta1s.shape)
    # exit()

    # costMatrix = np.zeros((len(theta0s), len(theta1s)))
    # for i in range(len(theta0s)):
    #     for j in range(len(theta1s)):
    #         Theta = np.matrix([theta0s[i], theta1s[j]]).T
    #         # compute cost
    #         costMatrix[i][j] = compute_cost(X_matrix, Y_matrix, Theta)

    costMatrix = np.zeros((len(theta1s), len(theta0s)))
    for i in range(len(theta1s)):
        for j in range(len(theta0s)):
            Theta = np.matrix([theta0s[j], theta1s[i]]).T
            # compute cost
            costMatrix[i][j] = compute_cost(X_matrix, Y_matrix, Theta)

    print (costMatrix.shape)
    # exit()
    # print (costMatrix)
    # plot.costPlot(theta0s, theta1s, costMatrix, Xlabel="Theta 0", Ylabel="Theta 1", Zlabel="Cost")
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
