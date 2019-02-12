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
    converged = None
    while (True):
        Theta = SGD(X_matrix, Y_matrix, Theta, eta, 20)
        cost = compute_cost(X_matrix, Y_matrix, Theta)
        # print ('Epoch: %d | Cost: %.8f | Theta: %f, %f' % (epoch, cost, Theta[0,0], Theta[1,0]) )
        epoch = epoch + 1
        # Stopping condition
        if (float(Theta[0]) >= 10000 or float(Theta[1]) >= 10000 or (cost is float('inf')) or (epoch > 6 and cost > 1.5 * old_cost)): # 50 iters for the purpose of plotting
            # Diverged
            print ("The model is diverging :(, please change the learning rate :)")
            converged = False
            break
        if (epoch > 5000):
            # too slow
            print ("The learning rate is too small :(. Stopping since taking >5000 epochs. To train faster use eta close to 1...")
            converged = False
            break

        if (epoch > 20 and cost < 1e-5 and abs(old_cost - cost) / old_cost < 0.0001):
            # Change in cost is less than 0.1% => Stop
            converged = True
            break
        costData.append( (float(cost), float(Theta[0, 0]), float(Theta[1,0]) / 10.0) )
        old_cost = cost

    Theta[1,:] = Theta[1,:] / 10
    X_matrix[:,1] = X_matrix[:,1] * 10
    
    # Print result
    print("Theta0: %.6f | Theta1: %.6f | #Epochs: %d" % (Theta[0], Theta[1], epoch))

    plot.regressionPlot(train_X, train_Y, Theta[1,0], Theta[0,0], Xlabel="Acidity", Ylabel="Density of Wine", marker="bx", fileName="Q1/plots/curve.png")
    animatedDesent(X_matrix, Y_matrix, np.array(costData), delay, converged=converged)
    return Theta


def animatedDesent (X_matrix, Y_matrix, costData, delay=0.2, converged=True):
    """
    Plots the J(Theta) curve in 3D space and contours
    Shows the real time gradient descent
    """
    theta0s = np.linspace(0.3, 1.7, 100)
    theta1s = np.linspace(-0.080, 0.080, 105)

    if (not converged):
        theta0s = np.linspace(-2000, 2000, 100)
        theta1s = np.linspace(-2000, 2000, 100)

    costMatrix = np.zeros((len(theta1s), len(theta0s)))
    for i in range(len(theta1s)):
        for j in range(len(theta0s)):
            Theta = np.matrix([theta0s[j], theta1s[i]]).T
            # compute cost
            costMatrix[i][j] = compute_cost(X_matrix, Y_matrix, Theta)

    plot.costPlot(theta0s, theta1s, costMatrix, costData, delay=delay, Xlabel="Theta 0", Ylabel="Theta 1", Zlabel="Cost")
    plot.contourPlot(theta0s, theta1s, costMatrix, costData, delay=delay, converged=converged, Xlabel="Theta 0", Ylabel="Theta 1", Zlabel="Cost")


def SGD (X, Y, Theta, eta, batch_size=20):
    """
    Computes one epoch of the batch gradient decent algorithm
    """

    Theta = Theta - eta * compute_gradient(X, Y, Theta)

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
