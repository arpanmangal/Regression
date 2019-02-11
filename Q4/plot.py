"""
Module to plot our data
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math

def scatterPlot (X, Y, Xlabel="X", Ylabel="Y", marker="ro", fileName="Q4/plots/raw.png", title="Scatter Plot"):
    """
    Plots Y vs X as scatter plot
    """
    circles = []
    crosses = []
    for x, label in zip(X, Y):
        if label == 0:
            circles.append(x)
        else:
            crosses.append(x)
    circles = np.array(circles)
    crosses = np.array(crosses)

    fig = plt.figure(1)
    cir = plt.plot(circles[:,0], circles[:,1], 'go', label="Alaska (0)")
    cro = plt.plot(crosses[:,0], crosses[:,1], 'rx', label="Canada (1)")
    plt.suptitle(title)
    plt.title("Please close the figure for next figure", fontsize=6)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.legend()
    fig.savefig(fileName)
    plt.show()


def GDAPlot (X, Y, Phi, Mu0, Mu1, Sigma0, Sigma1, Xlabel="X", Ylabel="Y", marker="ro", fileName="Q4/plots/guassian.png", title="Guassian Plot"):
    """
    Plots Y vs X as scatter plot
    """
    A = Sigma0_inv = np.matrix(np.linalg.inv(Sigma0))
    B = Sigma1_inv = np.matrix(np.linalg.inv(Sigma1))
    Mu0 = np.matrix(Mu0).T
    Mu1 = np.matrix(Mu1).T
    detA = np.linalg.det(Sigma0)
    detB = np.linalg.det(Sigma1)
    phi = Phi

    circles = []
    crosses = []
    for x, label in zip(X, Y):
        if label == 0:
            circles.append(x)
        else:
            crosses.append(x)
    circles = np.array(circles)
    crosses = np.array(crosses)

    fig = plt.figure(2)
    cir = plt.plot(circles[:,0], circles[:,1], 'go', label="Alaska (0)")
    cro = plt.plot(crosses[:,0], crosses[:,1], 'rx', label="Canada (1)")
    plt.plot(Mu0[0], Mu0[1], 'bP')
    plt.plot(Mu1[0], Mu1[1], 'bP')
    
    x = np.linspace(np.min(X[:,0]),np.max(X[:,0]),100)
    y = np.linspace(np.min(X[:,1]),np.max(X[:,1]),105)

    Z = np.zeros(shape=(y.shape[0], x.shape[0]))
    for i in range(y.shape[0]):
        for j in range(x.shape[0]):
            X = np.matrix([x[j], y[i]]).T
            LHS = (X-Mu0).T * Sigma0_inv * (X-Mu0)
            LHS = (1-phi) * np.exp(-0.5 * float(LHS)) / math.sqrt(detA)
            RHS = (X-Mu1).T * Sigma1_inv * (X-Mu1)
            RHS = (1-phi) * np.exp(-0.5 * float(RHS)) / math.sqrt(detB)
            Z[i][j] = LHS - RHS

    plt.contour(x,y,Z,[0])
    plt.suptitle(title)
    plt.title("Please close the figure for next figure", fontsize=6)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.legend()
    fig.savefig(fileName)
    plt.show()
