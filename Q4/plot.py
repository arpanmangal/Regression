"""
Module to plot our data
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def scatterPlot (X, Y, Xlabel="X", Ylabel="Y", marker="ro", fileName="Q4/plots/raw.png", title="Scatter Plot"):
    """
    Plots Y vs X as scatter plot
    """
    circles = []
    crosses = []
    for x, label in zip(X, Y):
        if label == 1:
            circles.append(x)
        else:
            crosses.append(x)
    circles = np.array(circles)
    crosses = np.array(crosses)

    fig = plt.figure(1)
    cir = plt.plot(circles[:,0], circles[:,1], 'go', label="Alaska (1)")
    cro = plt.plot(crosses[:,0], crosses[:,1], 'rx', label="Canada (0)")
    plt.suptitle(title)
    plt.title("Please close the figure for next figure", fontsize=6)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.legend()
    fig.savefig(fileName)
    plt.show()


def GDAPlot (X, Y, Mu0, Mu1, Sigma1, Sigma2, Xlabel="X", Ylabel="Y", marker="ro", fileName="Q4/plots/guassian.png", title="Guassian Plot"):
    """
    Plots Y vs X as scatter plot
    """
    Sigma1_inv = np.linalg.inv(Sigma1)
    Sigma2_inv = np.linalg.inv(Sigma2)

    circles = []
    crosses = []
    for x, label in zip(X, Y):
        if label == 1:
            circles.append(x)
        else:
            crosses.append(x)
    circles = np.array(circles)
    crosses = np.array(crosses)

    fig = plt.figure(2)
    cir = plt.plot(circles[:,0], circles[:,1], 'go', label="Alaska (1)")
    cro = plt.plot(crosses[:,0], crosses[:,1], 'rx', label="Canada (0)")
    plt.plot(Mu0[0], Mu0[1], 'bP')
    plt.plot(Mu1[0], Mu1[1], 'bP')

    x1 = np.linspace(50,180,100)
    x2 = np.linspace(300,500,100)

    Z = np.zeros(shape=(x1.shape[0], x2.shape[0]))
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            X = np.matrix([x1[i], x2[j]]).T
            Z[i][j] = (X-Mu0).T * Sigma1_inv * (X-Mu0) - (X-Mu1).T * Sigma2_inv * (X-Mu1)

    x1,x2 = np.meshgrid(x1, x2)
    plt.contour(x1,x2,Z,[0])
    plt.suptitle(title)
    plt.title("Please close the figure for next figure", fontsize=6)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.legend()
    fig.savefig(fileName)
    plt.show()
