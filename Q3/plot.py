"""
Module to plot our data
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def scatterPlot (X, Y, Xlabel="X", Ylabel="Y", marker="ro", fileName="plots/raw.png", title="Scatter Plot"):
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
    plt.plot(circles[:,0], circles[:,1], 'b.')
    plt.plot(crosses[:,0], crosses[:,1], 'rx')
    plt.suptitle(title)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    fig.savefig(fileName)
    plt.show()


def logisticPlot (X, Y, slope, intercept, Xlabel="X", Ylabel="Y", marker="ro", fileName="plots/raw.png", title="Scatter Plot"):
    """
    Plots Y vs X, as well as the logistic line
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
    plt.plot(circles[:,0], circles[:,1], 'b.')
    plt.plot(crosses[:,0], crosses[:,1], 'rx')
    plt.plot(X[:,0], X[:,0]*slope + intercept, 'g')
    plt.suptitle(title)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    fig.savefig(fileName)
    plt.show()