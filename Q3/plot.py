"""
Module to plot our data
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def scatterPlot (X, Y, Xlabel="X", Ylabel="Y", marker="ro", fileName="Q3/plots/raw.png", title="Scatter Plot"):
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
    cir = plt.plot(circles[:,0], circles[:,1], 'go', label="0")
    cro = plt.plot(crosses[:,0], crosses[:,1], 'rx', label="1")
    plt.suptitle(title)
    plt.title("Please close the figure for next figure", fontsize=6)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.legend()
    fig.savefig(fileName)
    plt.show()


def logisticPlot (X, Y, slope, intercept, Xlabel="X", Ylabel="Y", marker="ro", fileName="Q3/plots/raw.png", title="Scatter Plot"):
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

    fig = plt.figure(2)
    cir = plt.plot(circles[:,0], circles[:,1], 'go', label="0")
    cro = plt.plot(crosses[:,0], crosses[:,1], 'rx', label="1")
    plt.plot(X[:,0], X[:,0]*slope + intercept, 'b')
    plt.suptitle(title)
    plt.title("Please close the figure for next figure", fontsize=6)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.legend()
    fig.savefig(fileName)
    plt.show()
