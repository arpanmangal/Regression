"""
Module to plot our data
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def scatterPlot (X, Y, Xlabel="X", Ylabel="Y", marker="ro", fileName="Q2/plots/raw.png", title="Scatter Plot"):
    """
    Plots Y vs X as scatter plot
    """
    fig = plt.figure(1)
    plt.plot(X, Y, marker)
    plt.suptitle(title)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    fig.savefig(fileName)
    plt.show()


def regressionPlot (X, Y, slope, intercept, Xlabel="X", Ylabel="Y", marker="ro", linecolor="r", fileName="Q2/plots/regression.png", title="Regression"):
    """
    Plots Y vs X as scatter plot, joined by the line
    """
    fig = plt.figure(2)
    plt.plot(X, Y, marker)
    plt.suptitle(title)
    plt.plot(X, X*slope + intercept, linecolor)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    fig.savefig(fileName)
    plt.show()


def linePlot (X, Y, prediction, Xlabel="X", Ylabel="Y", marker="r-", fileName="Q2/plots/line.png", title="Line"):
    """
    Plots Y vs X as line plot
    """
    fig = plt.figure(3)
    plt.plot(X, Y, 'bx')
    plt.plot(X, prediction, marker)
    plt.suptitle(title)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    fig.savefig(fileName)
    plt.show()
