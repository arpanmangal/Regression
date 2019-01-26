"""
Module to plot our data
"""

import matplotlib.pyplot as plt
import numpy as np

def scatterPlot (X, Y, Xlabel="X", Ylabel="Y", marker="ro"):
    """
    Plots Y vs X as scatter plot
    """
    plt.plot(X, Y, marker)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.show()


def regressionPlot (X, Y, slope, intercept, Xlabel="X", Ylabel="Y", marker="ro", linecolor="r", fileName="plots/test.png"):
    """
    Plots Y vs X as scatter plot, joined by the line
    """
    fig = plt.figure(1)
    plt.plot(X, Y, marker)
    plt.plot(X, X*slope + intercept, linecolor)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    fig.savefig(fileName)
    plt.show()
