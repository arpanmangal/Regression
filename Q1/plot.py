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
    plt.show()


def regressionPlot (X, Y, slope, intercept, Xlabel="X", Ylabel="Y", marker="ro"):
    """
    Plots Y vs X as scatter plot, joined by the line
    """
    plt.figure(1)
    print(X.shape, Y.shape)
    print (slope, intercept)
    plt.plot(X, Y, marker)
    plt.plot(X, X*slope + intercept, 'r')
    # abline(slope, intercept)
    plt.show()


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    print (x_vals.shape, y_vals[0,:].shape)
    print (x_vals, y_vals[0,:])
    plt.plot(x_vals, y_vals[0,:], '--')